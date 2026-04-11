import warnings
import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


def get_num_transfer_tokens(mask_index, steps):
	mask_num = mask_index.sum(dim=1, keepdim=True)
	base = mask_num // steps
	remainder = mask_num % steps
	num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
	for i in range(mask_num.size(0)):
		num_transfer_tokens[i, :remainder[i]] += 1
	return num_transfer_tokens


def top_p_logits(logits, top_p=None):
	sorted_logits, sorted_indices = torch.sort(logits, descending=True)
	cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
	sorted_indices_to_remove = cumulative_probs > top_p
	sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
	sorted_indices_to_remove[..., 0] = 0
	mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
	mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
	logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
	return logits


def top_k_logits(logits, top_k=None):
	top_k = min(top_k, logits.size(-1))
	indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
	logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
	return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None,
				  margin_confidence=False, neg_entropy=False):
	if temperature > 0:
		logits = logits / temperature
	if top_p is not None and top_p < 1:
		logits = top_p_logits(logits, top_p)
	if top_k is not None:
		logits = top_k_logits(logits, top_k)
	probs = torch.softmax(logits, dim=-1)
	if temperature > 0:
		try:
			x0 = dists.Categorical(probs=probs).sample()
			confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
		except:
			confidence, x0 = probs.max(dim=-1)
	else:
		confidence, x0 = probs.max(dim=-1)
	if margin_confidence:
		sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
		confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
	if neg_entropy:
		epsilon = 1e-10
		log_probs = torch.log(probs + epsilon)
		confidence = torch.sum(probs * log_probs, dim=-1)
	return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
	sequences: torch.LongTensor = None
	history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
	def __init__(self, **kwargs):
		self.temperature: float = kwargs.pop("temperature", 0.0)
		self.top_p: Optional[float] = kwargs.pop("top_p", None)
		self.top_k: Optional[int] = kwargs.pop("top_k", None)
		self.max_length = kwargs.pop("max_length", 20)
		self.max_new_tokens = kwargs.pop("max_new_tokens", None)
		self.eps: float = kwargs.pop("eps", 1e-3)
		self.steps: int = kwargs.pop("steps", 512)
		self.alg: str = kwargs.pop("alg", 'origin')
		self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
		self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
		self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
		self.output_history: bool = kwargs.pop("output_history", False)
		self.mask_token_id = kwargs.pop("mask_token_id", None)
		self.pad_token_id = kwargs.pop("pad_token_id", None)
		self.bos_token_id = kwargs.pop("bos_token_id", None)
		self.eos_token_id = kwargs.pop("eos_token_id", None)
		self.generation_kwargs = kwargs.pop("generation_kwargs", {})
		self._from_model_config = kwargs.pop("_from_model_config", False)
		self._commit_hash = kwargs.pop("_commit_hash", None)
		self.transformers_version = kwargs.pop("transformers_version", __version__)
		if not self._from_model_config:
			for key, value in kwargs.items():
				try:
					setattr(self, key, value)
				except AttributeError as err:
					logger.error(f"Can't set {key} with value {value} for {self}")
					raise err
		self.validate(is_init=True)

	def validate(self, is_init=False):
		pass


class DreamGenerationMixin:
	@staticmethod
	def _expand_inputs_for_generation(expand_size=1, input_ids=None, attention_mask=None):
		if expand_size == 1:
			return input_ids, attention_mask
		if input_ids is not None:
			input_ids = input_ids.repeat_interleave(expand_size, dim=0)
		if attention_mask is not None:
			attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
		return input_ids, attention_mask

	def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
		if is_torchdynamo_compiling():
			return
		if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
			warnings.warn("Using default max_length.", UserWarning)
		if input_ids_length >= generation_config.max_length:
			raise ValueError(f"Input length {input_ids_length} >= max_length {generation_config.max_length}.")

	def _prepare_generated_length(self, generation_config, has_default_max_length, input_ids_length):
		if generation_config.max_new_tokens is not None:
			generation_config.max_length = generation_config.max_new_tokens + input_ids_length
		elif has_default_max_length:
			if generation_config.max_length == DreamGenerationConfig().max_length:
				generation_config.max_length = generation_config.max_length + input_ids_length
				max_pos = getattr(self.config, "max_position_embeddings", None)
				if max_pos is not None:
					generation_config.max_length = min(generation_config.max_length, max_pos)
		return generation_config

	def _prepare_generation_config(self, generation_config, **kwargs):
		using_model_generation_config = False
		if generation_config is None:
			generation_config = DreamGenerationConfig.from_model_config(self.config)
			using_model_generation_config = True
		if not is_torchdynamo_compiling():
			generation_config = copy.deepcopy(generation_config)
			generation_config.update(**kwargs)
			if not using_model_generation_config:
				if generation_config.bos_token_id is None:
					generation_config.bos_token_id = self.generation_config.bos_token_id
				if generation_config.eos_token_id is None:
					generation_config.eos_token_id = self.generation_config.eos_token_id
				if generation_config.pad_token_id is None:
					generation_config.pad_token_id = self.generation_config.pad_token_id
				if generation_config.mask_token_id is None:
					generation_config.mask_token_id = self.generation_config.mask_token_id
		return generation_config

	def _prepare_special_tokens(self, generation_config, device=None):
		def _tensor_or_none(token, device=None):
			if token is None:
				return token
			device = device if device is not None else self.device
			if isinstance(token, torch.Tensor):
				return token.to(device)
			return torch.tensor(token, device=device, dtype=torch.long)
		generation_config._bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
		eos = _tensor_or_none(generation_config.eos_token_id, device=device)
		if eos is not None and eos.ndim == 0:
			eos = eos.unsqueeze(0)
		generation_config._eos_token_tensor = eos
		pad = _tensor_or_none(generation_config.pad_token_id, device=device)
		if pad is None and eos is not None:
			pad = eos[0]
		generation_config._pad_token_tensor = pad
		generation_config._mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

	@torch.no_grad()
	def diffusion_generate(self, inputs=None, generation_config=None, **kwargs):
		generation_config = self._prepare_generation_config(generation_config, **kwargs)
		assert inputs is not None
		input_ids = inputs
		device = input_ids.device
		attention_mask = kwargs.pop("attention_mask", None)
		self._prepare_special_tokens(generation_config, device=device)

		input_ids_length = input_ids.shape[-1]
		has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
		generation_config = self._prepare_generated_length(
			generation_config=generation_config,
			has_default_max_length=has_default_max_length,
			input_ids_length=input_ids_length,
		)
		self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

		input_ids, attention_mask = self._expand_inputs_for_generation(
			expand_size=generation_config.num_return_sequences,
			input_ids=input_ids,
			attention_mask=attention_mask,
		)

		threshold = kwargs.get("threshold", 0.9)
		block_length = kwargs.get("block_length", 32)
		dual_cache = kwargs.get("dual_cache", False)
		constrained_decoder = kwargs.get("constrained_decoder", None)

		result = self._sample(
			input_ids, attention_mask=attention_mask,
			generation_config=generation_config,
			threshold=threshold, block_length=block_length,
			dual_cache=dual_cache,
			constrained_decoder=constrained_decoder,
		)
		return result


	def _cd_commit_position(self, _cd, x, seq_pos, b_pos, proposed_token, logits,
								temperature, mask_token_id):
		"""
		Commit a single token at seq_pos with constraint filtering.

		Returns:
			'constrained' - model's proposal was valid, committed directly
			'resampled'   - proposal invalid, resampled from constrained logits
			'unconstrained' - no left-segment neighbor, committed freely
			None          - could not commit (EOS/empty bytes)
		"""
		left_seg = _cd._seg_ending_at(seq_pos - 1)
		left_is_start = (seq_pos == _cd.mgr.gen_start)
		is_constrained = left_seg is not None or left_is_start

		if is_constrained:
			current_token = x[0, seq_pos].item()
			if current_token != mask_token_id and current_token != proposed_token:
				proposed_token = current_token  # validate the provisional
				
			# Check if model's proposal is DFA-valid
			if _cd.is_valid_at_position(seq_pos, proposed_token):
				tok_bytes = _cd.t2b.get(proposed_token, b'')
				if not tok_bytes:
					return None  # EOS at valid accept state — let it be
				x[0, seq_pos] = proposed_token
				_cd.commit_token(seq_pos, proposed_token)
				return 'constrained'
			else:
				# Resample from constrained distribution
				valid_mask = _cd.get_valid_mask(
					seq_pos, logits.device, logits_vocab_size=logits.shape[-1]
				)
				if valid_mask is None or not valid_mask.any():
					return None

				pos_logits = logits[0, b_pos].clone()
				pos_logits[~valid_mask] = float('-inf')
				if temperature > 0:
					probs = torch.softmax(pos_logits / temperature, dim=-1)
					chosen_token = torch.multinomial(probs, 1).item()
				else:
					chosen_token = pos_logits.argmax().item()

				chosen_bytes = _cd.t2b.get(chosen_token, b'')
				if not chosen_bytes:
					return None
				x[0, seq_pos] = chosen_token
				_cd.commit_token(seq_pos, chosen_token)
				return 'resampled'
		else:
			# No left-segment neighbor yet — commit anyway to create an isolated
			# segment. This segment will merge left when its neighbor is revealed.
			# The old provisional path skipped commit_token, breaking constraint
			# propagation for the LR backend (isolated tokens got no left context).
			tok_bytes = _cd.t2b.get(proposed_token, b'')
			if not tok_bytes:
				return None
			x[0, seq_pos] = proposed_token
			_cd.commit_token(seq_pos, proposed_token)   # always commit
			return 'provisional'


	def _sample(
			self, input_ids, attention_mask=None, generation_config=None,
			threshold=0.9, block_length=32, dual_cache=False,
			constrained_decoder=None,
		):
			output_history = generation_config.output_history
			return_dict_in_generate = generation_config.return_dict_in_generate
			max_length = generation_config.max_length
			mask_token_id = generation_config.mask_token_id
			steps = generation_config.steps
			temperature = generation_config.temperature
			top_p = generation_config.top_p
			top_k = generation_config.top_k
			alg = generation_config.alg
			alg_temp = generation_config.alg_temp

			histories = [] if (return_dict_in_generate and output_history) else None

			x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
			prompt_len = input_ids.shape[1]
			gen_length = max_length - prompt_len

			if block_length is None:
				block_length = gen_length
			assert gen_length % block_length == 0
			num_blocks = gen_length // block_length
			assert steps % num_blocks == 0
			steps_per_block = steps // num_blocks
			timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

			_cd = constrained_decoder

			if _cd is not None:
				print(f"[CD-INIT] gen_start={prompt_len}, gen_length={gen_length}, "
					f"num_blocks={num_blocks}, steps_per_block={steps_per_block}, "
					f"block_length={block_length}, alg={alg}", flush=True)

				_cd.gen_start = prompt_len
				_cd.gen_length = gen_length
				_cd.gen_end = prompt_len + gen_length - 1
				_cd.mgr.gen_start = prompt_len
				_cd.mgr.gen_length = gen_length
				_cd.mgr.gen_end = prompt_len + gen_length - 1
				_cd.mgr.reset()
				_cd._mask_cache.clear()

				if not _cd._state_mask_cache:
					print(f"[CD-INIT] Precomputing state masks for {_cd.dfa.num_states} states "
						f"× {len(_cd._nonempty_t2b)} tokens...", flush=True)
					t_pre = time.time()
					_cd.precompute_state_masks(x.device)
					print(f"[CD-INIT] Precomputed {len(_cd._state_mask_cache)} state masks "
						f"in {time.time() - t_pre:.2f}s", flush=True)
				else:
					print(f"[CD-INIT] State mask cache already warm ({len(_cd._state_mask_cache)} entries)",
						flush=True)

			if attention_mask is not None and torch.any(attention_mask == 0.0):
				attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
				tok_idx = attention_mask.long().cumsum(-1) - 1
				tok_idx.masked_fill_(attention_mask == 0, 1)
				attention_mask = torch.logical_and(
					attention_mask.unsqueeze(1).unsqueeze(-2),
					attention_mask.unsqueeze(1).unsqueeze(-1),
				)
			else:
				tok_idx = None
				attention_mask = "full"

			past_key_values = None

			for num_block in range(num_blocks):
				current_block_start = prompt_len + num_block * block_length
				current_block_end = current_block_start + block_length

				if _cd is not None:
					print(f"[CD-BLOCK] block {num_block}/{num_blocks}, "
						f"positions {current_block_start}-{current_block_end-1}", flush=True)

				t_block_start = time.time()

				# Full model forward pass for first token of block
				model_output = self(x, attention_mask, tok_idx, use_cache=True)
				past_key_values = model_output.past_key_values
				logits = model_output.logits
				logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

				if _cd is not None:
					print(f"[CD-BLOCK]   full model fwd: {time.time() - t_block_start:.3f}s", flush=True)

				# ---- First token of block: constrain if applicable ----
				if _cd is not None:
					pos = current_block_start
					if pos not in _cd.mgr.committed:
						vocab_dim = logits.shape[-1]
						t_mask = time.time()
						valid_mask = _cd.get_valid_mask(pos, logits.device, logits_vocab_size=vocab_dim)
						if valid_mask is not None and valid_mask.any():
							logits[0, pos][~valid_mask] = float('-inf')
						print(f"[CD-BLOCK]   first-token constraint: {time.time() - t_mask:.4f}s "
							f"(valid={valid_mask.sum().item() if valid_mask is not None else 'unconstrained'})",
							flush=True)

				confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
				first_token = x0[0, current_block_start].item()
				x[:, current_block_start] = first_token

				if _cd is not None:
					first_bytes = _cd.t2b.get(first_token, b'')
					if first_token != mask_token_id and first_bytes:
						_cd.commit_token(current_block_start, first_token)
						print(f"[CD-BLOCK]   first-token committed: pos={current_block_start}, "
							f"token={first_token}, bytes={first_bytes!r}, "
							f"segs={_cd.mgr.num_segments}, committed={_cd.mgr.num_committed}",
							flush=True)
					elif not first_bytes:
						print(f"[CD-BLOCK]   first-token EOS/empty: pos={current_block_start}, "
							f"token={first_token}, bytes={first_bytes!r}", flush=True)
						x[:, current_block_start] = mask_token_id

				# Truncate KV cache
				new_past_key_values = []
				for ii in range(len(past_key_values)):
					new_past_key_values.append(())
					for jj in range(len(past_key_values[ii])):
						new_past_key_values[ii] += (past_key_values[ii][jj][:, :current_block_start, :],)
				past_key_values = new_past_key_values

				# ---- Inner denoising steps ----
				i = 1
				while True:
					mask_index = (x[:, current_block_start:] == mask_token_id)
					num_masked_now = mask_index[:, :block_length].sum().item()

					if attention_mask != "full":
						current_attention_mask = attention_mask[:, :, :, current_block_start:]
					else:
						current_attention_mask = attention_mask

					t_model_start = time.time()
					model_output = self(
						x[:, current_block_start:], current_attention_mask,
						tok_idx[:, current_block_start:] if tok_idx is not None else None,
						past_key_values=past_key_values, use_cache=True,
					)
					logits = model_output.logits
					logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
					t_model_end = time.time()

					if alg == 'confidence_threshold':
						mask_logits = logits[mask_index]
						confidence, x0 = sample_tokens(mask_logits, temperature=temperature,
													top_p=top_p, top_k=top_k)

						x_ = torch.zeros_like(x[:, current_block_start:], device=self.device,
											dtype=torch.long) + mask_token_id
						full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf,
														device=self.device, dtype=logits.dtype)

						x_[mask_index] = x0.clone()
						full_confidence[mask_index] = confidence
						full_confidence[:, block_length:] = -torch.inf

						current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
						selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
						transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
						select_index = select_index.to(x.device)
						transfer_index[0, select_index[0]] = True
						for k in range(1, current_transfer_tokens):
							if selected_confidence[0, k] < threshold:
								transfer_index[0, select_index[0, k]] = False

						# Commit with constraint filtering
						if _cd is not None:
							t_cd_start = time.time()
							num_committed_step = 0
							num_constrained = 0
							num_resampled = 0
							num_unconstrained = 0

							for b_pos in range(transfer_index.shape[1]):
								if not transfer_index[0, b_pos]:
									continue
								seq_pos = current_block_start + b_pos
								proposed_token = x_[0, b_pos].item()
								if proposed_token == mask_token_id:
									continue

								committed = self._cd_commit_position(
									_cd, x, seq_pos, b_pos, proposed_token, logits,
									temperature, mask_token_id
								)
								if committed == 'constrained':
									num_constrained += 1
									num_committed_step += 1
								elif committed == 'resampled':
									num_constrained += 1
									num_resampled += 1
									num_committed_step += 1
								elif committed == 'unconstrained':
									num_unconstrained += 1
									num_committed_step += 1

							# === Greedy absorption of provisional tokens ===
							# Walk right from main segment, absorbing valid provisionals
							num_absorbed = 0
							while True:
								main_idx = _cd.mgr._find_seg_starting_at(_cd.mgr.gen_start)
								if main_idx is not None:
									absorb_pos = _cd.mgr._segments[main_idx].end + 1
								else:
									absorb_pos = _cd.mgr.gen_start

								if absorb_pos > _cd.mgr.gen_end:
									break
								if absorb_pos in _cd.mgr.committed:
									break

								current_tok = x[0, absorb_pos].item()
								if current_tok == mask_token_id:
									break  # No provisional token here

								# Validate against DFA
								if _cd.is_valid_at_position(absorb_pos, current_tok):
									tok_bytes = _cd.t2b.get(current_tok, b'')
									if tok_bytes:
										_cd.commit_token(absorb_pos, current_tok)
										num_absorbed += 1
									else:
										break  # EOS — stop absorbing
								else:
									# Invalid provisional token — resample
									b_pos_absorb = absorb_pos - current_block_start
									if b_pos_absorb < 0 or b_pos_absorb >= logits.shape[1]:
										break
									valid_mask = _cd.get_valid_mask(
										absorb_pos, logits.device,
										logits_vocab_size=logits.shape[-1]
									)
									if valid_mask is not None and valid_mask.any():
										pos_logits = logits[0, absorb_pos - current_block_start].clone()
										# Check bounds — logits might not cover this position
										if absorb_pos - current_block_start < 0 or absorb_pos - current_block_start >= logits.shape[1]:
											break  # Can't resample without logits
										pos_logits[~valid_mask] = float('-inf')
										if temperature > 0:
											probs = torch.softmax(pos_logits / temperature, dim=-1)
											chosen_token = torch.multinomial(probs, 1).item()
										else:
											chosen_token = pos_logits.argmax().item()
										chosen_bytes = _cd.t2b.get(chosen_token, b'')
										if chosen_bytes:
											x[0, absorb_pos] = chosen_token
											_cd.commit_token(absorb_pos, chosen_token)
											num_absorbed += 1
										else:
											break
									else:
										break
	 
							t_cd_end = time.time()   
	   
							parts = [f"{num_committed_step}ok"]
							if num_constrained > 0: parts.append(f"{num_constrained}cst")
							if num_resampled > 0: parts.append(f"{num_resampled}resamp")
							if num_unconstrained > 0: parts.append(f"{num_unconstrained}uncon")
							if num_absorbed > 0: parts.append(f"{num_absorbed}absorb")
							print(f"[CD-STEP] blk={num_block} step={i}/{steps_per_block}: "
								f"model={t_model_end - t_model_start:.3f}s, "
								f"cd={t_cd_end - t_cd_start:.4f}s "
								f"({'/'.join(parts)}), "
								f"masked={num_masked_now}, xfer={current_transfer_tokens.item()}, "
								f"total_committed={_cd.mgr.num_committed}, "
								f"segs={_cd.mgr.num_segments}",
		
								flush=True)
						else:
							x[:, current_block_start:][transfer_index] = x_[transfer_index]

					else:  # origin algorithm
						if i == steps_per_block:
							break
						t_val = timesteps[i]
						s = timesteps[i + 1]
						mask_index[:, block_length:] = False

						mask_logits = logits[mask_index]
						confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p,
													top_k=top_k, neg_entropy=True)
						num_mask_token = mask_index.sum() / mask_index.shape[0]
						number_transfer_tokens = (int(num_mask_token * (1 - s / t_val))
												if i < steps_per_block - 1
												else int(num_mask_token))

						full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf,
														device=self.device, dtype=logits.dtype)
						full_confidence[mask_index] = confidence
						full_confidence[:, block_length:] = -torch.inf

						if number_transfer_tokens > 0:
							if alg_temp is None or alg_temp == 0:
								_, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
							else:
								full_confidence_samp = full_confidence / alg_temp
								full_confidence_samp = F.softmax(full_confidence_samp, dim=-1)
								transfer_index = torch.multinomial(full_confidence_samp,
																num_samples=number_transfer_tokens)

							x_ = torch.zeros_like(x[:, current_block_start:], device=self.device,
												dtype=torch.long) + mask_token_id
							x_[mask_index] = x0.clone()

							if _cd is not None:
								# ============================================
								# CONSTRAINT-FILTERED COMMIT
								# ============================================
								t_cd_start = time.time()
								num_committed_step = 0
								num_constrained = 0
								num_resampled = 0
								num_unconstrained = 0

								for k in range(number_transfer_tokens):
									b_pos = transfer_index[0, k].item()
									seq_pos = current_block_start + b_pos

									# Skip if already committed
									if seq_pos in _cd.mgr.committed:
										continue
									# if x[0, seq_pos].item() != mask_token_id:
									# 	continue

									proposed_token = x_[0, b_pos].item()
									if proposed_token == mask_token_id:
										continue

									committed = self._cd_commit_position(
										_cd, x, seq_pos, b_pos, proposed_token, logits,
										temperature, mask_token_id
									)
									if committed == 'constrained':
										num_constrained += 1
										num_committed_step += 1
									elif committed == 'resampled':
										num_constrained += 1
										num_resampled += 1
										num_committed_step += 1
									elif committed == 'unconstrained':
										num_unconstrained += 1
										num_committed_step += 1
		  
								# === Greedy absorption of provisional tokens ===
								# Walk right from main segment, absorbing valid provisionals
								num_absorbed = 0
								while True:
									main_idx = _cd.mgr._find_seg_starting_at(_cd.mgr.gen_start)
									if main_idx is not None:
										absorb_pos = _cd.mgr._segments[main_idx].end + 1
									else:
										absorb_pos = _cd.mgr.gen_start

									if absorb_pos > _cd.mgr.gen_end:
										break
									if absorb_pos in _cd.mgr.committed:
										break

									current_tok = x[0, absorb_pos].item()
									if current_tok == mask_token_id:
										break  # No provisional token here

									# Validate against DFA
									if _cd.is_valid_at_position(absorb_pos, current_tok):
										tok_bytes = _cd.t2b.get(current_tok, b'')
										if tok_bytes:
											_cd.commit_token(absorb_pos, current_tok)
											num_absorbed += 1
										else:
											break  # EOS — stop absorbing
									else:
										b_pos_absorb = absorb_pos - current_block_start
										if b_pos_absorb < 0 or b_pos_absorb >= logits.shape[1]:
											break
										# Log why absorption stopped
										valid_mask = _cd.get_valid_mask(
											absorb_pos, logits.device,
											logits_vocab_size=logits.shape[-1]
										)
										if valid_mask is not None and valid_mask.any():
											pos_logits = logits[0, absorb_pos - current_block_start].clone()
											# Check bounds — logits might not cover this position
											if absorb_pos - current_block_start < 0 or absorb_pos - current_block_start >= logits.shape[1]:
												break  # Can't resample without logits
											pos_logits[~valid_mask] = float('-inf')
											if temperature > 0:
												probs = torch.softmax(pos_logits / temperature, dim=-1)
												chosen_token = torch.multinomial(probs, 1).item()
											else:
												chosen_token = pos_logits.argmax().item()
											chosen_bytes = _cd.t2b.get(chosen_token, b'')
											if chosen_bytes:
												x[0, absorb_pos] = chosen_token
												_cd.commit_token(absorb_pos, chosen_token)
												num_absorbed += 1
											else:
												break
										else:
											break

								t_cd_end = time.time()
		
								parts = [f"{num_committed_step}ok"]
								if num_constrained > 0: parts.append(f"{num_constrained}cst")
								if num_resampled > 0: parts.append(f"{num_resampled}resamp")
								if num_unconstrained > 0: parts.append(f"{num_unconstrained}uncon")
								if num_absorbed > 0: parts.append(f"{num_absorbed}absorb")

								print(f"[CD-STEP] blk={num_block} step={i}/{steps_per_block}: "
									f"model={t_model_end - t_model_start:.3f}s, "
									f"cd={t_cd_end - t_cd_start:.4f}s "
									f"({'/'.join(parts)}), "
									f"masked={num_masked_now}, xfer={number_transfer_tokens}, "
									f"total_committed={_cd.mgr.num_committed}, "
									f"segs={_cd.mgr.num_segments}",
									flush=True)
							else:
								# No constraint — batch commit
								row_indices = (torch.arange(x.size(0), device=self.device)
											.unsqueeze(1).expand_as(transfer_index))
								x[:, current_block_start:][row_indices, transfer_index] = \
									x_[row_indices, transfer_index]

						i += 1

					if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
						if _cd is not None:
							print(f"[CD-BLOCK] block {num_block} done (all unmasked), "
								f"total block time={time.time() - t_block_start:.3f}s", flush=True)
						break

			# ---- End-of-generation diagnostics ----
			if _cd is not None:
				remaining_masks = (x[0, prompt_len:] == mask_token_id).sum().item()
				total_committed = _cd.mgr.num_committed
				total_segs = _cd.mgr.num_segments
				print(f"[CD-DONE] Generation complete: "
					f"committed={total_committed}/{gen_length}, "
					f"remaining_masks={remaining_masks}, "
					f"segments={total_segs}",
					flush=True)
				if remaining_masks > 0:
					mask_positions = [p for p in range(prompt_len, prompt_len + gen_length)
									if x[0, p].item() == mask_token_id]
					print(f"[CD-DONE] Unfilled mask positions ({len(mask_positions)}): "
						f"{mask_positions[:20]}{'...' if len(mask_positions) > 20 else ''}",
						flush=True)
					eos_token_id = generation_config.eos_token_id
					if eos_token_id is not None:
						if isinstance(eos_token_id, list):
							eos_token_id = eos_token_id[0]
						x[0, prompt_len:][x[0, prompt_len:] == mask_token_id] = eos_token_id
						print(f"[CD-DONE] Replaced {remaining_masks} masks with EOS (id={eos_token_id})",
							flush=True)
				if total_segs > 0:
					empty_segs = sum(1 for s in _cd.mgr._segments if len(s.pairs) == 0)
					if empty_segs > 0:
						print(f"[CD-DONE] WARNING: {empty_segs} segments have empty pair sets "
							f"(irreconcilable tokens)", flush=True)
					if _cd.mgr.is_valid_complete():
						print(f"[CD-DONE] Valid complete JSON path exists", flush=True)
					else:
						print(f"[CD-DONE] No valid complete JSON path", flush=True)

			if return_dict_in_generate:
				return DreamModelOutput(sequences=x, history=histories)
			else:
				return x