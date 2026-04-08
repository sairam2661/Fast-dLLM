"""
Fast-dLLM generation with grammar-constrained decoding.
Optimized version — key changes:

1. Uses commit_token() instead of update_committed() — avoids O(gen_length) scan per token
2. Uses is_valid_at_position() for rejection — avoids O(vocab) mask computation
3. Precomputes state masks before generation starts
4. Batches unconstrained positions (skips constraint check entirely)
5. Reduced debug output (controlled by _cd_verbose flag)
"""

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
        _cd_verbose = True  # Always log when constrained decoder is present

        if _cd is not None:
            print(f"[CD-INIT] gen_start={prompt_len}, gen_length={gen_length}, "
                  f"num_blocks={num_blocks}, steps_per_block={steps_per_block}, "
                  f"block_length={block_length}, alg={alg}", flush=True)

        # ---- CONSTRAINED DECODING: initialize ----
        if _cd is not None:
            _cd.gen_start = prompt_len
            _cd.gen_length = gen_length
            _cd.gen_end = prompt_len + gen_length - 1
            _cd.mgr.gen_start = prompt_len
            _cd.mgr.gen_length = gen_length
            _cd.mgr.gen_end = prompt_len + gen_length - 1
            _cd.mgr.reset()
            _cd._mask_cache.clear()

            # Precompute state masks for all DFA states on first run
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
            model_output = self(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            if _cd is not None:
                print(f"[CD-BLOCK]   full model fwd: {time.time() - t_block_start:.3f}s", flush=True)

            # ---- CONSTRAINED DECODING: first token of block ----
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

            # Only commit if the first token has actual bytes (not EOS/padding)
            if _cd is not None:
                first_bytes = _cd.t2b.get(first_token, b'')
                if first_token != mask_token_id and first_bytes:
                    _cd.commit_token(current_block_start, first_token)
                elif not first_bytes:
                    # EOS at block start — model wants to end here.
                    # Revert to mask so it doesn't pollute the sequence.
                    x[:, current_block_start] = mask_token_id

            if not dual_cache:
                new_past_key_values = []
                for ii in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for jj in range(len(past_key_values[ii])):
                        new_past_key_values[ii] += (past_key_values[ii][jj][:, :current_block_start, :],)
                past_key_values = new_past_key_values
            else:
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1

            i = 1
            while True:
                t_step_start = time.time()

                if dual_cache:
                    mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                else:
                    mask_index = (x[:, current_block_start:] == mask_token_id)

                num_masked_now = mask_index[:, :block_length].sum().item()

                if attention_mask != "full":
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask

                t_model_start = time.time()
                if dual_cache:
                    model_output = self(
                        x[:, current_block_start:current_block_end], current_attention_mask,
                        tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None,
                        past_key_values=past_key_values, use_cache=True,
                        dual_cache=dual_cache, replace_position=replace_position,
                    )
                else:
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
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)

                    if dual_cache:
                        x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)

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
                    if dual_cache:
                        x[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
                    else:
                        x[:, current_block_start:][transfer_index] = x_[transfer_index]

                else:  # origin algorithm
                    if i == steps_per_block:
                        break
                    t_val = timesteps[i]
                    s = timesteps[i + 1]
                    mask_index[:, block_length:] = False

                    mask_logits = logits[mask_index]

                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t_val)) if i < steps_per_block - 1 else int(num_mask_token)

                    if dual_cache:
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    full_confidence[:, block_length:] = -torch.inf

                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence_samp = full_confidence / alg_temp
                            full_confidence_samp = F.softmax(full_confidence_samp, dim=-1)
                            transfer_index = torch.multinomial(full_confidence_samp, num_samples=number_transfer_tokens)

                        if dual_cache:
                            x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        else:
                            x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()

                        if _cd is not None:
                            # ============================================
                            # FRONTIER-PRIORITY CONSTRAINED COMMIT
                            # ============================================
                            t_cd_start = time.time()
                            num_committed_step = 0
                            num_constrained = 0
                            num_resampled = 0
                            num_unconstrained = 0
                            num_frontier = 0

                            # === PHASE 1: Frontier-priority commitment ===
                            # Grow the main segment rightward, AR-style.
                            # The frontier has a single DFA exit state, giving
                            # tight constraints identical to AR decoding.
                            frontier_budget = number_transfer_tokens
                            while frontier_budget > 0:
                                # Find main segment (the one starting at gen_start)
                                main_idx = _cd.mgr._find_seg_starting_at(_cd.mgr.gen_start)
                                if main_idx is not None:
                                    main_seg = _cd.mgr._segments[main_idx]
                                    frontier_pos = main_seg.end + 1
                                else:
                                    frontier_pos = _cd.mgr.gen_start

                                # Stop if frontier is past generation region
                                if frontier_pos > _cd.mgr.gen_end:
                                    break
                                # Stop if frontier already committed
                                if frontier_pos in _cd.mgr.committed:
                                    # Frontier might have been committed by a previous
                                    # block/step. Advance past it by re-looping — the
                                    # main segment will have absorbed it.
                                    break
                                # Stop if frontier position isn't masked
                                if x[0, frontier_pos].item() != mask_token_id:
                                    break

                                # Compute position within logits tensor
                                if dual_cache:
                                    b_pos = frontier_pos - current_block_start
                                else:
                                    b_pos = frontier_pos - current_block_start
                                # Stop if frontier is outside current logits window
                                if b_pos < 0 or b_pos >= logits.shape[1]:
                                    break

                                # Get constrained valid mask at frontier
                                valid_mask = _cd.get_valid_mask(
                                    frontier_pos, logits.device,
                                    logits_vocab_size=logits.shape[-1]
                                )

                                pos_logits = logits[0, b_pos].clone()
                                if valid_mask is not None and valid_mask.any():
                                    pos_logits[~valid_mask] = float('-inf')

                                # Sample from constrained logits
                                if temperature > 0:
                                    probs = torch.softmax(pos_logits / temperature, dim=-1)
                                    chosen_token = torch.multinomial(probs, 1).item()
                                else:
                                    chosen_token = pos_logits.argmax().item()

                                chosen_bytes = _cd.t2b.get(chosen_token, b'')
                                if chosen_bytes:
                                    # Real token — commit and extend frontier
                                    x[0, frontier_pos] = chosen_token
                                    _cd.commit_token(frontier_pos, chosen_token)
                                    num_frontier += 1
                                    num_committed_step += 1
                                    frontier_budget -= 1
                                else:
                                    # EOS/empty-bytes token. If it survived the valid
                                    # mask, the DFA is in an accept state. Write it
                                    # and stop frontier growth.
                                    x[0, frontier_pos] = chosen_token
                                    num_frontier += 1
                                    num_committed_step += 1
                                    break

                            # === PHASE 2: Confidence-ordered for remaining budget ===
                            remaining_budget = number_transfer_tokens - num_committed_step
                            if remaining_budget > 0:
                                for k in range(number_transfer_tokens):
                                    if num_committed_step >= number_transfer_tokens:
                                        break
                                    b_pos = transfer_index[0, k].item()
                                    seq_pos = current_block_start + b_pos

                                    # Skip if already committed in frontier phase
                                    if seq_pos in _cd.mgr.committed:
                                        continue
                                    # Skip if not masked
                                    if x[0, seq_pos].item() != mask_token_id:
                                        continue

                                    proposed_token = x_[0, b_pos].item()
                                    if proposed_token == mask_token_id:
                                        continue

                                    left_seg = _cd._seg_ending_at(seq_pos - 1)
                                    left_is_start = (seq_pos == _cd.mgr.gen_start)
                                    is_constrained = left_seg is not None or left_is_start

                                    if is_constrained:
                                        num_constrained += 1
                                        if _cd.is_valid_at_position(seq_pos, proposed_token):
                                            x[0, seq_pos] = proposed_token
                                            tok_bytes = _cd.t2b.get(proposed_token, b'')
                                            if tok_bytes:
                                                _cd.commit_token(seq_pos, proposed_token)
                                            num_committed_step += 1
                                        else:
                                            valid_mask = _cd.get_valid_mask(
                                                seq_pos, logits.device,
                                                logits_vocab_size=logits.shape[-1]
                                            )
                                            if valid_mask is not None and valid_mask.any():
                                                pos_logits = logits[0, b_pos].clone()
                                                pos_logits[~valid_mask] = float('-inf')
                                                if temperature > 0:
                                                    probs = torch.softmax(
                                                        pos_logits / temperature, dim=-1
                                                    )
                                                    chosen_token = torch.multinomial(probs, 1).item()
                                                else:
                                                    chosen_token = pos_logits.argmax().item()
                                                x[0, seq_pos] = chosen_token
                                                chosen_bytes = _cd.t2b.get(chosen_token, b'')
                                                if chosen_bytes:
                                                    _cd.commit_token(seq_pos, chosen_token)
                                                num_committed_step += 1
                                                num_resampled += 1
                                    else:
                                        tok_bytes = _cd.t2b.get(proposed_token, b'')
                                        if not tok_bytes:
                                            continue
                                        num_unconstrained += 1
                                        x[0, seq_pos] = proposed_token
                                        _cd.commit_token(seq_pos, proposed_token)
                                        num_committed_step += 1

                            t_cd_end = time.time()

                            parts = [f"{num_committed_step}ok"]
                            if num_frontier > 0: parts.append(f"{num_frontier}front")
                            if num_constrained > 0: parts.append(f"{num_constrained}cst")
                            if num_resampled > 0: parts.append(f"{num_resampled}resamp")
                            if num_unconstrained > 0: parts.append(f"{num_unconstrained}uncon")
                            rej_str = "(" + "/".join(parts) + ")"
                            print(f"[CD-STEP] blk={num_block} step={i}/{steps_per_block}: "
                                f"model={t_model_end - t_model_start:.3f}s, "
                                f"cd={t_cd_end - t_cd_start:.4f}s "
                                f"{rej_str}, "
                                f"masked={num_masked_now}, xfer={number_transfer_tokens}, "
                                f"total_committed={_cd.mgr.num_committed}, "
                                f"segs={_cd.mgr.num_segments}",
                                flush=True)
                        else:
                            # No constraint — batch commit
                            row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                            if dual_cache:
                                x[:, current_block_start:current_block_end][row_indices, transfer_index] = x_[row_indices, transfer_index]
                            else:
                                x[:, current_block_start:][row_indices, transfer_index] = x_[row_indices, transfer_index]

                    i += 1

                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    if _cd is not None:
                        print(f"[CD-BLOCK] block {num_block} done (all unmasked), "
                              f"total block time={time.time() - t_block_start:.3f}s", flush=True)
                    break

        if _cd is not None:
            # End-of-generation diagnostics
            remaining_masks = (x[0, prompt_len:] == mask_token_id).sum().item()
            total_committed = _cd.mgr.num_committed
            total_segs = _cd.mgr.num_segments
            print(f"[CD-DONE] Generation complete: "
                  f"committed={total_committed}/{gen_length}, "
                  f"remaining_masks={remaining_masks}, "
                  f"segments={total_segs}",
                  flush=True)
            if remaining_masks > 0:
                # Find where the masks are
                mask_positions = []
                for p in range(prompt_len, prompt_len + gen_length):
                    if x[0, p].item() == mask_token_id:
                        mask_positions.append(p)
                print(f"[CD-DONE] Unfilled mask positions ({len(mask_positions)}): "
                      f"{mask_positions[:20]}{'...' if len(mask_positions) > 20 else ''}",
                      flush=True)

                # Replace remaining masks with EOS
                eos_token_id = generation_config.eos_token_id
                if eos_token_id is not None:
                    if isinstance(eos_token_id, list):
                        eos_token_id = eos_token_id[0]
                    x[0, prompt_len:][x[0, prompt_len:] == mask_token_id] = eos_token_id
                    print(f"[CD-DONE] Replaced {remaining_masks} masks with EOS (id={eos_token_id})",
                          flush=True)
            # Check segment validity
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