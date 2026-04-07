"""
Constrained decoding integration for Fast-dLLM's denoising loop.
Optimized version v2 — fixes from profiling run:

1. Multi-exit-state fast path: union precomputed per-state masks with torch.logical_or
   instead of falling through to the 10s trie path.
2. Safe segment lookup: _find_seg_ending_at / _find_seg_starting_at return Segment
   objects, not list indices (fixes IndexError on stale indices).
3. commit_token() for O(1) incremental updates.
4. is_valid_at_position() for O(|bytes|) single-token checks.
5. Vectorized precompute_state_masks via numpy.
"""

from __future__ import annotations
from typing import Optional, Callable
import torch
import numpy as np

from constrained.dfa import DFA, DEAD, build_json_dfa
from constrained.segments import Segment, create, extend_right, extend_left, merge, merge_with_bridge
from constrained.manager import SegmentManager
from constrained.trie import TokenTrie


class ConstrainedDecoder:

    def __init__(
        self,
        dfa: DFA,
        trie: TokenTrie,
        token_to_bytes: dict[int, bytes],
        gen_start: int,
        gen_length: int,
        mask_token_id: int,
    ):
        self.dfa = dfa
        self.trie = trie
        self.t2b = token_to_bytes
        self.gen_start = gen_start
        self.gen_length = gen_length
        self.gen_end = gen_start + gen_length - 1
        self.mask_token_id = mask_token_id

        self.mgr = SegmentManager(
            dfa=dfa,
            gen_start=gen_start,
            gen_length=gen_length,
            token_to_bytes=lambda tid: token_to_bytes.get(tid, b''),
        )

        # position -> valid mask tensor (or None for unconstrained)
        self._mask_cache: dict[int, Optional[torch.Tensor]] = {}

        # (state, device) -> valid token mask tensor. Permanent.
        self._state_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        # Pre-filter to non-empty byte sequences
        self._nonempty_t2b = {tid: b for tid, b in token_to_bytes.items() if b}

    # ------------------------------------------------------------------
    # Segment lookup helpers (return Segment, not index — avoids stale indices)
    # ------------------------------------------------------------------

    def _seg_ending_at(self, pos: int) -> Optional[Segment]:
        """Find the segment whose .end == pos, or None."""
        for seg in self.mgr._segments:
            if seg.end == pos:
                return seg
        return None

    def _seg_starting_at(self, pos: int) -> Optional[Segment]:
        """Find the segment whose .start == pos, or None."""
        for seg in self.mgr._segments:
            if seg.start == pos:
                return seg
        return None

    # ------------------------------------------------------------------
    # Left/right state queries (replicated from manager but safe)
    # ------------------------------------------------------------------

    def _left_exit_states(self, position: int) -> frozenset[int]:
        seg = self._seg_ending_at(position - 1)
        if seg is not None:
            if seg.start <= self.mgr.gen_start:
                return frozenset(
                    x for e, x in seg.pairs if e == self.dfa.start_state
                )
            return seg.exit_states()
        elif position == self.mgr.gen_start:
            return frozenset({self.dfa.start_state})
        else:
            return frozenset(range(self.dfa.num_states))

    def _right_entry_states(self, position: int) -> frozenset[int]:
        seg = self._seg_starting_at(position + 1)
        if seg is not None:
            if seg.end >= self.mgr.gen_end:
                return frozenset(
                    e for e, x in seg.pairs if x in self.dfa.accept_states
                )
            return seg.entry_states()
        elif position == self.mgr.gen_end:
            return self.dfa.accept_states
        else:
            return frozenset(range(self.dfa.num_states))

    # ------------------------------------------------------------------
    # O(1) commit
    # ------------------------------------------------------------------

    def commit_token(self, position: int, token_id: int):
        if position in self.mgr.committed:
            return
        self.mgr.reveal_token(position, token_id)
        for p in (position - 1, position, position + 1):
            self._mask_cache.pop(p, None)

    def sync_committed(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x[0]
        changed = False
        for pos in range(self.gen_start, self.gen_start + self.gen_length):
            tid = x[pos].item()
            if tid != self.mask_token_id and pos not in self.mgr.committed:
                self.mgr.reveal_token(pos, tid)
                changed = True
        if changed:
            self._mask_cache.clear()

    # Keep old name as alias for compatibility with generation loop
    def update_committed(self, x: torch.Tensor):
        self.sync_committed(x)

    # ------------------------------------------------------------------
    # Single-token validity check (for sequential commit rejection)
    # ------------------------------------------------------------------

    def is_valid_at_position(self, position: int, token_id: int) -> bool:
        left_exits = self._left_exit_states(position)
        right_entries = self._right_entry_states(position)

        left_seg = self._seg_ending_at(position - 1)
        left_is_start = (position == self.mgr.gen_start)
        if not (left_seg is not None or left_is_start):
            return True

        right_seg = self._seg_starting_at(position + 1)
        right_is_end = (position == self.mgr.gen_end)
        right_tight = right_seg is not None or right_is_end
        effective_right = right_entries if right_tight else None

        token_bytes = self.t2b.get(token_id, b'')
        if not token_bytes:
            return False

        for q in left_exits:
            result = self.dfa.transition_seq(q, token_bytes)
            if result != DEAD:
                if effective_right is None or result in effective_right:
                    return True
        return False

    def diagnose_rejection(self, position: int, token_id: int) -> dict:
        """
        Detailed diagnosis of why a token was rejected.
        Returns a dict with all the context needed to understand the rejection.
        """
        from constrained.dfa import DEAD

        left_seg = self._seg_ending_at(position - 1)
        right_seg = self._seg_starting_at(position + 1)
        left_is_start = (position == self.mgr.gen_start)
        right_is_end = (position == self.mgr.gen_end)

        left_tight = left_seg is not None or left_is_start
        right_tight = right_seg is not None or right_is_end

        left_exits = self._left_exit_states(position)
        right_entries = self._right_entry_states(position)
        effective_right = right_entries if right_tight else None

        token_bytes = self.t2b.get(token_id, b'')

        # Walk DFA from each left exit state
        transitions = []
        for q in left_exits:
            result = self.dfa.transition_seq(q, token_bytes)
            reached_live = result != DEAD
            in_right = result in right_entries if (reached_live and effective_right is not None) else None
            transitions.append({
                'entry': q,
                'exit': result,
                'alive': reached_live,
                'in_right': in_right,
            })

        return {
            'position': position,
            'token_id': token_id,
            'token_bytes': token_bytes,
            'token_repr': repr(token_bytes),
            'left_tight': left_tight,
            'left_is_start': left_is_start,
            'left_seg': f'[{left_seg.start}-{left_seg.end}]({len(left_seg.pairs)}p)' if left_seg else None,
            'right_tight': right_tight,
            'right_is_end': right_is_end,
            'right_seg': f'[{right_seg.start}-{right_seg.end}]({len(right_seg.pairs)}p)' if right_seg else None,
            'num_left_exits': len(left_exits),
            'num_right_entries': len(effective_right) if effective_right is not None else 'unconstrained',
            'transitions': transitions,
            'reason': self._rejection_reason(transitions, effective_right, token_bytes),
        }

    def _rejection_reason(self, transitions, effective_right, token_bytes):
        if not token_bytes:
            return 'empty_bytes'
        all_dead = all(not t['alive'] for t in transitions)
        if all_dead:
            return 'all_transitions_dead'
        if effective_right is not None:
            alive_exits = [t['exit'] for t in transitions if t['alive']]
            return f'alive_but_not_in_right(exits={alive_exits})'
        return 'unknown'

    # ------------------------------------------------------------------
    # Full valid mask
    # ------------------------------------------------------------------

    def get_valid_mask(self, position: int, device: torch.device,
                       logits_vocab_size: int = 0) -> Optional[torch.Tensor]:
        if position in self._mask_cache:
            cached = self._mask_cache[position]
            if cached is None:
                return None
            if logits_vocab_size > 0 and cached.shape[0] < logits_vocab_size:
                cached = torch.nn.functional.pad(
                    cached, (0, logits_vocab_size - cached.shape[0]), value=False
                )
            return cached

        left_exits = self._left_exit_states(position)
        right_entries = self._right_entry_states(position)

        left_seg = self._seg_ending_at(position - 1)
        left_is_start = (position == self.mgr.gen_start)
        left_tight = left_seg is not None or left_is_start

        if not left_tight:
            self._mask_cache[position] = None
            return None

        right_seg = self._seg_starting_at(position + 1)
        right_is_end = (position == self.mgr.gen_end)
        right_tight = right_seg is not None or right_is_end
        effective_right = right_entries if right_tight else None

        # -----------------------------------------------------------
        # FAST PATH: use precomputed per-state masks (works for ANY
        # number of left exit states, not just 1)
        # -----------------------------------------------------------
        if effective_right is None and self._state_mask_cache:
            # Union of precomputed masks for all left exit states.
            # Each mask lookup is O(1); OR is O(vocab_size) but on GPU.
            mask = None
            for state in left_exits:
                state_mask = self._get_precomputed_state_mask(
                    state, device, logits_vocab_size
                )
                if mask is None:
                    mask = state_mask.clone()
                else:
                    mask.logical_or_(state_mask)
            if mask is not None:
                self._mask_cache[position] = mask
                return mask

        # -----------------------------------------------------------
        # RIGHT-CONSTRAINED PATH: need to intersect with right entries.
        # If few left exits, do it via precomputed masks + filtering.
        # -----------------------------------------------------------
        if effective_right is not None and len(left_exits) <= 20 and self._state_mask_cache:
            # For each left exit state, find which tokens lead to a
            # right entry state. We can do this by checking each token
            # in the union mask against the DFA, which is cheaper than
            # the full trie when the union mask is small-ish.
            # But if left_exits is large, fall through to trie.
            #
            # Actually, the most efficient approach: use precomputed masks
            # to get the candidate set, then filter by right constraint.
            union_mask = None
            for state in left_exits:
                state_mask = self._get_precomputed_state_mask(
                    state, device, logits_vocab_size
                )
                if union_mask is None:
                    union_mask = state_mask.clone()
                else:
                    union_mask.logical_or_(state_mask)

            if union_mask is not None:
                # Filter: for each candidate token, check if any left exit
                # state transitions through it to a right entry state.
                candidate_indices = union_mask.nonzero(as_tuple=True)[0]
                size = max(self.trie.vocab_size, logits_vocab_size)
                mask = torch.zeros(size, dtype=torch.bool, device=device)

                for idx in candidate_indices:
                    tid = idx.item()
                    tok_bytes = self.t2b.get(tid, b'')
                    if not tok_bytes:
                        continue
                    for q in left_exits:
                        result = self.dfa.transition_seq(q, tok_bytes)
                        if result != DEAD and result in effective_right:
                            mask[tid] = True
                            break

                self._mask_cache[position] = mask
                return mask

        # -----------------------------------------------------------
        # FALLBACK: trie path (should rarely be needed now)
        # -----------------------------------------------------------
        valid_set = self.trie.compute_valid_set(left_exits, effective_right, self.dfa)
        size = max(self.trie.vocab_size, logits_vocab_size)
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        if valid_set:
            indices = torch.tensor(list(valid_set), dtype=torch.long, device=device)
            mask[indices] = True
        self._mask_cache[position] = mask
        return mask

    # ------------------------------------------------------------------
    # Per-state mask (precomputed or on-demand)
    # ------------------------------------------------------------------

    def _get_precomputed_state_mask(self, state: int, device: torch.device,
                                     logits_vocab_size: int = 0) -> torch.Tensor:
        cache_key = (state, device)
        if cache_key in self._state_mask_cache:
            mask = self._state_mask_cache[cache_key]
            if logits_vocab_size > 0 and mask.shape[0] < logits_vocab_size:
                mask = torch.nn.functional.pad(
                    mask, (0, logits_vocab_size - mask.shape[0]), value=False
                )
            return mask

        # Check other devices
        for (s, d), m in self._state_mask_cache.items():
            if s == state:
                mask = m.to(device)
                self._state_mask_cache[cache_key] = mask
                if logits_vocab_size > 0 and mask.shape[0] < logits_vocab_size:
                    mask = torch.nn.functional.pad(
                        mask, (0, logits_vocab_size - mask.shape[0]), value=False
                    )
                return mask

        # On-demand fallback (shouldn't happen after precompute)
        size = max(self.trie.vocab_size, logits_vocab_size)
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        for tid, tok_bytes in self._nonempty_t2b.items():
            result = self.dfa.transition_seq(state, tok_bytes)
            if result != DEAD and tid < size:
                mask[tid] = True
        self._state_mask_cache[cache_key] = mask
        return mask

    # ------------------------------------------------------------------
    # Vectorized precompute
    # ------------------------------------------------------------------

    def precompute_state_masks(self, device: torch.device,
                                logits_vocab_size: int = 0,
                                states: Optional[set[int]] = None):
        import time

        t0 = time.time()
        num_states = self.dfa.num_states
        if states is None:
            states = set(range(num_states))

        # Build numpy transition table from DFA's forward (list[list[int]])
        dead_sentinel = num_states
        trans_np = np.array(self.dfa.forward, dtype=np.int32)  # [num_states, 256]
        trans_np[trans_np == DEAD] = dead_sentinel
        sentinel_row = np.full((1, 256), dead_sentinel, dtype=np.int32)
        trans_np = np.vstack([trans_np, sentinel_row])  # [num_states+1, 256]

        t_table = time.time()
        print(f"  [precompute] Transition table ({trans_np.shape}) built in "
              f"{t_table - t0:.3f}s", flush=True)

        # Walk all states through each token's bytes in parallel
        vocab_size = max(self.trie.vocab_size, logits_vocab_size)
        valid = np.zeros((num_states, vocab_size), dtype=np.bool_)
        all_states = np.arange(num_states, dtype=np.int32)
        num_tokens = len(self._nonempty_t2b)

        for count, (tid, tok_bytes) in enumerate(self._nonempty_t2b.items()):
            if tid >= vocab_size:
                continue
            current = all_states.copy()
            for b in tok_bytes:
                current = trans_np[current, b]
            valid[current != dead_sentinel, tid] = True

            if (count + 1) % 50000 == 0:
                elapsed = time.time() - t_table
                rate = (count + 1) / elapsed
                eta = (num_tokens - count - 1) / rate
                print(f"  [precompute] {count+1}/{num_tokens} tokens "
                      f"({elapsed:.1f}s, ~{eta:.0f}s left)", flush=True)

        t_scan = time.time()
        print(f"  [precompute] Vocab scan: {num_tokens} tokens in "
              f"{t_scan - t_table:.1f}s", flush=True)

        for state in sorted(states):
            mask = torch.from_numpy(valid[state].copy()).to(device)
            self._state_mask_cache[(state, device)] = mask

        print(f"  [precompute] Done: {len(states)} masks cached, "
              f"{time.time() - t0:.1f}s total", flush=True)


def build_constrained_decoder(
    tokenizer,
    max_depth: int = 6,
) -> tuple:
    byte_decoder = tokenizer.byte_decoder
    t2b = {}
    for token_id in range(tokenizer.vocab_size):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        if token_str is None:
            t2b[token_id] = b""
            continue
        try:
            t2b[token_id] = bytes(byte_decoder[c] for c in token_str)
        except KeyError:
            t2b[token_id] = b""
    dfa = build_json_dfa(max_depth=max_depth)
    trie = TokenTrie(t2b)
    return dfa, trie, t2b