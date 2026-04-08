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

        # Precompute mask of tokens with empty bytes (special/meta tokens).
        # These can never be valid JSON content — reject them at the logit level.
        self._empty_byte_token_ids = frozenset(
            tid for tid, b in token_to_bytes.items() if not b
        )

    def get_empty_byte_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        """
        Returns a boolean mask where True = token should be blocked.
        Blocks both:
        - Tokens with empty bytes in t2b (special tokens within tokenizer vocab)
        - Tokens with IDs >= len(t2b) (padding tokens beyond tokenizer vocab)
        Apply as: logits[:, mask] = -inf
        """
        if not hasattr(self, '_empty_byte_mask_cache'):
            self._empty_byte_mask_cache = {}
        cache_key = (vocab_size, device)
        if cache_key not in self._empty_byte_mask_cache:
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            # Block tokens explicitly mapped to empty bytes
            for tid in self._empty_byte_token_ids:
                if tid < vocab_size:
                    mask[tid] = True
            # Block tokens beyond the tokenizer's vocabulary
            # (padding in the embedding layer, no byte representation)
            max_known_tid = max(self.t2b.keys()) if self.t2b else 0
            if vocab_size > max_known_tid + 1:
                mask[max_known_tid + 1:] = True
            # Also block the mask token itself
            if self.mask_token_id < vocab_size:
                mask[self.mask_token_id] = True
            self._empty_byte_mask_cache[cache_key] = mask
        return self._empty_byte_mask_cache[cache_key]

    # ------------------------------------------------------------------
    # Force-close: compute minimal byte sequence to reach accept state
    # ------------------------------------------------------------------

    def compute_closing_bytes(self) -> Optional[bytes]:
        """
        Find the shortest byte sequence that drives the DFA from the
        main segment's exit state to an accept state.

        Returns None if already in accept state or no path exists.
        Uses BFS over DFA states.
        """
        # Find the main segment (the one starting at or before gen_start)
        main_seg = None
        for seg in self.mgr._segments:
            if seg.start <= self.mgr.gen_start:
                main_seg = seg
                break
        if main_seg is None and self.mgr._segments:
            # Use the first segment
            main_seg = self.mgr._segments[0]
        if main_seg is None:
            return None

        # Get exit states filtered by start_state entry
        if main_seg.start <= self.mgr.gen_start:
            exit_states = frozenset(
                x for e, x in main_seg.pairs if e == self.dfa.start_state
            )
        else:
            exit_states = main_seg.exit_states()

        if not exit_states:
            return None

        # Check if any exit state is already an accept state
        for s in exit_states:
            if s in self.dfa.accept_states:
                return b''  # already valid

        # BFS: find shortest byte path from any exit state to any accept state
        from collections import deque
        # State: (dfa_state, bytes_so_far)
        queue = deque()
        visited = set()
        for s in exit_states:
            queue.append((s, b''))
            visited.add(s)

        while queue:
            state, path = queue.popleft()
            if len(path) > 20:
                # Safety limit — don't search forever
                break

            for byte_val in range(256):
                next_state = self.dfa.forward[state][byte_val]
                if next_state == DEAD or next_state in visited:
                    continue
                new_path = path + bytes([byte_val])
                if next_state in self.dfa.accept_states:
                    return new_path
                visited.add(next_state)
                queue.append((next_state, new_path))

        return None  # no path found

    def find_closing_tokens(self) -> list[int]:
        """
        Find token IDs that could close the JSON from current state.

        Returns a list of token IDs to append, or empty list if
        no closing sequence found.
        """
        closing_bytes = self.compute_closing_bytes()
        if closing_bytes is None or closing_bytes == b'':
            return []

        # Find tokens whose bytes match prefixes of closing_bytes
        # Greedy: find the longest token that matches a prefix
        result = []
        remaining = closing_bytes
        while remaining:
            best_tid = None
            best_len = 0
            for tid, tok_bytes in self._nonempty_t2b.items():
                if remaining.startswith(tok_bytes) and len(tok_bytes) > best_len:
                    best_tid = tid
                    best_len = len(tok_bytes)
            if best_tid is None:
                # No token matches — try single-byte tokens
                byte_val = remaining[0]
                for tid, tok_bytes in self._nonempty_t2b.items():
                    if tok_bytes == bytes([byte_val]):
                        best_tid = tid
                        best_len = 1
                        break
            if best_tid is None:
                break  # can't find a matching token
            result.append(best_tid)
            remaining = remaining[best_len:]
        return result

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
            return True  # unconstrained

        right_seg = self._seg_starting_at(position + 1)
        right_is_end = (position == self.mgr.gen_end)
        right_tight = right_seg is not None or right_is_end
        effective_right = right_entries if right_tight else None

        token_bytes = self.t2b.get(token_id, b'')

        # EOS/empty-byte token: valid iff the current DFA state is accepting.
        # This means the JSON is syntactically complete at this point.
        # All positions after this will also see an accept state and allow EOS.
        if not token_bytes:
            # EOS is valid when any left exit state is an accept state
            # AND (if right is constrained) the accept state is compatible
            for q in left_exits:
                if q in self.dfa.accept_states:
                    if effective_right is None or q in effective_right:
                        return True
            return False  # JSON not complete — EOS rejected

        # Normal byte-bearing token
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

        # Check if EOS should be allowed (any left exit is an accept state)
        eos_allowed = any(q in self.dfa.accept_states for q in left_exits)
        if eos_allowed and effective_right is not None:
            # Also need accept state to be in right entries
            eos_allowed = any(
                q in self.dfa.accept_states and q in effective_right
                for q in left_exits
            )

        def _mark_eos_valid(mask):
            """If EOS is allowed, add all EOS-like token IDs to valid set."""
            if not eos_allowed:
                return mask
            for tid in self._empty_byte_token_ids:
                if tid < mask.shape[0]:
                    mask[tid] = True
            max_known = max(self.t2b.keys()) if self.t2b else 0
            if mask.shape[0] > max_known + 1:
                mask[max_known + 1:] = True
            if self.mask_token_id < mask.shape[0]:
                mask[self.mask_token_id] = True
            return mask

        # -----------------------------------------------------------
        # FAST PATH: use precomputed per-state masks
        # -----------------------------------------------------------
        if effective_right is None and self._state_mask_cache:
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
                mask = _mark_eos_valid(mask)
                self._mask_cache[position] = mask
                return mask

        # -----------------------------------------------------------
        # RIGHT-CONSTRAINED PATH: need to intersect with right entries.
        # Only feasible with small exit state sets.
        # -----------------------------------------------------------
        if effective_right is not None and len(left_exits) <= 10 and self._state_mask_cache:
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

                mask = _mark_eos_valid(mask)
                self._mask_cache[position] = mask
                return mask

        # -----------------------------------------------------------
        # LARGE EXIT SET WITH RIGHT CONSTRAINT: use union mask as
        # over-approximation (skip right filtering for speed)
        # -----------------------------------------------------------
        if effective_right is not None and len(left_exits) > 10 and self._state_mask_cache:
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
                mask = _mark_eos_valid(mask)
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
        mask = _mark_eos_valid(mask)
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