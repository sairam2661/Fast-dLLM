"""
Constrained decoding integration for the diffusion denoising loop.

Supports two backends:

  Legacy DFA backend:
    Build with build_constrained_decoder_dfa() (unchanged from v2).
    automaton is a DFA; scanner is None.

  Scanner + LR backend (new):
    Build with build_constrained_decoder_lr().
    automaton is a CompositeAutomaton wrapping BoundedLRAutomaton + JsonScanner.
    Segment pairs carry (scanner_state, parser_config) composite IDs.
    Precomputed masks are indexed by composite ID.

The get_valid_mask / is_valid_at_position / commit_token interfaces are
identical in both modes. The caller does not need to know which backend
is active.

Changes from v2:
  - ConstrainedDecoder.__init__ accepts `automaton` + `scanner` instead of `dfa`.
    `dfa` kwarg is still accepted for backward compatibility.
  - precompute_state_masks iterates over composite IDs (num_states composites).
  - _left_exit_states / _right_entry_states return composite IDs.
  - compute_closing_bytes does BFS over composite states.
  - trie calls pass scanner= argument.
"""

from __future__ import annotations
from typing import Optional, Callable
import torch
import numpy as np

from constrained.trie import TokenTrie
from constrained.manager import SegmentManager, CompositeAutomaton
from constrained.segments import Segment, create, extend_right, extend_left, merge, merge_with_bridge


class ConstrainedDecoder:

    def __init__(
        self,
        automaton,
        trie: TokenTrie,
        token_to_bytes: dict[int, bytes],
        gen_start: int,
        gen_length: int,
        mask_token_id: int,
        scanner=None,
        # Legacy compat: accept dfa= kwarg
        dfa=None,
    ):
        # Support legacy dfa= kwarg
        if dfa is not None and automaton is None:
            automaton = dfa
        self.automaton = automaton
        self.scanner = scanner
        self.trie = trie
        self.t2b = token_to_bytes
        self.gen_start = gen_start
        self.gen_length = gen_length
        self.gen_end = gen_start + gen_length - 1
        self.mask_token_id = mask_token_id

        self.mgr = SegmentManager(
            automaton=automaton,
            gen_start=gen_start,
            gen_length=gen_length,
            token_to_bytes=lambda tid: token_to_bytes.get(tid, b''),
        )

        self._mask_cache: dict[int, Optional[torch.Tensor]] = {}
        # composite_id -> torch.Tensor (bool mask over vocab)
        self._state_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        self._nonempty_t2b = {tid: b for tid, b in token_to_bytes.items() if b}
        self._empty_byte_token_ids = frozenset(
            tid for tid, b in token_to_bytes.items() if not b
        )

    # Keep .dfa as alias for legacy code that accesses it directly
    @property
    def dfa(self):
        return self.automaton

    # ------------------------------------------------------------------
    # Empty-byte mask
    # ------------------------------------------------------------------

    def get_empty_byte_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        if not hasattr(self, '_empty_byte_mask_cache'):
            self._empty_byte_mask_cache = {}
        key = (vocab_size, device)
        if key not in self._empty_byte_mask_cache:
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            for tid in self._empty_byte_token_ids:
                if tid < vocab_size:
                    mask[tid] = True
            max_known = max(self.t2b.keys()) if self.t2b else 0
            if vocab_size > max_known + 1:
                mask[max_known + 1:] = True
            if self.mask_token_id < vocab_size:
                mask[self.mask_token_id] = True
            self._empty_byte_mask_cache[key] = mask
        return self._empty_byte_mask_cache[key]

    # ------------------------------------------------------------------
    # Segment lookup helpers
    # ------------------------------------------------------------------

    def _seg_ending_at(self, pos: int) -> Optional[Segment]:
        for seg in self.mgr._segments:
            if seg.end == pos:
                return seg
        return None

    def _seg_starting_at(self, pos: int) -> Optional[Segment]:
        for seg in self.mgr._segments:
            if seg.start == pos:
                return seg
        return None

    # ------------------------------------------------------------------
    # Left/right composite state queries
    # ------------------------------------------------------------------

    def _left_exit_states(self, position: int) -> frozenset[int]:
        seg = self._seg_ending_at(position - 1)
        if seg is not None:
            if seg.start <= self.mgr.gen_start:
                start = self.automaton.start_state
                return frozenset(x for e, x in seg.pairs if e == start)
            return seg.exit_states()
        elif position == self.mgr.gen_start:
            return frozenset({self.automaton.start_state})
        else:
            return frozenset(self.automaton.all_configs())

    def _right_entry_states(self, position: int) -> frozenset[int]:
        seg = self._seg_starting_at(position + 1)
        if seg is not None:
            if seg.end >= self.mgr.gen_end:
                accepts = self.automaton.accept_states
                return frozenset(e for e, x in seg.pairs if x in accepts)
            return seg.entry_states()
        elif position == self.mgr.gen_end:
            return self.automaton.accept_states
        else:
            return frozenset(self.automaton.all_configs())

    # ------------------------------------------------------------------
    # Closing bytes / tokens (composite BFS)
    # ------------------------------------------------------------------

    def compute_closing_bytes(self) -> Optional[bytes]:
        """
        BFS over composite states to find shortest byte sequence that
        drives from current exit composites to an accept composite.
        """
        # Find exit composites from the main (leftmost) segment
        main_seg = None
        for seg in self.mgr._segments:
            if seg.start <= self.mgr.gen_start:
                main_seg = seg
                break
        if main_seg is None and self.mgr._segments:
            main_seg = self.mgr._segments[0]
        if main_seg is None:
            return None

        start = self.automaton.start_state
        if main_seg.start <= self.mgr.gen_start:
            exit_composites = frozenset(x for e, x in main_seg.pairs if e == start)
        else:
            exit_composites = main_seg.exit_states()

        if not exit_composites:
            return None

        # Check if any exit composite is already accepting
        accepts = self.automaton.accept_states
        for c in exit_composites:
            if c in accepts:
                return b''

        from collections import deque
        queue = deque()
        visited: set[int] = set()
        for c in exit_composites:
            queue.append((c, b''))
            visited.add(c)

        while queue:
            composite, path = queue.popleft()
            if len(path) > 20:
                break
            for byte_val in range(256):
                next_composites = self.automaton.transition_seq(composite, bytes([byte_val]))
                for nc in next_composites:
                    if nc in visited:
                        continue
                    new_path = path + bytes([byte_val])
                    if nc in accepts:
                        return new_path
                    visited.add(nc)
                    queue.append((nc, new_path))

        return None

    def find_closing_tokens(self) -> list[int]:
        closing_bytes = self.compute_closing_bytes()
        if closing_bytes is None or closing_bytes == b'':
            return []
        result = []
        remaining = closing_bytes
        while remaining:
            best_tid = best_len = None
            for tid, tok_bytes in self._nonempty_t2b.items():
                if remaining.startswith(tok_bytes) and (best_len is None or len(tok_bytes) > best_len):
                    best_tid, best_len = tid, len(tok_bytes)
            if best_tid is None:
                byte_val = remaining[0]
                for tid, tok_bytes in self._nonempty_t2b.items():
                    if tok_bytes == bytes([byte_val]):
                        best_tid, best_len = tid, 1
                        break
            if best_tid is None:
                break
            result.append(best_tid)
            remaining = remaining[best_len:]
        return result

    # ------------------------------------------------------------------
    # Commit
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

    def update_committed(self, x: torch.Tensor):
        self.sync_committed(x)

    # ------------------------------------------------------------------
    # Single-token validity
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

        if not token_bytes:
            # EOS: valid iff a left-exit composite is an accept composite
            for q in left_exits:
                if q in self.automaton.accept_states:
                    if effective_right is None or q in effective_right:
                        return True
            return False

        for q in left_exits:
            result = self.automaton.transition_seq(q, token_bytes)
            # result is frozenset (CompositeAutomaton) or int (legacy DFA)
            hits = result if isinstance(result, frozenset) else (
                frozenset() if result == -1 else frozenset({result})
            )
            if effective_right is None:
                if hits:
                    return True
            else:
                if hits & effective_right:
                    return True
        return False

    def diagnose_rejection(self, position: int, token_id: int) -> dict:
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

        transitions = []
        for q in left_exits:
            result = self.automaton.transition_seq(q, token_bytes)
            hits = result if isinstance(result, frozenset) else (
                frozenset() if result == -1 else frozenset({result})
            )
            alive = bool(hits)
            in_right = bool(hits & effective_right) if (alive and effective_right is not None) else None
            transitions.append({'entry': q, 'exit': list(hits), 'alive': alive, 'in_right': in_right})

        all_dead = all(not t['alive'] for t in transitions)
        reason = ('empty_bytes' if not token_bytes
                  else 'all_transitions_dead' if all_dead
                  else f'alive_but_not_in_right' if effective_right is not None
                  else 'unknown')

        return {
            'position': position,
            'token_id': token_id,
            'token_bytes': token_bytes,
            'left_tight': left_tight,
            'left_seg': f'[{left_seg.start}-{left_seg.end}]({len(left_seg.pairs)}p)' if left_seg else None,
            'right_tight': right_tight,
            'right_seg': f'[{right_seg.start}-{right_seg.end}]({len(right_seg.pairs)}p)' if right_seg else None,
            'num_left_exits': len(left_exits),
            'num_right_entries': len(effective_right) if effective_right is not None else 'unconstrained',
            'transitions': transitions,
            'reason': reason,
        }

    # ------------------------------------------------------------------
    # Full valid mask
    # ------------------------------------------------------------------

    def get_valid_mask(
        self,
        position: int,
        device: torch.device,
        logits_vocab_size: int = 0,
    ) -> Optional[torch.Tensor]:

        if position in self._mask_cache:
            cached = self._mask_cache[position]
            if cached is None:
                return None
            if logits_vocab_size > 0 and cached.shape[0] < logits_vocab_size:
                cached = torch.nn.functional.pad(
                    cached, (0, logits_vocab_size - cached.shape[0]), value=False)
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

        accepts = self.automaton.accept_states
        eos_allowed = bool(left_exits & accepts)
        if eos_allowed and effective_right is not None:
            eos_allowed = bool(left_exits & accepts & effective_right)

        def _mark_eos_valid(mask):
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

        # FAST PATH: union precomputed per-state masks (no right constraint)
        if effective_right is None and self._state_mask_cache:
            mask = None
            for state in left_exits:
                state_mask = self._get_precomputed_state_mask(state, device, logits_vocab_size)
                mask = state_mask.clone() if mask is None else mask.logical_or_(state_mask)
            if mask is not None:
                mask = _mark_eos_valid(mask)
                self._mask_cache[position] = mask
                return mask

        # RIGHT-CONSTRAINED FAST PATH: union then re-filter
        if effective_right is not None and len(left_exits) <= 20 and self._state_mask_cache:
            union_mask = None
            for state in left_exits:
                sm = self._get_precomputed_state_mask(state, device, logits_vocab_size)
                union_mask = sm.clone() if union_mask is None else union_mask.logical_or_(sm)

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
                        result = self.automaton.transition_seq(q, tok_bytes)
                        hits = result if isinstance(result, frozenset) else (
                            frozenset() if result == -1 else frozenset({result})
                        )
                        if hits & effective_right:
                            mask[tid] = True
                            break

                mask = _mark_eos_valid(mask)
                self._mask_cache[position] = mask
                return mask

        # LARGE EXIT SET WITH RIGHT CONSTRAINT: union mask as over-approx
        if effective_right is not None and len(left_exits) > 20 and self._state_mask_cache:
            mask = None
            for state in left_exits:
                sm = self._get_precomputed_state_mask(state, device, logits_vocab_size)
                mask = sm.clone() if mask is None else mask.logical_or_(sm)
            if mask is not None:
                mask = _mark_eos_valid(mask)
                self._mask_cache[position] = mask
                return mask

        # FALLBACK: trie traversal
        valid_set = self.trie.compute_valid_set(
            left_exits, effective_right, self.automaton
        )
        size = max(self.trie.vocab_size, logits_vocab_size)
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        if valid_set:
            indices = torch.tensor(list(valid_set), dtype=torch.long, device=device)
            mask[indices] = True
        mask = _mark_eos_valid(mask)
        self._mask_cache[position] = mask
        return mask

    # ------------------------------------------------------------------
    # Per-composite-state mask
    # ------------------------------------------------------------------

    def _get_precomputed_state_mask(
        self, composite: int, device: torch.device, logits_vocab_size: int = 0
    ) -> torch.Tensor:
        key = (composite, device)
        if key in self._state_mask_cache:
            mask = self._state_mask_cache[key]
            if logits_vocab_size > 0 and mask.shape[0] < logits_vocab_size:
                mask = torch.nn.functional.pad(
                    mask, (0, logits_vocab_size - mask.shape[0]), value=False)
            return mask

        # Try other devices
        for (c, d), m in self._state_mask_cache.items():
            if c == composite:
                mask = m.to(device)
                self._state_mask_cache[key] = mask
                if logits_vocab_size > 0 and mask.shape[0] < logits_vocab_size:
                    mask = torch.nn.functional.pad(
                        mask, (0, logits_vocab_size - mask.shape[0]), value=False)
                return mask

        # On-demand: compute via trie traversal for this single composite
        size = max(self.trie.vocab_size, logits_vocab_size)
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        valid = self.trie.compute_valid_set(
            frozenset({composite}), None, self.automaton
        )
        for tid in valid:
            if tid < size:
                mask[tid] = True
        self._state_mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------
    # Vectorized precompute
    # ------------------------------------------------------------------

    def precompute_state_masks(
        self,
        device: torch.device,
        logits_vocab_size: int = 0,
        composites: Optional[set[int]] = None,
    ):
        """
        Precompute per-composite-state valid token masks.

        For the composite (scanner+LR) backend, iterates over all reachable
        composite states and computes which tokens are valid from each.

        For the legacy DFA backend, iterates over DFA states using the
        vectorized numpy approach.
        """
        import time
        t0 = time.time()

        # Legacy DFA: use fast numpy path
        if self.scanner is None:
            self._precompute_dfa_masks(device, logits_vocab_size, composites)
            return

        # Composite path: use trie traversal per composite state
        all_composites = composites if composites is not None else set(self.automaton.all_configs())
        vocab_size = max(self.trie.vocab_size, logits_vocab_size)

        print(f"  [precompute] {len(all_composites)} composite states, vocab={vocab_size}", flush=True)

        for count, composite in enumerate(sorted(all_composites)):
            valid = self.trie.compute_valid_set(
                frozenset({composite}), None, self.automaton
            )
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            for tid in valid:
                if tid < vocab_size:
                    mask[tid] = True
            self._state_mask_cache[(composite, device)] = mask

            if (count + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (count + 1) / elapsed
                remaining = (len(all_composites) - count - 1) / rate
                print(f"  [precompute] {count+1}/{len(all_composites)} "
                      f"({elapsed:.1f}s, ~{remaining:.0f}s left)", flush=True)

        print(f"  [precompute] Done: {len(all_composites)} masks in "
              f"{time.time()-t0:.1f}s", flush=True)

    def _precompute_dfa_masks(
        self, device: torch.device, logits_vocab_size: int, states
    ):
        """Legacy DFA vectorized precompute (unchanged from v2)."""
        import time
        from constrained.dfa import DEAD
        t0 = time.time()
        dfa = self.automaton
        num_states = dfa.num_states
        if states is None:
            states = set(range(num_states))

        dead_sentinel = num_states
        trans_np = np.array(dfa.forward, dtype=np.int32)
        trans_np[trans_np == DEAD] = dead_sentinel
        sentinel_row = np.full((1, 256), dead_sentinel, dtype=np.int32)
        trans_np = np.vstack([trans_np, sentinel_row])

        vocab_size = max(self.trie.vocab_size, logits_vocab_size)
        valid = np.zeros((num_states, vocab_size), dtype=np.bool_)
        all_states_arr = np.arange(num_states, dtype=np.int32)
        num_tokens = len(self._nonempty_t2b)

        for count, (tid, tok_bytes) in enumerate(self._nonempty_t2b.items()):
            if tid >= vocab_size:
                continue
            current = all_states_arr.copy()
            for b in tok_bytes:
                current = trans_np[current, b]
            valid[current != dead_sentinel, tid] = True
            if (count + 1) % 50000 == 0:
                elapsed = time.time() - t0
                print(f"  [precompute] {count+1}/{num_tokens} tokens "
                      f"({elapsed:.1f}s)", flush=True)

        for state in sorted(states):
            mask = torch.from_numpy(valid[state].copy()).to(device)
            self._state_mask_cache[(state, device)] = mask

        print(f"  [precompute] Done: {len(states)} DFA masks in "
              f"{time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_constrained_decoder_lr(
    tokenizer,
    schema: dict,
) -> "ConstrainedDecoder":
    """
    Build a ConstrainedDecoder for a given JSON Schema.

    Args:
        tokenizer: HuggingFace tokenizer with byte_decoder attribute.
        schema: JSON Schema dict.

    Returns:
        ConstrainedDecoder instance ready for use.
    """
    from constrained.schema_compiler import compile_schema, recommended_depth
    from constrained.cfg import BoundedLRAutomaton
    from constrained.scanner import JsonScanner

    # Build token byte mapping
    byte_decoder = tokenizer.byte_decoder
    t2b: dict[int, bytes] = {}
    for token_id in range(tokenizer.vocab_size):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        if token_str is None:
            t2b[token_id] = b""
            continue
        try:
            t2b[token_id] = bytes(byte_decoder[c] for c in token_str)
        except KeyError:
            t2b[token_id] = b""

    # Compile schema
    key_strings, grammar = compile_schema(schema)
    depth = recommended_depth(schema)
    lr_automaton = BoundedLRAutomaton(grammar, depth=depth)
    scanner = JsonScanner(key_strings=key_strings)
    composite = CompositeAutomaton(lr_automaton, scanner)

    trie = TokenTrie(t2b)
    return ConstrainedDecoder(
        automaton=composite,
        trie=trie,
        token_to_bytes=t2b,
        gen_start=0,
        gen_length=0,
        mask_token_id=tokenizer.mask_token_id or 0,
        scanner=scanner,
    )


def build_constrained_decoder(
    tokenizer,
    max_depth: int = 6,
) -> tuple:
    """Legacy DFA factory (unchanged from v2)."""
    from constrained.dfa import build_json_dfa
    byte_decoder = tokenizer.byte_decoder
    t2b: dict[int, bytes] = {}
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