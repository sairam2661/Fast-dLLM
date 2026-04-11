"""
Segment Manager for order-agnostic incremental parsing.

Maintains the collection of segments across the generation region.
Routes token revelations to the correct primitive operation (create,
extend_left, extend_right, merge_with_bridge).

Supports two backends via duck typing on the `automaton` argument:

  DFA (legacy):
    automaton is a DFA instance.
    Segment pairs are (entry_dfa_state, exit_dfa_state) ints.
    automaton.start_state, automaton.accept_states, automaton.num_states.

  BoundedLRAutomaton + JsonScanner (new):
    automaton is a BoundedLRAutomaton.
    scanner is a JsonScanner.
    Segment pairs are (entry_composite, exit_composite) ints, where:
        composite = scanner_state * automaton.num_configs + parser_config
    automaton.start_composite, automaton.accept_composites, automaton.num_composites
    (these are added by build_composite_automaton() below).

In both cases _left_exit_states / _right_entry_states return frozenset[int]
of composite IDs (or plain DFA state ints in legacy mode).
"""

from __future__ import annotations
from typing import Optional, Callable

from constrained.segments import (
    Segment, create, extend_right, extend_left, merge, merge_with_bridge,
)


# ---------------------------------------------------------------------------
# Composite automaton wrapper
# ---------------------------------------------------------------------------

class CompositeAutomaton:
    """
    Wraps a BoundedLRAutomaton + JsonScanner into a single object that
    looks like a DFA to the segment machinery.

    Composite state: integer = scanner_state * num_configs + parser_config.

    Exposes:
      num_states        : total composite states (scanner_states * num_configs)
      start_state       : composite for (scanner.start_state, automaton.start_config)
      accept_states     : frozenset of composites where the scanner is IDLE and
                          parser_config is in automaton.accept_configs
      transition_seq(composite, byte_seq) -> frozenset[int]  (set of composites)

    This is the object passed to segments.py (which calls transition_seq and
    all_configs via duck typing).
    """

    def __init__(self, automaton, scanner):
        self._lr = automaton
        self._sc = scanner
        self.num_configs = automaton.num_configs
        self.num_scanner_states = scanner.num_states
        self.num_states = scanner.num_states * automaton.num_configs

        # Composite for the canonical start: scanner=IDLE, parser=start_config
        self.start_state = scanner.start_state * automaton.num_configs + automaton.start_config

        # Accept composites: scanner is in a state where pending_terminal matches
        # an accept config, or scanner is IDLE and parser is in accept_configs.
        # Most common: scanner=IDLE, parser in accept_configs.
        # Also: scanner=NUM_INT/NUM_FLOAT/WORD_DONE (pending terminal would flush to accept).
        self.accept_states = self._compute_accept_states()

        # Precompute all_configs range (composite IDs are 0..num_states-1)
        self._all_configs = range(self.num_states)

    def _compute_accept_states(self) -> frozenset[int]:
        """
        A composite state is accepting if processing zero more bytes
        (end-of-input) yields an accepted parse.

        This means:
          - Flush any pending terminal from the scanner state
          - Check if the resulting parser config is in accept_configs
        """
        accepts: set[int] = set()
        nc = self.num_configs
        for sc in range(self.num_scanner_states):
            pend = self._sc.pending_terminal(sc)
            for pc in range(nc):
                if pend is not None:
                    # Flush pending terminal
                    nxt = self._lr._trans[pc].get(pend, frozenset())
                    if nxt & self._lr.accept_configs:
                        accepts.add(sc * nc + pc)
                else:
                    # No pending terminal: check if parser is already accepting
                    # and scanner is in IDLE (between lexemes)
                    sc_kind = self._sc.state_kind(sc)
                    if sc_kind == "IDLE" and pc in self._lr.accept_configs:
                        accepts.add(sc * nc + pc)
        return frozenset(accepts)

    def transition_seq(self, composite: int, byte_seq: bytes) -> frozenset[int]:
        """
        Advance a composite state through a byte sequence.
        Returns frozenset of reachable composite states (empty = dead).

        Mirrors BoundedLRAutomaton.transition_seq but over composite states.
        Used by segments.py create/extend operations.
        """
        current: frozenset[int] = frozenset({composite})
        for byte_val in byte_seq:
            nxt: set[int] = set()
            for comp in current:
                sc = comp // self.num_configs
                pc = comp % self.num_configs

                new_sc, terminals = self._sc.step(sc, byte_val)
                if new_sc == self._sc.dead_state:
                    continue

                # Advance parser through all emitted terminals
                cur_pcs: frozenset[int] = frozenset({pc})
                for terminal in terminals:
                    next_pcs: set[int] = set()
                    for c in cur_pcs:
                        next_pcs.update(self._lr._trans[c].get(terminal, frozenset()))
                    if not next_pcs:
                        cur_pcs = frozenset()
                        break
                    cur_pcs = frozenset(next_pcs)

                for new_pc in cur_pcs:
                    nxt.add(new_sc * self.num_configs + new_pc)

            if not nxt:
                return frozenset()
            current = frozenset(nxt)
        return current

    def all_configs(self) -> range:
        return self._all_configs


# ---------------------------------------------------------------------------
# Segment Manager
# ---------------------------------------------------------------------------

class SegmentManager:
    """
    Manages segments across the generation region.

    The `automaton` argument is either a DFA (legacy) or a CompositeAutomaton.
    In both cases segment pairs are (entry_int, exit_int) and the manager
    calls automaton.start_state, automaton.accept_states, automaton.num_states,
    automaton.transition_seq.
    """

    def __init__(
        self,
        automaton,
        gen_start: int,
        gen_length: int,
        token_to_bytes: Callable[[int], bytes],
    ):
        self.automaton = automaton
        self.gen_start = gen_start
        self.gen_length = gen_length
        self.gen_end = gen_start + gen_length - 1
        self.token_to_bytes = token_to_bytes

        self._segments: list[Segment] = []
        self.committed: dict[int, int] = {}
        self._end_to_idx: dict[int, int] = {}
        self._start_to_idx: dict[int, int] = {}

    # Keep dfa as an alias so legacy callers (constrained_decoder) still work
    @property
    def dfa(self):
        return self.automaton

    def reset(self):
        self._segments.clear()
        self.committed.clear()
        self._end_to_idx.clear()
        self._start_to_idx.clear()

    @property
    def num_segments(self): return len(self._segments)
    @property
    def num_committed(self): return len(self.committed)
    @property
    def num_masked(self): return self.gen_length - len(self.committed)

    def get_segments(self) -> list[Segment]:
        return list(self._segments)

    def _find_seg_ending_at(self, pos):
        return self._end_to_idx.get(pos)

    def _find_seg_starting_at(self, pos):
        return self._start_to_idx.get(pos)

    def _insert_segment(self, seg: Segment):
        idx = len(self._segments)
        for i, existing in enumerate(self._segments):
            if existing.start > seg.start:
                idx = i
                break
        self._segments.insert(idx, seg)

    def _remove_indices(self, *indices: int):
        for i in sorted(indices, reverse=True):
            self._segments.pop(i)

    def _rebuild_index(self):
        self._end_to_idx = {seg.end: i for i, seg in enumerate(self._segments)}
        self._start_to_idx = {seg.start: i for i, seg in enumerate(self._segments)}

    # ------------------------------------------------------------------
    # Core operation: reveal a token
    # ------------------------------------------------------------------

    def reveal_token(self, position: int, token_id: int):
        assert position not in self.committed
        assert self.gen_start <= position <= self.gen_end

        token_bytes = self.token_to_bytes(token_id)
        self.committed[position] = token_id

        left_idx = self._end_to_idx.get(position - 1)
        right_idx = self._start_to_idx.get(position + 1)

        if left_idx is None and right_idx is None:
            seg = create(position, token_bytes, self.automaton)
        elif left_idx is not None and right_idx is None:
            left_seg = self._segments[left_idx]
            seg = extend_right(left_seg, position, token_bytes, self.automaton)
            self._segments[left_idx] = seg
            self._rebuild_index()
            return
        elif left_idx is None and right_idx is not None:
            right_seg = self._segments[right_idx]
            seg = extend_left(right_seg, position, token_bytes, self.automaton)
            self._segments[right_idx] = seg
            self._rebuild_index()
            return
        else:
            left_seg = self._segments[left_idx]
            right_seg = self._segments[right_idx]
            seg = merge_with_bridge(left_seg, position, token_bytes, right_seg, self.automaton)
            self._remove_indices(left_idx, right_idx)

        self._insert_segment(seg)
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Valid token queries
    # ------------------------------------------------------------------

    def get_valid_tokens(self, position: int, vocab_size: int) -> set[int]:
        assert position not in self.committed
        left_exits = self._left_exit_states(position)
        right_entries = self._right_entry_states(position)
        valid: set[int] = set()
        for token_id in range(vocab_size):
            token_bytes = self.token_to_bytes(token_id)
            for q in left_exits:
                result = self.automaton.transition_seq(q, token_bytes)
                # result is frozenset (new) or int (legacy DFA)
                hits = result if isinstance(result, frozenset) else (
                    frozenset() if result == -1 else frozenset({result})
                )
                if hits & right_entries:
                    valid.add(token_id)
                    break
        return valid

    def is_token_valid(self, position: int, token_id: int) -> bool:
        assert position not in self.committed
        left_exits = self._left_exit_states(position)
        right_entries = self._right_entry_states(position)
        token_bytes = self.token_to_bytes(token_id)
        for q in left_exits:
            result = self.automaton.transition_seq(q, token_bytes)
            hits = result if isinstance(result, frozenset) else (
                frozenset() if result == -1 else frozenset({result})
            )
            if hits & right_entries:
                return True
        return False

    # ------------------------------------------------------------------
    # Left/right state queries
    # ------------------------------------------------------------------

    def _left_exit_states(self, position: int) -> frozenset[int]:
        idx = self._find_seg_ending_at(position - 1)
        if idx is not None:
            seg = self._segments[idx]
            if seg.start <= self.gen_start:
                # Leftmost segment: only pairs entered from the canonical start state
                start = self.automaton.start_state
                return frozenset(x for e, x in seg.pairs if e == start)
            return seg.exit_configs()
        elif position == self.gen_start:
            return frozenset({self.automaton.start_state})
        else:
            return frozenset(self.automaton.all_configs())

    def _right_entry_states(self, position: int) -> frozenset[int]:
        idx = self._find_seg_starting_at(position + 1)
        if idx is not None:
            seg = self._segments[idx]
            if seg.end >= self.gen_end:
                # Rightmost segment: only pairs that exit into an accept state
                accepts = self.automaton.accept_states
                return frozenset(e for e, x in seg.pairs if x in accepts)
            return seg.entry_configs()
        elif position == self.gen_end:
            return self.automaton.accept_states
        else:
            return frozenset(self.automaton.all_configs())

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------

    def has_empty_segment(self) -> bool:
        return any(len(seg.pairs) == 0 for seg in self._segments)

    def is_valid_complete(self) -> bool:
        if self.num_masked > 0:
            return False
        if len(self._segments) != 1:
            return False
        seg = self._segments[0]
        start = self.automaton.start_state
        accepts = self.automaton.accept_states
        return any(e == start and x in accepts for e, x in seg.pairs)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_with_prompt(self, prompt_bytes: list[bytes]):
        if not prompt_bytes:
            return
        prompt_start = self.gen_start - len(prompt_bytes)
        seg = create(prompt_start, prompt_bytes[0], self.automaton)
        for i, tok_bytes in enumerate(prompt_bytes[1:], start=1):
            seg = extend_right(seg, prompt_start + i, tok_bytes, self.automaton)
        self._insert_segment(seg)
        self._rebuild_index()

    def __repr__(self) -> str:
        segs = ", ".join(
            f"[{s.start}-{s.end}]({len(s.pairs)}p)"
            for s in self._segments
        )
        return (
            f"SegmentManager(committed={self.num_committed}/{self.gen_length}, "
            f"segments=[{segs}])"
        )