"""
Segment data structure for order-agnostic incremental parsing.

A segment represents a contiguous run of committed (unmasked) tokens.
It stores a transition relation: a set of (entry_state, exit_state) pairs
representing all valid DFA paths through the tokens in that segment.

Three primitive operations:
  - create: new length-1 segment for an isolated token
  - extend_left / extend_right: grow a segment by one position
  - merge: combine two adjacent segments

The transition relation tracks which DFA entry states can produce which
exit states. This prevents false positives that would arise from tracking
entry and exit state sets independently — a path must be consistent from
entry to exit through the entire segment.

Usage:
    seg = create(pos=5, token_bytes=b'"', dfa=dfa)
    seg = extend_right(seg, pos=6, token_bytes=b'a', dfa=dfa)
    seg = extend_right(seg, pos=7, token_bytes=b'"', dfa=dfa)
    # seg now represents '"a"' at positions 5-7
    # seg.pairs tells you which DFA entry states can traverse this segment
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from constrained.dfa import DFA, DEAD


@dataclass(frozen=True)
class Segment:
    """
    A contiguous run of committed tokens with tracked DFA state.

    Attributes:
        start: position of first token (inclusive)
        end: position of last token (inclusive)
        pairs: frozenset of (entry_state, exit_state) pairs representing
               all valid DFA paths through this segment's tokens.
    """
    start: int
    end: int
    pairs: frozenset[tuple[int, int]]

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def entry_states(self) -> frozenset[int]:
        """All possible entry states for this segment."""
        return frozenset(e for e, _ in self.pairs)

    def exit_states(self) -> frozenset[int]:
        """All possible exit states for this segment."""
        return frozenset(x for _, x in self.pairs)

    def exits_for_entry(self, entry: int) -> frozenset[int]:
        """Exit states reachable from a specific entry state."""
        return frozenset(x for e, x in self.pairs if e == entry)

    def entries_for_exit(self, exit_state: int) -> frozenset[int]:
        """Entry states that can reach a specific exit state."""
        return frozenset(e for e, x in self.pairs if x == exit_state)


def create(pos: int, token_bytes: bytes, dfa: DFA) -> Segment:
    """
    Create a length-1 segment for a newly unmasked isolated token.

    Tries every possible DFA entry state and records which (entry, exit)
    pairs are reachable by processing the token's bytes.

    Args:
        pos: position index of the token
        token_bytes: byte content of the token
        dfa: the constraint DFA

    Returns:
        A new Segment at the given position.
    """
    pairs = set()
    for q in range(dfa.num_states):
        exit_q = dfa.transition_seq(q, token_bytes)
        if exit_q != DEAD:
            pairs.add((q, exit_q))
    return Segment(start=pos, end=pos, pairs=frozenset(pairs))


def extend_right(seg: Segment, pos: int, token_bytes: bytes, dfa: DFA) -> Segment:
    """
    Extend segment by one token to the right.

    For each existing (entry, exit) pair, apply the new token's byte
    transitions to the exit state. Keep pairs where the transition succeeds.

    Args:
        seg: existing segment
        pos: position of the new token (must be seg.end + 1)
        token_bytes: byte content of the new token
        dfa: the constraint DFA

    Returns:
        A new Segment extended by one position to the right.
    """
    assert pos == seg.end + 1, (
        f"extend_right: pos {pos} must be seg.end + 1 = {seg.end + 1}"
    )
    new_pairs = set()
    for entry, exit_state in seg.pairs:
        new_exit = dfa.transition_seq(exit_state, token_bytes)
        if new_exit != DEAD:
            new_pairs.add((entry, new_exit))
    return Segment(start=seg.start, end=pos, pairs=frozenset(new_pairs))


def extend_left(seg: Segment, pos: int, token_bytes: bytes, dfa: DFA) -> Segment:
    """
    Extend segment by one token to the left.

    Finds all DFA states q such that processing token_bytes from q
    reaches one of the segment's current entry states. Each such q
    becomes a new entry state paired with the corresponding exits.

    Args:
        seg: existing segment
        pos: position of the new token (must be seg.start - 1)
        token_bytes: byte content of the new token
        dfa: the constraint DFA

    Returns:
        A new Segment extended by one position to the left.
    """
    assert pos == seg.start - 1, (
        f"extend_left: pos {pos} must be seg.start - 1 = {seg.start - 1}"
    )
    # Build map: for each current entry state, which states q can reach it
    # via token_bytes?
    entry_states = seg.entry_states()
    # predecessors[e] = set of q such that transition_seq(q, token_bytes) == e
    predecessors: dict[int, set[int]] = {e: set() for e in entry_states}
    for q in range(dfa.num_states):
        result = dfa.transition_seq(q, token_bytes)
        if result != DEAD and result in entry_states:
            predecessors[result].add(q)

    new_pairs = set()
    for entry, exit_state in seg.pairs:
        for new_entry in predecessors.get(entry, ()):
            new_pairs.add((new_entry, exit_state))

    return Segment(start=pos, end=seg.end, pairs=frozenset(new_pairs))


def merge(left: Segment, right: Segment) -> Segment:
    """
    Merge two directly adjacent segments (no gap between them).

    Composes the transition relations: a merged pair (e_L, x_R) exists
    iff there is some intermediate state m where (e_L, m) is in the left
    segment and (m, x_R) is in the right segment.

    Args:
        left: left segment
        right: right segment (must satisfy left.end + 1 == right.start)

    Returns:
        A new Segment spanning both, with composed transition relation.
    """
    assert left.end + 1 == right.start, (
        f"merge: segments must be adjacent. left.end={left.end}, right.start={right.start}"
    )
    # Index right segment entries for fast lookup
    right_by_entry: dict[int, set[int]] = {}
    for e_r, x_r in right.pairs:
        right_by_entry.setdefault(e_r, set()).add(x_r)

    new_pairs = set()
    for e_l, x_l in left.pairs:
        # left exit must match right entry
        if x_l in right_by_entry:
            for x_r in right_by_entry[x_l]:
                new_pairs.add((e_l, x_r))

    return Segment(start=left.start, end=right.end, pairs=frozenset(new_pairs))


def merge_with_bridge(
    left: Segment, bridge_pos: int, bridge_bytes: bytes,
    right: Segment, dfa: DFA,
) -> Segment:
    """
    Merge two segments separated by a single bridging token.

    Equivalent to extend_right(left, bridge_bytes) followed by
    merge(extended, right), but bundled for clarity.

    Args:
        left: left segment
        bridge_pos: position of the bridging token (must be left.end + 1)
        bridge_bytes: byte content of the bridging token
        right: right segment (must satisfy bridge_pos + 1 == right.start)
        dfa: the constraint DFA

    Returns:
        A new Segment spanning left + bridge + right.
    """
    assert left.end + 1 == bridge_pos, (
        f"merge_with_bridge: bridge_pos {bridge_pos} must be left.end + 1 = {left.end + 1}"
    )
    assert bridge_pos + 1 == right.start, (
        f"merge_with_bridge: right.start {right.start} must be bridge_pos + 1 = {bridge_pos + 1}"
    )
    extended = extend_right(left, bridge_pos, bridge_bytes, dfa)
    return merge(extended, right)