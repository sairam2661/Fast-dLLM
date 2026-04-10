"""
Segment data structure for order-agnostic incremental parsing.

A segment represents a contiguous run of committed (unmasked) tokens.
It stores a transition relation: a set of (entry_config, exit_config) pairs
representing all valid parser paths through the tokens in that segment.

Supports two automaton backends via a structural protocol:

  DFA (dfa.py):
    - transition_seq(state: int, bytes) -> int   (DEAD = -1 on failure)
    - all_configs() -> range(num_states)

  BoundedLRAutomaton (cfg.py):
    - transition_seq(config_id: int, bytes) -> frozenset[int]  (empty on failure)
    - all_configs() -> range(num_configs)

The segment operations below handle both backends. The only structural
difference is that BoundedLRAutomaton.transition_seq returns a SET of
successor configs (LR parsing is nondeterministic from a bounded prefix),
so create/extend_right/extend_left fan out over that set. merge/merge_with_bridge
are backend-agnostic: they compose integer pairs.

The Automaton protocol (duck-typed, no ABC):
    num_configs: int          (or num_states for DFA)
    transition_seq(config_id: int, byte_seq: bytes) -> int | frozenset[int]
    all_configs() -> range

Usage:
    # With BoundedLRAutomaton:
    seg = create(pos=5, token_bytes=b'"', automaton=blr)
    seg = extend_right(seg, pos=6, token_bytes=b'a', automaton=blr)
    seg = extend_right(seg, pos=7, token_bytes=b'"', automaton=blr)
    merged = merge(seg, other_seg)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


# Sentinel returned by DFA.transition_seq on failure
_DFA_DEAD = -1


def _successor_configs(automaton: Any, config_id: int, byte_seq: bytes) -> frozenset[int]:
    """
    Uniform interface for both backends.

    DFA:             transition_seq returns int; wrap in frozenset, empty on DEAD.
    BoundedLRAutomaton: transition_seq returns frozenset[int]; pass through.
    """
    result = automaton.transition_seq(config_id, byte_seq)
    if isinstance(result, int):
        return frozenset() if result == _DFA_DEAD else frozenset({result})
    return result  # already frozenset


@dataclass(frozen=True)
class Segment:
    """
    A contiguous run of committed tokens with tracked parser state.

    Attributes:
        start: position of first token (inclusive)
        end: position of last token (inclusive)
        pairs: frozenset of (entry_config, exit_config) pairs representing
               all valid parser paths through this segment's tokens.
               Config IDs are integers assigned by the automaton backend.
    """
    start: int
    end: int
    pairs: frozenset[tuple[int, int]]

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def entry_configs(self) -> frozenset[int]:
        """All possible entry configs for this segment."""
        return frozenset(e for e, _ in self.pairs)

    def exit_configs(self) -> frozenset[int]:
        """All possible exit configs for this segment."""
        return frozenset(x for _, x in self.pairs)

    def exits_for_entry(self, entry: int) -> frozenset[int]:
        """Exit configs reachable from a specific entry config."""
        return frozenset(x for e, x in self.pairs if e == entry)

    def entries_for_exit(self, exit_config: int) -> frozenset[int]:
        """Entry configs that can reach a specific exit config."""
        return frozenset(e for e, x in self.pairs if x == exit_config)


def create(pos: int, token_bytes: bytes, automaton: Any) -> Segment:
    """
    Create a length-1 segment for a newly unmasked isolated token.

    Tries every possible entry config and records which (entry, exit) pairs
    are reachable by processing the token's bytes.

    For BoundedLRAutomaton, transition_seq returns a set of successors,
    so each entry config may contribute multiple (entry, exit) pairs.

    Args:
        pos: position index of the token
        token_bytes: byte content of the token
        automaton: DFA or BoundedLRAutomaton

    Returns:
        A new Segment at the given position.
    """
    pairs: set[tuple[int, int]] = set()
    for q in automaton.all_configs():
        for exit_q in _successor_configs(automaton, q, token_bytes):
            pairs.add((q, exit_q))
    return Segment(start=pos, end=pos, pairs=frozenset(pairs))


def extend_right(seg: Segment, pos: int, token_bytes: bytes, automaton: Any) -> Segment:
    """
    Extend segment by one token to the right.

    For each existing (entry, exit) pair, apply the new token's byte
    transitions to the exit config. Each exit config may fan out to
    multiple successors (BoundedLRAutomaton case).

    Args:
        seg: existing segment
        pos: position of the new token (must be seg.end + 1)
        token_bytes: byte content of the new token
        automaton: DFA or BoundedLRAutomaton

    Returns:
        A new Segment extended by one position to the right.
    """
    assert pos == seg.end + 1, (
        f"extend_right: pos {pos} must be seg.end + 1 = {seg.end + 1}"
    )
    new_pairs: set[tuple[int, int]] = set()
    for entry, exit_config in seg.pairs:
        for new_exit in _successor_configs(automaton, exit_config, token_bytes):
            new_pairs.add((entry, new_exit))
    return Segment(start=seg.start, end=pos, pairs=frozenset(new_pairs))


def extend_left(seg: Segment, pos: int, token_bytes: bytes, automaton: Any) -> Segment:
    """
    Extend segment by one token to the left.

    Finds all configs q such that processing token_bytes from q reaches
    one of the segment's current entry configs. Each such q becomes a new
    entry config paired with the corresponding exits.

    For BoundedLRAutomaton, transition_seq fans out, so one candidate
    entry config may reach multiple current entry configs.

    Args:
        seg: existing segment
        pos: position of the new token (must be seg.start - 1)
        token_bytes: byte content of the new token
        automaton: DFA or BoundedLRAutomaton

    Returns:
        A new Segment extended by one position to the left.
    """
    assert pos == seg.start - 1, (
        f"extend_left: pos {pos} must be seg.start - 1 = {seg.start - 1}"
    )
    current_entries = seg.entry_configs()

    # For each candidate predecessor config q, compute which of the
    # segment's current entry configs are reachable from q via token_bytes.
    # Build: reached_entry -> set of candidate q's
    predecessors: dict[int, set[int]] = {}
    for q in automaton.all_configs():
        for reached in _successor_configs(automaton, q, token_bytes):
            if reached in current_entries:
                predecessors.setdefault(reached, set()).add(q)

    new_pairs: set[tuple[int, int]] = set()
    for entry, exit_config in seg.pairs:
        for new_entry in predecessors.get(entry, ()):
            new_pairs.add((new_entry, exit_config))

    return Segment(start=pos, end=seg.end, pairs=frozenset(new_pairs))


def merge(left: Segment, right: Segment) -> Segment:
    """
    Merge two directly adjacent segments (no gap between them).

    Composes the transition relations: a merged pair (e_L, x_R) exists
    iff there is some intermediate config m where (e_L, m) is in the left
    segment and (m, x_R) is in the right segment.

    Backend-agnostic: operates on integer pairs only.

    Args:
        left: left segment
        right: right segment (must satisfy left.end + 1 == right.start)

    Returns:
        A new Segment spanning both, with composed transition relation.
    """
    assert left.end + 1 == right.start, (
        f"merge: segments must be adjacent. left.end={left.end}, right.start={right.start}"
    )
    # Index right segment by entry config for O(1) lookup
    right_by_entry: dict[int, set[int]] = {}
    for e_r, x_r in right.pairs:
        right_by_entry.setdefault(e_r, set()).add(x_r)

    new_pairs: set[tuple[int, int]] = set()
    for e_l, x_l in left.pairs:
        for x_r in right_by_entry.get(x_l, ()):
            new_pairs.add((e_l, x_r))

    return Segment(start=left.start, end=right.end, pairs=frozenset(new_pairs))


def merge_with_bridge(
    left: Segment,
    bridge_pos: int,
    bridge_bytes: bytes,
    right: Segment,
    automaton: Any,
) -> Segment:
    """
    Merge two segments separated by a single bridging token.

    Equivalent to extend_right(left, bridge_bytes) then merge(extended, right).

    Args:
        left: left segment
        bridge_pos: position of the bridging token (must be left.end + 1)
        bridge_bytes: byte content of the bridging token
        right: right segment (must satisfy bridge_pos + 1 == right.start)
        automaton: DFA or BoundedLRAutomaton

    Returns:
        A new Segment spanning left + bridge + right.
    """
    assert left.end + 1 == bridge_pos, (
        f"merge_with_bridge: bridge_pos {bridge_pos} must be left.end + 1 = {left.end + 1}"
    )
    assert bridge_pos + 1 == right.start, (
        f"merge_with_bridge: right.start {right.start} must be bridge_pos + 1 = {bridge_pos + 1}"
    )
    extended = extend_right(left, bridge_pos, bridge_bytes, automaton)
    return merge(extended, right)