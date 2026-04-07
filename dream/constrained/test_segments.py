"""
Unit tests for segments.py.

Tests:
1. Segment primitives on a simple (ab)* DFA
2. Transition relation correctness against brute-force
3. Order independence: all permutations of token revelation produce
   the same final transition relation
4. Order independence on a JSON string
5. Merge with bridge operation
6. Empty pairs detection for invalid sequences
7. extend_left correctness

Run:
    python test_segments.py
"""

from itertools import permutations
from dfa import build_json_dfa, DFA, DEAD
from segments import Segment, create, extend_right, extend_left, merge, merge_with_bridge


# ---------------------------------------------------------------------------
# Helper: tiny (ab)* DFA for easy reasoning
# ---------------------------------------------------------------------------

def build_ab_star_dfa() -> DFA:
    """DFA accepting (ab)*. States: 0 (start/accept), 1 (saw a)."""
    transitions = {
        (0, ord('a')): 1,
        (1, ord('b')): 0,
    }
    return DFA.from_transitions(num_states=2, start=0, accept={0},
                                transitions=transitions)


# ---------------------------------------------------------------------------
# Helper: brute-force transition relation for comparison
# ---------------------------------------------------------------------------

def brute_force_pairs(byte_sequence: list[bytes], dfa: DFA) -> frozenset[tuple[int, int]]:
    """
    Compute the correct transition relation for a sequence of tokens
    by trying every possible entry state.
    """
    pairs = set()
    for q in range(dfa.num_states):
        state = q
        for token_bytes in byte_sequence:
            state = dfa.transition_seq(state, token_bytes)
            if state == DEAD:
                break
        if state != DEAD:
            pairs.add((q, state))
    return frozenset(pairs)


# ---------------------------------------------------------------------------
# Helper: build final segment from a permutation of (position, bytes) pairs
# ---------------------------------------------------------------------------

def build_from_order(positions_and_bytes: list[tuple[int, bytes]], dfa: DFA,
                     order: tuple[int, ...]) -> Segment:
    """
    Simulate revealing tokens in the given order, using the correct
    primitive operations (create / extend_left / extend_right / merge).

    Returns the final segment after all tokens are revealed.
    """
    # Track segments by their start position
    segments: dict[int, Segment] = {}  # start_pos -> Segment
    committed: set[int] = set()

    def find_seg_ending_at(pos):
        for seg in segments.values():
            if seg.end == pos:
                return seg
        return None

    def find_seg_starting_at(pos):
        return segments.get(pos)

    def remove_seg(seg):
        del segments[seg.start]

    def add_seg(seg):
        segments[seg.start] = seg

    for idx in order:
        pos, token_bytes = positions_and_bytes[idx]
        committed.add(pos)

        left_seg = find_seg_ending_at(pos - 1)
        right_seg = find_seg_starting_at(pos + 1)

        if left_seg is None and right_seg is None:
            add_seg(create(pos, token_bytes, dfa))

        elif left_seg is not None and right_seg is None:
            remove_seg(left_seg)
            add_seg(extend_right(left_seg, pos, token_bytes, dfa))

        elif left_seg is None and right_seg is not None:
            remove_seg(right_seg)
            add_seg(extend_left(right_seg, pos, token_bytes, dfa))

        else:
            # Bridge: extend left then merge
            remove_seg(left_seg)
            remove_seg(right_seg)
            seg = merge_with_bridge(left_seg, pos, token_bytes, right_seg, dfa)
            add_seg(seg)

    assert len(segments) == 1, f"Expected 1 final segment, got {len(segments)}"
    return list(segments.values())[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create():
    dfa = build_ab_star_dfa()

    seg_a = create(5, b'a', dfa)
    assert seg_a.start == 5 and seg_a.end == 5
    # state 0 + 'a' -> 1;  state 1 + 'a' -> DEAD
    assert seg_a.pairs == frozenset({(0, 1)}), f"Got {seg_a.pairs}"
    print("  PASS: create 'a'")

    seg_b = create(3, b'b', dfa)
    assert seg_b.pairs == frozenset({(1, 0)}), f"Got {seg_b.pairs}"
    print("  PASS: create 'b'")


def test_extend_right():
    dfa = build_ab_star_dfa()

    seg = create(0, b'a', dfa)          # pairs: {(0,1)}
    seg = extend_right(seg, 1, b'b', dfa)  # 'ab': 0->1->0
    assert seg.pairs == frozenset({(0, 0)}), f"Got {seg.pairs}"
    assert seg.start == 0 and seg.end == 1
    print("  PASS: extend_right 'a'+'b' = 'ab'")

    seg = extend_right(seg, 2, b'a', dfa)  # 'aba': 0->1->0->1
    assert seg.pairs == frozenset({(0, 1)}), f"Got {seg.pairs}"
    print("  PASS: extend_right 'ab'+'a' = 'aba'")

    seg = extend_right(seg, 3, b'b', dfa)  # 'abab': 0->1->0->1->0
    assert seg.pairs == frozenset({(0, 0)}), f"Got {seg.pairs}"
    print("  PASS: extend_right 'aba'+'b' = 'abab'")


def test_extend_left():
    dfa = build_ab_star_dfa()

    seg = create(5, b'b', dfa)            # pairs: {(1,0)}
    seg = extend_left(seg, 4, b'a', dfa)  # 'ab': 0->1->0
    assert seg.pairs == frozenset({(0, 0)}), f"Got {seg.pairs}"
    assert seg.start == 4 and seg.end == 5
    print("  PASS: extend_left 'a'+'b' = 'ab'")

    seg = extend_left(seg, 3, b'b', dfa)  # 'bab': 1->0->1->0
    assert seg.pairs == frozenset({(1, 0)}), f"Got {seg.pairs}"
    print("  PASS: extend_left 'b'+'ab' = 'bab'")


def test_merge():
    dfa = build_ab_star_dfa()

    # 'ab' at 0-1 merged with 'ab' at 2-3
    left = create(0, b'a', dfa)
    left = extend_right(left, 1, b'b', dfa)  # {(0,0)}
    right = create(2, b'a', dfa)
    right = extend_right(right, 3, b'b', dfa)  # {(0,0)}

    merged = merge(left, right)
    # 'abab': only works if left exit (0) == right entry (0). Yes.
    assert merged.pairs == frozenset({(0, 0)}), f"Got {merged.pairs}"
    assert merged.start == 0 and merged.end == 3
    print("  PASS: merge 'ab'+'ab' = 'abab'")

    # Incompatible: 'ab' at 0-1 merged with 'ba' at 2-3
    right2 = create(2, b'b', dfa)
    right2 = extend_right(right2, 3, b'a', dfa)  # {(1,1)}
    merged2 = merge(left, right2)
    # left exit 0, right entry 1 — mismatch
    assert merged2.pairs == frozenset(), f"Got {merged2.pairs}"
    print("  PASS: merge 'ab'+'ba' = empty (incompatible)")


def test_merge_with_bridge():
    dfa = build_ab_star_dfa()

    left = create(0, b'a', dfa)   # pos 0, pairs {(0,1)}
    right = create(2, b'a', dfa)  # pos 2, pairs {(0,1)}

    # Bridge with 'b' at pos 1: 'a'+'b'+'a' = 'aba'
    merged = merge_with_bridge(left, 1, b'b', right, dfa)
    assert merged.pairs == frozenset({(0, 1)}), f"Got {merged.pairs}"
    assert merged.start == 0 and merged.end == 2
    print("  PASS: merge_with_bridge 'a'+'b'+'a' = 'aba'")

    # Bridge with 'a' at pos 1: 'a'+'a'+'a' = 'aaa' (invalid in (ab)*)
    merged2 = merge_with_bridge(left, 1, b'a', right, dfa)
    assert merged2.pairs == frozenset(), f"Got {merged2.pairs}"
    print("  PASS: merge_with_bridge 'a'+'a'+'a' = empty (invalid)")


def test_brute_force_agreement():
    """Segment operations agree with brute-force on the (ab)* DFA."""
    dfa = build_ab_star_dfa()

    sequences = [
        [b'a'],
        [b'a', b'b'],
        [b'a', b'b', b'a', b'b'],
        [b'b', b'a'],
        [b'a', b'a'],  # invalid
        [b'a', b'b', b'a'],
    ]

    for seq in sequences:
        expected = brute_force_pairs(seq, dfa)
        # Build via extend_right
        seg = create(0, seq[0], dfa)
        for i, tok in enumerate(seq[1:], start=1):
            seg = extend_right(seg, i, tok, dfa)
        assert seg.pairs == expected, (
            f"Mismatch for {seq}: segment={seg.pairs}, brute_force={expected}"
        )

    print("  PASS: all sequences match brute force")


def test_order_independence_ab():
    """All permutations of 'abab' produce the same transition relation."""
    dfa = build_ab_star_dfa()
    tokens = [(0, b'a'), (1, b'b'), (2, b'a'), (3, b'b')]
    expected = brute_force_pairs([b'a', b'b', b'a', b'b'], dfa)

    count = 0
    for perm in permutations(range(4)):
        seg = build_from_order(tokens, dfa, perm)
        assert seg.pairs == expected, (
            f"Order {perm}: got {seg.pairs}, expected {expected}"
        )
        count += 1

    print(f"  PASS: all {count} orderings of 'abab' give {expected}")


def test_order_independence_json():
    """All permutations of a short JSON string produce the same pairs."""
    dfa = build_json_dfa(max_depth=4)

    json_str = '{"a":1}'
    tokens = [(i, bytes([b])) for i, b in enumerate(json_str.encode())]
    expected = brute_force_pairs([t[1] for t in tokens], dfa)

    count = 0
    for perm in permutations(range(len(tokens))):
        seg = build_from_order(tokens, dfa, perm)
        assert seg.pairs == expected, (
            f"Order {perm}: got {seg.pairs}, expected {expected}"
        )
        count += 1

    # Verify the sequence is actually valid (start -> accept)
    has_valid = any(e == dfa.start_state and x in dfa.accept_states
                    for e, x in expected)
    assert has_valid, f"No valid path in {expected}"
    print(f"  PASS: all {count} orderings of '{json_str}' consistent")


def test_order_independence_json_longer():
    """Order independence for a longer JSON string (sampled permutations)."""
    import random
    random.seed(42)

    dfa = build_json_dfa(max_depth=4)

    json_str = '{"k":[1,2]}'
    tokens = [(i, bytes([b])) for i, b in enumerate(json_str.encode())]
    expected = brute_force_pairs([t[1] for t in tokens], dfa)
    n = len(tokens)

    # 11! = 39916800 — too many. Sample 1000 random permutations.
    num_samples = 1000
    for _ in range(num_samples):
        perm = tuple(random.sample(range(n), n))
        seg = build_from_order(tokens, dfa, perm)
        assert seg.pairs == expected, (
            f"Order {perm}: got {seg.pairs}, expected {expected}"
        )

    has_valid = any(e == dfa.start_state and x in dfa.accept_states
                    for e, x in expected)
    assert has_valid
    print(f"  PASS: {num_samples} random orderings of '{json_str}' consistent")


def test_empty_pairs_detection():
    """Invalid token sequences produce empty transition relations."""
    dfa = build_ab_star_dfa()

    # 'aaa' — invalid in (ab)*
    seg = create(0, b'a', dfa)
    seg = extend_right(seg, 1, b'a', dfa)
    seg = extend_right(seg, 2, b'a', dfa)
    assert len(seg.pairs) == 0, f"Expected empty pairs for 'aaa', got {seg.pairs}"
    print("  PASS: 'aaa' -> empty pairs")

    # 'bb' — invalid
    seg2 = create(0, b'b', dfa)
    seg2 = extend_right(seg2, 1, b'b', dfa)
    assert len(seg2.pairs) == 0, f"Expected empty pairs for 'bb', got {seg2.pairs}"
    print("  PASS: 'bb' -> empty pairs")


def test_entry_exit_helpers():
    """Segment helper methods return correct state subsets."""
    dfa = build_ab_star_dfa()

    # 'ab' at 0-1: pairs {(0, 0)}
    seg = create(0, b'a', dfa)
    seg = extend_right(seg, 1, b'b', dfa)

    assert seg.entry_states() == frozenset({0})
    assert seg.exit_states() == frozenset({0})
    assert seg.exits_for_entry(0) == frozenset({0})
    assert seg.exits_for_entry(1) == frozenset()
    assert seg.entries_for_exit(0) == frozenset({0})
    assert seg.entries_for_exit(1) == frozenset()
    print("  PASS: entry/exit helpers")


def test_multibyte_tokens():
    """Segments work with multi-byte tokens (simulating BPE tokens)."""
    dfa = build_ab_star_dfa()

    # Token "ab" as a single 2-byte token
    seg = create(0, b'ab', dfa)
    # From state 0: a->1, b->0. So (0, 0).
    # From state 1: a->DEAD.
    assert seg.pairs == frozenset({(0, 0)}), f"Got {seg.pairs}"
    print("  PASS: multi-byte token 'ab'")

    # Token "abab" as a single 4-byte token
    seg2 = create(0, b'abab', dfa)
    assert seg2.pairs == frozenset({(0, 0)}), f"Got {seg2.pairs}"
    print("  PASS: multi-byte token 'abab'")

    # Token "ba" — from state 1: b->0, a->1. So (1, 1).
    seg3 = create(0, b'ba', dfa)
    assert seg3.pairs == frozenset({(1, 1)}), f"Got {seg3.pairs}"
    print("  PASS: multi-byte token 'ba'")

    # Extend 'ab' (pos 0) right with 'ab' (pos 1) — "abab"
    seg4 = create(0, b'ab', dfa)
    seg4 = extend_right(seg4, 1, b'ab', dfa)
    assert seg4.pairs == frozenset({(0, 0)}), f"Got {seg4.pairs}"
    print("  PASS: extend_right multi-byte 'ab'+'ab'")


if __name__ == '__main__':
    print("=== create ===")
    test_create()

    print("\n=== extend_right ===")
    test_extend_right()

    print("\n=== extend_left ===")
    test_extend_left()

    print("\n=== merge ===")
    test_merge()

    print("\n=== merge_with_bridge ===")
    test_merge_with_bridge()

    print("\n=== brute_force_agreement ===")
    test_brute_force_agreement()

    print("\n=== order_independence (ab*) ===")
    test_order_independence_ab()

    print("\n=== order_independence (JSON short) ===")
    test_order_independence_json()

    print("\n=== order_independence (JSON longer, sampled) ===")
    test_order_independence_json_longer()

    print("\n=== empty_pairs_detection ===")
    test_empty_pairs_detection()

    print("\n=== entry/exit helpers ===")
    test_entry_exit_helpers()

    print("\n=== multi-byte tokens ===")
    test_multibyte_tokens()

    print("\nAll tests passed.")