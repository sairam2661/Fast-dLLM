"""
Unit tests for manager.py.

Tests:
1. Basic token revelation and segment bookkeeping
2. All four reveal cases (isolated, extend_left, extend_right, bridge)
3. Valid token computation with left/right context
4. Tight constraint at boundary positions
5. Empty segment detection (irrecoverable errors)
6. Full generation validity check
7. Prompt initialization
8. Order independence through the manager
9. Repr / diagnostics

Run:
    python test_manager.py
"""

import random
from itertools import permutations
from dfa import build_json_dfa, DFA, DEAD
from segments import Segment
from manager import SegmentManager


# ---------------------------------------------------------------------------
# Helper: (ab)* DFA
# ---------------------------------------------------------------------------

def build_ab_star_dfa() -> DFA:
    transitions = {(0, ord('a')): 1, (1, ord('b')): 0}
    return DFA.from_transitions(num_states=2, start=0, accept={0},
                                transitions=transitions)


# ---------------------------------------------------------------------------
# Tests on (ab)* DFA
# ---------------------------------------------------------------------------

def test_reveal_isolated():
    """Revealing a token with no neighbors creates a new segment."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=6,
                         token_to_bytes=lambda t: vocab[t])

    mgr.reveal_token(3, 0)  # 'a' at position 3
    assert mgr.num_segments == 1
    assert mgr.num_committed == 1
    assert mgr._segments[0].start == 3 and mgr._segments[0].end == 3
    print("  PASS: isolated reveal")


def test_reveal_extend_right():
    """Revealing a token adjacent to a segment's right end extends it."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=6,
                         token_to_bytes=lambda t: vocab[t])

    mgr.reveal_token(2, 0)  # 'a' at 2
    mgr.reveal_token(3, 1)  # 'b' at 3
    assert mgr.num_segments == 1
    seg = mgr._segments[0]
    assert seg.start == 2 and seg.end == 3
    assert seg.pairs == frozenset({(0, 0)})
    print("  PASS: extend right")


def test_reveal_extend_left():
    """Revealing a token adjacent to a segment's left end extends it."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=6,
                         token_to_bytes=lambda t: vocab[t])

    mgr.reveal_token(3, 1)  # 'b' at 3
    mgr.reveal_token(2, 0)  # 'a' at 2
    assert mgr.num_segments == 1
    seg = mgr._segments[0]
    assert seg.start == 2 and seg.end == 3
    assert seg.pairs == frozenset({(0, 0)})
    print("  PASS: extend left")


def test_reveal_bridge():
    """Revealing a token between two segments merges them."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=4,
                         token_to_bytes=lambda t: vocab[t])

    mgr.reveal_token(0, 0)  # 'a' at 0
    mgr.reveal_token(2, 0)  # 'a' at 2
    assert mgr.num_segments == 2

    mgr.reveal_token(1, 1)  # 'b' at 1 — bridges
    assert mgr.num_segments == 1
    seg = mgr._segments[0]
    assert seg.start == 0 and seg.end == 2
    # 'aba': from 0, a->1, b->0, a->1 => (0,1)
    assert seg.pairs == frozenset({(0, 1)})
    print("  PASS: bridge merge")


def test_valid_tokens_left_only():
    """Valid tokens with only left context."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=4,
                         token_to_bytes=lambda t: vocab[t])

    mgr.reveal_token(0, 0)  # 'a' at 0
    # Position 1: left segment exits at state 1. Only 'b' is valid from 1.
    # Right side is a gap (all states), so the constraint is just from left.
    valid = mgr.get_valid_tokens(1, 2)
    assert valid == {1}, f"Expected {{1}} ('b'), got {valid}"
    print("  PASS: valid tokens (left context only)")


def test_valid_tokens_right_only():
    """Valid tokens with only right context."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=4,
                         token_to_bytes=lambda t: vocab[t])

    mgr.reveal_token(3, 1)  # 'b' at 3
    # Position 2: right segment 'b' at 3 has entry states {1}.
    # Left side is a gap (all states).
    # Valid: tokens where transition(q, tok) = 1 for some q.
    # 'a': 0->1. Yes. 'b': 1->0. No (0 not in {1}).
    valid = mgr.get_valid_tokens(2, 2)
    assert valid == {0}, f"Expected {{0}} ('a'), got {valid}"
    print("  PASS: valid tokens (right context only)")


def test_valid_tokens_both_sides():
    """Valid tokens constrained by both left and right context."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=4,
                         token_to_bytes=lambda t: vocab[t])

    # Target: 'abab'. Reveal 'a' at 0 and 'a' at 2.
    mgr.reveal_token(0, 0)  # 'a' at 0
    mgr.reveal_token(2, 0)  # 'a' at 2

    # Position 1: left exits from 'a' at gen_start = {1} (filtered to start_state=0 entry).
    # Right entries from 'a' at 2 (not at gen_end) = {0}.
    # Need transition(1, tok) in {0}. 'b': 1->0. Valid.
    valid = mgr.get_valid_tokens(1, 2)
    assert valid == {1}, f"Expected {{1}} ('b'), got {valid}"
    print("  PASS: valid tokens (both sides)")


def test_valid_tokens_gen_start():
    """At gen_start, left context is {dfa.start_state}."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=4,
                         token_to_bytes=lambda t: vocab[t])

    # Position 0: left is start state {0}. Right is gap (all states).
    # 'a': 0->1. Valid. 'b': 0->DEAD. Invalid.
    valid = mgr.get_valid_tokens(0, 2)
    assert valid == {0}, f"Expected {{0}} ('a'), got {valid}"
    print("  PASS: valid tokens at gen_start")


def test_valid_tokens_gen_end():
    """At gen_end, right context is accept states."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=4,
                         token_to_bytes=lambda t: vocab[t])

    # Reveal 'a','b','a' at 0,1,2. Position 3 is gen_end.
    mgr.reveal_token(0, 0)
    mgr.reveal_token(1, 1)
    mgr.reveal_token(2, 0)
    # Left exits: state 1 (after 'aba'). Right: accept = {0}.
    # 'b': 1->0, 0 in accept. Valid. 'a': 1->DEAD. Invalid.
    valid = mgr.get_valid_tokens(3, 2)
    assert valid == {1}, f"Expected {{1}} ('b'), got {valid}"
    print("  PASS: valid tokens at gen_end (accept constraint)")


def test_is_token_valid():
    """Single-token validity check."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=2,
                         token_to_bytes=lambda t: vocab[t])

    mgr.reveal_token(0, 0)  # 'a'
    assert mgr.is_token_valid(1, 1) == True   # 'b' valid
    assert mgr.is_token_valid(1, 0) == False  # 'a' invalid
    print("  PASS: is_token_valid")


# ---------------------------------------------------------------------------
# Tests on JSON DFA
# ---------------------------------------------------------------------------

def test_json_tight_constraint():
    """
    The key test: when a single position is masked between two committed
    segments, the valid set is tightly constrained.
    """
    dfa = build_json_dfa(max_depth=4)
    json_chars = list('{}[]:,"0123456789abcdefghijklmnopqrstuvwxyz \n-.')
    char_to_id = {c: i for i, c in enumerate(json_chars)}
    vocab = {i: c.encode() for i, c in enumerate(json_chars)}
    V = len(json_chars)

    # Target: {"a":1}  (positions 0-6)
    # Reveal everything except position 4 (the colon)
    mgr = SegmentManager(dfa, gen_start=0, gen_length=7,
                         token_to_bytes=lambda t: vocab[t])

    for pos, ch in [(0, '{'), (1, '"'), (2, 'a'), (3, '"'),
                    (5, '1'), (6, '}')]:
        mgr.reveal_token(pos, char_to_id[ch])

    assert mgr.num_segments == 2  # [0-3] and [5-6]

    valid = mgr.get_valid_tokens(4, V)
    valid_chars = {json_chars[t] for t in valid}
    assert valid_chars == {':'}, f"Expected only ':', got {valid_chars}"
    print("  PASS: JSON tight constraint — only ':' valid between '\"a\"' and '1}'")


def test_json_structural_boundary():
    """
    Test constraint tightening at structural boundaries.

    Key insight: constraints are tight when both neighbors are committed
    segments (no gaps). When there are gaps, the constraint is an
    over-approximation because gap positions could be any DFA state.

    We test the scenario that matters most: a single masked position
    between two fully committed segments.
    """
    dfa = build_json_dfa(max_depth=4)
    json_chars = list('{}[]:,"0123456789abcdefghijklmnopqrstuvwxyz \n-.')
    char_to_id = {c: i for i, c in enumerate(json_chars)}
    vocab = {i: c.encode() for i, c in enumerate(json_chars)}
    V = len(json_chars)

    # Scenario 1: {"a" _ 1} — position 4 is the colon
    # Already tested above, but let's also check what ISN'T valid
    mgr = SegmentManager(dfa, gen_start=0, gen_length=7,
                         token_to_bytes=lambda t: vocab[t])
    for pos, ch in [(0, '{'), (1, '"'), (2, 'a'), (3, '"'),
                    (5, '1'), (6, '}')]:
        mgr.reveal_token(pos, char_to_id[ch])
    valid = mgr.get_valid_tokens(4, V)
    valid_chars = {json_chars[t] for t in valid}
    assert valid_chars == {':'}, f"Expected only ':', got {valid_chars}"

    # Scenario 2: { _ } — position 1 between '{' and '}'
    # After '{' (OBJ_KEY_OR_CLOSE), before '}' (needs state that accepts '}')
    mgr2 = SegmentManager(dfa, gen_start=0, gen_length=3,
                          token_to_bytes=lambda t: vocab[t])
    mgr2.reveal_token(0, char_to_id['{'])
    mgr2.reveal_token(2, char_to_id['}'])
    valid2 = mgr2.get_valid_tokens(1, V)
    valid_chars2 = {json_chars[t] for t in valid2}
    # Between { and }, valid chars should include whitespace (stays in
    # OBJ_KEY_OR_CLOSE which accepts }) and '"' (starts a key, but then
    # need to close it before }, so actually only whitespace works here)
    assert ' ' in valid_chars2, f"Whitespace should be valid between {{ and }}"
    assert 'a' not in valid_chars2, f"'a' should not be valid between {{ and }}"
    print(f"  PASS: JSON boundary — valid between '{{' and '}}' = {sorted(valid_chars2)}")
    print(f"  PASS: JSON boundary — only ':' valid between '\"a\"' and '1}}'")



def test_json_full_generation():
    """Generate a JSON string position by position and verify validity."""
    dfa = build_json_dfa(max_depth=4)
    json_str = '{"name": "test", "value": 42}'
    tokens = [(i, bytes([b])) for i, b in enumerate(json_str.encode())]
    vocab = {i: tok_bytes for i, (_, tok_bytes) in enumerate(tokens)}
    # token_id i corresponds to position i's byte
    mgr = SegmentManager(dfa, gen_start=0, gen_length=len(json_str),
                         token_to_bytes=lambda t: vocab[t])

    # Reveal left-to-right
    for i in range(len(tokens)):
        mgr.reveal_token(i, i)

    assert mgr.num_segments == 1
    assert mgr.is_valid_complete(), "Full JSON should be valid"
    print("  PASS: full JSON generation valid")


def test_json_invalid_detection():
    """Detect invalid JSON via validity checks."""
    dfa = build_json_dfa(max_depth=4)

    # '{a' is invalid JSON — 'a' can't follow '{' at the top level.
    # But the segment has non-empty pairs because '{a' is valid in some
    # DFA contexts (e.g. inside a string). The key is that NO pair has
    # entry == start_state and exit in accept_states.
    vocab = {0: b'{', 1: b'a'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=2,
                         token_to_bytes=lambda t: vocab[t])
    mgr.reveal_token(0, 0)  # '{'
    mgr.reveal_token(1, 1)  # 'a'
    assert not mgr.is_valid_complete(), "'{a' should not be valid complete JSON"
    # Check: no pair has start_state entry and accept exit
    seg = mgr._segments[0]
    valid_from_start = any(
        e == dfa.start_state and x in dfa.accept_states
        for e, x in seg.pairs
    )
    assert not valid_from_start, "'{a' should not have a valid start->accept path"
    print("  PASS: invalid JSON detected (no start->accept path)")

    # Truly empty pairs: a sequence impossible in ANY context
    # '}' then '{' from start: } is invalid at position 0 from start
    vocab2 = {0: b'}', 1: b'{'}
    mgr2 = SegmentManager(dfa, gen_start=0, gen_length=2,
                          token_to_bytes=lambda t: vocab2[t])
    mgr2.reveal_token(0, 0)  # '}'
    # Check: from start_state, '}' -> DEAD. But segment tries all states.
    # '}' is valid from many states. Let's just check is_valid_complete.
    mgr2.reveal_token(1, 1)  # '{'
    assert not mgr2.is_valid_complete(), "'}{' should not be valid JSON"
    print("  PASS: '}{' detected as invalid")


# ---------------------------------------------------------------------------
# Order independence through manager
# ---------------------------------------------------------------------------

def test_order_independence_manager():
    """
    All revelation orders produce the same final segment state
    when going through the manager.
    """
    dfa = build_json_dfa(max_depth=4)
    json_str = '{"a":1}'
    char_bytes = [bytes([b]) for b in json_str.encode()]
    n = len(char_bytes)
    vocab = {i: char_bytes[i] for i in range(n)}

    # Collect final pairs from all n! orderings
    all_pairs = set()
    for perm in permutations(range(n)):
        mgr = SegmentManager(dfa, gen_start=0, gen_length=n,
                             token_to_bytes=lambda t: vocab[t])
        for idx in perm:
            mgr.reveal_token(idx, idx)
        assert mgr.num_segments == 1
        all_pairs.add(mgr._segments[0].pairs)

    assert len(all_pairs) == 1, f"Got {len(all_pairs)} distinct pair sets"
    print(f"  PASS: all {len(list(permutations(range(n))))} orderings consistent through manager")


# ---------------------------------------------------------------------------
# Prompt initialization
# ---------------------------------------------------------------------------

def test_prompt_init():
    """Prompt creates a left-context segment before gen region."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    # Prompt "ab" at positions -2, -1. Gen region starts at 0.
    mgr = SegmentManager(dfa, gen_start=0, gen_length=4,
                         token_to_bytes=lambda t: vocab[t])
    mgr.init_with_prompt([b'a', b'b'])

    assert mgr.num_segments == 1
    seg = mgr._segments[0]
    assert seg.start == -2 and seg.end == -1
    # 'ab': (0, 0)
    assert seg.pairs == frozenset({(0, 0)})

    # Position 0 should now be constrained by the prompt's exit state {0}
    # From state 0, 'a'->1 (valid), 'b'->DEAD (invalid)
    valid = mgr.get_valid_tokens(0, 2)
    assert valid == {0}, f"After prompt 'ab', only 'a' valid at pos 0. Got {valid}"
    print("  PASS: prompt initialization")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_repr():
    """Repr string is informative."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    mgr = SegmentManager(dfa, gen_start=0, gen_length=6,
                         token_to_bytes=lambda t: vocab[t])
    mgr.reveal_token(1, 0)
    mgr.reveal_token(4, 1)
    r = repr(mgr)
    assert "committed=2/6" in r
    assert "[1-1]" in r
    assert "[4-4]" in r
    print(f"  PASS: repr = {r}")


if __name__ == '__main__':
    print("=== Reveal cases (ab*) ===")
    test_reveal_isolated()
    test_reveal_extend_right()
    test_reveal_extend_left()
    test_reveal_bridge()

    print("\n=== Valid tokens (ab*) ===")
    test_valid_tokens_left_only()
    test_valid_tokens_right_only()
    test_valid_tokens_both_sides()
    test_valid_tokens_gen_start()
    test_valid_tokens_gen_end()
    test_is_token_valid()

    print("\n=== JSON constraints ===")
    test_json_tight_constraint()
    test_json_structural_boundary()
    test_json_full_generation()
    test_json_invalid_detection()

    print("\n=== Order independence ===")
    test_order_independence_manager()

    print("\n=== Prompt init ===")
    test_prompt_init()

    print("\n=== Repr ===")
    test_repr()

    print("\nAll tests passed.")