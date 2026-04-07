"""
Unit tests for trie.py.

Tests:
1. Trie construction and stats
2. Correctness against brute-force (all tokens, small vocab)
3. Correctness against brute-force (JSON DFA, larger vocab)
4. Pruning actually reduces work (fewer nodes visited)
5. Right-entry filtering
6. Edge cases (empty vocab, single-byte tokens, multi-byte tokens)
7. Agreement with SegmentManager.get_valid_tokens

Run:
    python test_trie.py
"""

from dfa import build_json_dfa, DFA, DEAD
from segments import Segment, create, extend_right
from manager import SegmentManager
from trie import TokenTrie


# ---------------------------------------------------------------------------
# Helper DFAs
# ---------------------------------------------------------------------------

def build_ab_star_dfa() -> DFA:
    transitions = {(0, ord('a')): 1, (1, ord('b')): 0}
    return DFA.from_transitions(num_states=2, start=0, accept={0},
                                transitions=transitions)


# ---------------------------------------------------------------------------
# Brute-force reference implementation
# ---------------------------------------------------------------------------

def brute_force_valid(
    left_exits: frozenset[int],
    right_entries: frozenset[int] | None,
    dfa: DFA,
    token_to_bytes: dict[int, bytes],
) -> set[int]:
    """Compute valid tokens by checking every token in the vocab."""
    valid = set()
    for tid, tbytes in token_to_bytes.items():
        if len(tbytes) == 0:
            if right_entries is None or (left_exits & right_entries):
                valid.add(tid)
            continue
        for q in left_exits:
            result = dfa.transition_seq(q, tbytes)
            if result != DEAD:
                if right_entries is None or result in right_entries:
                    valid.add(tid)
                    break
    return valid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_construction():
    """Trie builds and reports sensible stats."""
    vocab = {0: b'a', 1: b'b', 2: b'ab', 3: b'ba', 4: b'abc'}
    trie = TokenTrie(vocab)
    s = trie.stats()
    print(f"  Stats: {s}")
    assert s['num_tokens'] == 5
    assert s['vocab_size'] == 5
    assert s['num_nodes'] > 5  # internal nodes too
    assert s['max_depth'] >= 3  # 'abc' has depth 3
    print("  PASS")


def test_single_byte_vocab():
    """Trie with single-byte tokens matches brute force."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    trie = TokenTrie(vocab)

    # From state 0 (start), no right constraint
    valid = trie.compute_valid_set(frozenset({0}), None, dfa)
    expected = brute_force_valid(frozenset({0}), None, dfa, vocab)
    assert valid == expected == {0}, f"Got {valid}, expected {expected}"
    print("  PASS: from state 0, no right constraint")

    # From state 1, no right constraint
    valid = trie.compute_valid_set(frozenset({1}), None, dfa)
    expected = brute_force_valid(frozenset({1}), None, dfa, vocab)
    assert valid == expected == {1}, f"Got {valid}, expected {expected}"
    print("  PASS: from state 1, no right constraint")

    # From state 0, right must be state 0 (accept)
    valid = trie.compute_valid_set(frozenset({0}), frozenset({0}), dfa)
    expected = brute_force_valid(frozenset({0}), frozenset({0}), dfa, vocab)
    assert valid == expected
    print(f"  PASS: from state 0, right={{0}}: {valid}")

    # From state 1, right must be state 0
    valid = trie.compute_valid_set(frozenset({1}), frozenset({0}), dfa)
    expected = brute_force_valid(frozenset({1}), frozenset({0}), dfa, vocab)
    assert valid == expected == {1}  # 'b': 1->0
    print("  PASS: from state 1, right={0}")


def test_multibyte_vocab():
    """Trie with multi-byte tokens matches brute force."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b', 2: b'ab', 3: b'ba', 4: b'abab', 5: b'aa'}
    trie = TokenTrie(vocab)

    # From state 0, no right constraint
    valid = trie.compute_valid_set(frozenset({0}), None, dfa)
    expected = brute_force_valid(frozenset({0}), None, dfa, vocab)
    assert valid == expected
    # 'a': 0->1 ok, 'ab': 0->0 ok, 'abab': 0->0 ok, 'aa': 0->DEAD no
    # 'b': 0->DEAD no, 'ba': 0->DEAD no
    assert valid == {0, 2, 4}, f"Got {valid}"
    print(f"  PASS: multi-byte from state 0: {valid}")

    # From state 1, right must be 0 (accept)
    valid = trie.compute_valid_set(frozenset({1}), frozenset({0}), dfa)
    expected = brute_force_valid(frozenset({1}), frozenset({0}), dfa, vocab)
    assert valid == expected
    # 'b': 1->0 ok, 'ba': 1->1 no (1 not in {0}), 'bab'? not in vocab
    assert valid == {1}, f"Got {valid}"
    print(f"  PASS: multi-byte from state 1, right={{0}}: {valid}")


def test_multiple_left_exits():
    """Trie handles multiple left exit states correctly."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    trie = TokenTrie(vocab)

    # Both states as exits, no right constraint
    valid = trie.compute_valid_set(frozenset({0, 1}), None, dfa)
    expected = brute_force_valid(frozenset({0, 1}), None, dfa, vocab)
    assert valid == expected == {0, 1}
    print("  PASS: multiple left exits")


def test_empty_result():
    """No valid tokens when constraints are incompatible."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    trie = TokenTrie(vocab)

    # From state 0, right must be state 1: need transition(0,t)=1, that's 'a'.
    # But then also from state 1, right must be state 1: need transition(1,t)=1,
    # 'b': 1->0, 'a': DEAD. So nothing.
    # Actually let's use: from state 0, right must be empty set.
    valid = trie.compute_valid_set(frozenset({0}), frozenset(), dfa)
    assert valid == set()
    print("  PASS: empty right_entries -> no valid tokens")


def test_json_brute_force_agreement():
    """
    Trie matches brute force on JSON DFA with a realistic char-level vocab.
    Tests multiple constraint configurations.
    """
    dfa = build_json_dfa(max_depth=4)

    # Char-level vocab covering JSON-relevant bytes
    chars = '{}[]:,"0123456789abcdefghijklmnopqrstuvwxyz \t\n\r-.+eE\\/bfnrtu'
    # Deduplicate
    seen = set()
    unique_chars = []
    for c in chars:
        if c not in seen:
            seen.add(c)
            unique_chars.append(c)

    vocab = {i: c.encode() for i, c in enumerate(unique_chars)}
    trie = TokenTrie(vocab)
    print(f"  Vocab size: {len(vocab)}, Trie nodes: {trie.num_nodes}")

    # Test several constraint configurations
    configs = [
        ("start, no right", frozenset({dfa.start_state}), None),
        ("start, accept right", frozenset({dfa.start_state}), dfa.accept_states),
        ("all states, no right", frozenset(range(dfa.num_states)), None),
        ("all states, accept right", frozenset(range(dfa.num_states)), dfa.accept_states),
    ]

    # Also get exit states from a real segment
    seg = create(0, b'{', dfa)
    seg = extend_right(seg, 1, b'"', dfa)
    seg = extend_right(seg, 2, b'k', dfa)
    seg = extend_right(seg, 3, b'"', dfa)
    configs.append((
        "after '{\"k\"', no right",
        seg.exit_states(),
        None,
    ))

    seg2 = create(0, b'}', dfa)
    configs.append((
        "after '{\"k\"', right='}'",
        seg.exit_states(),
        seg2.entry_states(),
    ))

    for label, left_exits, right_entries in configs:
        trie_result = trie.compute_valid_set(left_exits, right_entries, dfa)
        bf_result = brute_force_valid(left_exits, right_entries, dfa, vocab)
        assert trie_result == bf_result, (
            f"Mismatch for {label}:\n"
            f"  trie:  {trie_result}\n"
            f"  brute: {bf_result}\n"
            f"  diff:  trie-bf={trie_result - bf_result}, bf-trie={bf_result - trie_result}"
        )
        valid_chars = sorted(unique_chars[t] for t in trie_result)
        print(f"  PASS: {label} -> {len(trie_result)} valid tokens")


def test_json_multibyte_tokens():
    """Trie works with multi-byte tokens on JSON DFA."""
    dfa = build_json_dfa(max_depth=4)
    vocab = {
        0: b'{',
        1: b'}',
        2: b'"',
        3: b':',
        4: b',',
        5: b'key',     # multi-byte
        6: b'value',   # multi-byte
        7: b'{"',      # multi-byte structural
        8: b'"}',      # multi-byte structural
        9: b': "',     # multi-byte with space
        10: b'true',
        11: b'null',
        12: b'42',
        13: b' ',
    }
    trie = TokenTrie(vocab)

    # From start, no right constraint
    valid = trie.compute_valid_set(frozenset({dfa.start_state}), None, dfa)
    bf = brute_force_valid(frozenset({dfa.start_state}), None, dfa, vocab)
    assert valid == bf, f"Mismatch: trie={valid}, bf={bf}"
    print(f"  PASS: multi-byte JSON tokens from start: {valid}")

    # From start, must reach accept
    valid_acc = trie.compute_valid_set(
        frozenset({dfa.start_state}), dfa.accept_states, dfa)
    bf_acc = brute_force_valid(
        frozenset({dfa.start_state}), dfa.accept_states, dfa, vocab)
    assert valid_acc == bf_acc
    print(f"  PASS: multi-byte JSON tokens start->accept: {valid_acc}")


def test_manager_agreement():
    """
    Trie-based valid set matches SegmentManager.get_valid_tokens.
    This is the integration test.
    """
    dfa = build_json_dfa(max_depth=4)
    chars = list('{}[]:,"0123456789abcdefghijklmnopqrstuvwxyz \n-.')
    char_to_id = {c: i for i, c in enumerate(chars)}
    vocab = {i: c.encode() for i, c in enumerate(chars)}
    V = len(chars)
    trie = TokenTrie(vocab)

    # Setup: {"a" _ 1}  — position 4 masked
    mgr = SegmentManager(dfa, gen_start=0, gen_length=7,
                         token_to_bytes=lambda t: vocab[t])
    for pos, ch in [(0, '{'), (1, '"'), (2, 'a'), (3, '"'),
                    (5, '1'), (6, '}')]:
        mgr.reveal_token(pos, char_to_id[ch])

    # Manager's brute-force result
    mgr_valid = mgr.get_valid_tokens(4, V)

    # Trie result using same left/right constraints
    left_exits = mgr._left_exit_states(4)
    right_entries = mgr._right_entry_states(4)
    trie_valid = trie.compute_valid_set(left_exits, right_entries, dfa)

    assert trie_valid == mgr_valid, (
        f"Mismatch: trie={trie_valid}, manager={mgr_valid}"
    )
    valid_chars = {chars[t] for t in trie_valid}
    assert valid_chars == {':'}, f"Expected only ':', got {valid_chars}"
    print(f"  PASS: trie matches manager for colon position: {valid_chars}")

    # Another position: { _ } (position 1 between { and })
    mgr2 = SegmentManager(dfa, gen_start=0, gen_length=3,
                          token_to_bytes=lambda t: vocab[t])
    mgr2.reveal_token(0, char_to_id['{'])
    mgr2.reveal_token(2, char_to_id['}'])

    mgr2_valid = mgr2.get_valid_tokens(1, V)
    left2 = mgr2._left_exit_states(1)
    right2 = mgr2._right_entry_states(1)
    trie2_valid = trie.compute_valid_set(left2, right2, dfa)
    assert trie2_valid == mgr2_valid, (
        f"Mismatch: trie={trie2_valid}, manager={mgr2_valid}"
    )
    print(f"  PASS: trie matches manager for {{_}}: {sorted(chars[t] for t in trie2_valid)}")


def test_bool_mask():
    """compute_valid_mask returns correct boolean list."""
    dfa = build_ab_star_dfa()
    vocab = {0: b'a', 1: b'b'}
    trie = TokenTrie(vocab)

    mask = trie.compute_valid_mask(frozenset({0}), None, dfa)
    assert mask == [True, False]  # 'a' valid, 'b' not from state 0
    print("  PASS: bool mask")


def test_empty_vocab():
    """Trie handles empty vocabulary."""
    trie = TokenTrie({})
    assert trie.num_nodes == 1  # just root
    dfa = build_ab_star_dfa()
    valid = trie.compute_valid_set(frozenset({0}), None, dfa)
    assert valid == set()
    print("  PASS: empty vocab")


if __name__ == '__main__':
    print("=== Construction ===")
    test_construction()

    print("\n=== Single-byte vocab (ab*) ===")
    test_single_byte_vocab()

    print("\n=== Multi-byte vocab (ab*) ===")
    test_multibyte_vocab()

    print("\n=== Multiple left exits ===")
    test_multiple_left_exits()

    print("\n=== Empty result ===")
    test_empty_result()

    print("\n=== JSON brute-force agreement ===")
    test_json_brute_force_agreement()

    print("\n=== JSON multi-byte tokens ===")
    test_json_multibyte_tokens()

    print("\n=== Manager agreement ===")
    test_manager_agreement()

    print("\n=== Bool mask ===")
    test_bool_mask()

    print("\n=== Empty vocab ===")
    test_empty_vocab()

    print("\nAll tests passed.")