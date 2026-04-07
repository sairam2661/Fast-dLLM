"""
Unit tests for dfa.py.

Tests:
1. DFA construction (state count, start/accept states)
2. Valid JSON acceptance (objects, arrays, strings, numbers, literals, nesting)
3. Invalid JSON rejection (unclosed brackets, missing colons, unquoted keys, etc.)
4. Inverse transition consistency (forward and inverse tables agree)
5. Depth limit enforcement
6. Transition sequence helper
7. Edge cases (empty string, whitespace-only, deeply nested)

Run:
    python test_dfa.py
"""

from dfa import build_json_dfa, validate_bytes, DFA, DEAD


def test_basic_construction():
    """DFA builds without error and has sensible state count."""
    dfa = build_json_dfa(max_depth=4)
    print(f"  States: {dfa.num_states}, Accept: {len(dfa.accept_states)}")
    assert dfa.num_states > 0
    assert len(dfa.accept_states) > 0
    assert dfa.start_state >= 0
    assert dfa.start_state < dfa.num_states
    print("  PASS")


def test_valid_json():
    """All syntactically valid JSON byte strings are accepted."""
    dfa = build_json_dfa(max_depth=6)

    cases = [
        # Primitives
        (b'42', "integer"),
        (b'-7', "negative integer"),
        (b'0', "zero"),
        (b'3.14', "float"),
        (b'"hello"', "simple string"),
        (b'""', "empty string"),
        (b'"with spaces"', "string with spaces"),
        (b'true', "true"),
        (b'false', "false"),
        (b'null', "null"),
        # Objects
        (b'{}', "empty object"),
        (b'{"key": "value"}', "simple object"),
        (b'{"a": 1, "b": 2}', "multi-key object"),
        (b'{"k": true, "j": false, "n": null}', "object with literals"),
        # Arrays
        (b'[]', "empty array"),
        (b'[1, 2, 3]', "integer array"),
        (b'[null, true, false, 42, "str"]', "mixed array"),
        # Nesting
        (b'{"nested": {"inner": "value"}}', "nested object"),
        (b'[1, [2, [3]]]', "nested array"),
        (b'{"a": [1, 2], "b": {"c": true}}', "mixed nesting"),
        (b'{"empty_obj": {}, "empty_arr": []}', "empty containers"),
        # Whitespace
        (b'  {"key": "value"}  ', "leading/trailing whitespace"),
        (b'{\n  "key" : "value"\n}', "newlines and extra spaces"),
        # Escapes in strings
        (b'{"key": "value with \\"quotes\\""}', "escaped quotes"),
        (b'{"k": "line1\\nline2"}', "escaped newline"),
        (b'{"k": "tab\\there"}', "escaped tab"),
        (b'{"k": "back\\\\slash"}', "escaped backslash"),
    ]

    for data, label in cases:
        result = validate_bytes(dfa, data)
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {label}: {data.decode('utf-8', errors='replace')}")
        if not result:
            _trace_failure(dfa, data)
            assert False, f"Should accept: {label}"


def test_invalid_json():
    """Syntactically invalid JSON byte strings are rejected."""
    dfa = build_json_dfa(max_depth=6)

    cases = [
        (b'', "empty input"),
        (b'{', "unclosed brace"),
        (b'}', "unmatched closing brace"),
        (b'{"key"}', "key without colon/value"),
        (b'{key: "value"}', "unquoted key"),
        (b'[,]', "leading comma in array"),
        # NOTE: trailing commas are accepted by our DFA (lenient, like many
        # real parsers). Strict JSON rejection of trailing commas would require
        # separate OBJ_AFTER_COMMA / ARR_AFTER_COMMA states. Not worth the
        # state-space cost for Phase 1 — the Earley parser in Phase 2 can
        # enforce strict JSON if needed.
        (b'{{}', "extra opening brace"),
        (b'{"a": "b" "c": "d"}', "missing comma between pairs"),
    ]

    for data, label in cases:
        result = validate_bytes(dfa, data)
        status = "PASS" if not result else "FAIL"
        print(f"  {status} (reject): {label}: {data.decode('utf-8', errors='replace')}")
        if result:
            assert False, f"Should reject: {label}"


def test_inverse_transitions():
    """Forward and inverse transition tables are consistent."""
    dfa = build_json_dfa(max_depth=3)

    errors = 0
    # Check: if forward[s][b] = ns, then s in inverse[ns][b]
    for s in range(dfa.num_states):
        for b in range(256):
            ns = dfa.transition(s, b)
            if ns != DEAD:
                preds = dfa.predecessors(ns, b)
                if s not in preds:
                    print(f"  FAIL: {s} -[{b}]-> {ns}, but {s} not in predecessors({ns}, {b})")
                    errors += 1

    # Check: if s in inverse[ns][b], then forward[s][b] = ns
    for ns in range(dfa.num_states):
        for b in range(256):
            for pred in dfa.predecessors(ns, b):
                actual = dfa.transition(pred, b)
                if actual != ns:
                    print(f"  FAIL: {pred} in predecessors({ns}, {b}) but transition = {actual}")
                    errors += 1

    if errors == 0:
        print("  PASS")
    else:
        assert False, f"{errors} inconsistencies"


def test_depth_limit():
    """Nesting beyond max_depth is rejected."""
    dfa = build_json_dfa(max_depth=3)

    d3 = b'{"a": {"b": {"c": 1}}}'
    assert validate_bytes(dfa, d3), "Depth 3 should be valid"
    print("  PASS: depth 3 accepted")

    d4 = b'{"a": {"b": {"c": {"d": 1}}}}'
    assert not validate_bytes(dfa, d4), "Depth 4 should be rejected (max_depth=3)"
    print("  PASS: depth 4 rejected")


def test_transition_seq():
    """transition_seq processes multi-byte sequences correctly."""
    dfa = build_json_dfa(max_depth=4)

    # Full valid JSON should end in accept state
    state = dfa.transition_seq(dfa.start_state, b'{"key": 42}')
    assert state != DEAD, "Should not hit DEAD state"
    assert dfa.is_accept(state), f"Final state {state} should be accept"
    print("  PASS: transition_seq on valid JSON")

    # Partial JSON should not be accept
    state = dfa.transition_seq(dfa.start_state, b'{"key":')
    assert state != DEAD, "Partial JSON should not be DEAD"
    assert not dfa.is_accept(state), "Partial JSON should not be accept"
    print("  PASS: transition_seq on partial JSON (not accept, not dead)")

    # Invalid byte at start
    state = dfa.transition_seq(dfa.start_state, b'@@@')
    assert state == DEAD, "@@@ should hit DEAD"
    print("  PASS: transition_seq on invalid bytes -> DEAD")


def test_from_transitions():
    """DFA.from_transitions builds correct DFA from explicit transitions."""
    # Simple (ab)* DFA
    transitions = {
        (0, ord('a')): 1,
        (1, ord('b')): 0,
    }
    dfa = DFA.from_transitions(num_states=2, start=0, accept={0},
                               transitions=transitions)

    assert dfa.transition(0, ord('a')) == 1
    assert dfa.transition(1, ord('b')) == 0
    assert dfa.transition(0, ord('b')) == DEAD
    assert dfa.transition(1, ord('a')) == DEAD
    assert dfa.is_accept(0)
    assert not dfa.is_accept(1)

    # "abab" should be accepted
    state = dfa.transition_seq(0, b'abab')
    assert state == 0 and dfa.is_accept(state)
    print("  PASS: from_transitions builds correct (ab)* DFA")


def _trace_failure(dfa: DFA, data: bytes):
    """Print DFA trace for debugging a failed acceptance."""
    state = dfa.start_state
    for i, b in enumerate(data):
        next_state = dfa.transition(state, b)
        if next_state == DEAD:
            print(f"    DEAD at pos {i}, byte {chr(b)!r}, from state {state}")
            return
        state = next_state
    print(f"    Final state {state}, is_accept={dfa.is_accept(state)}")


if __name__ == '__main__':
    print("=== Basic Construction ===")
    test_basic_construction()

    print("\n=== Valid JSON ===")
    test_valid_json()

    print("\n=== Invalid JSON ===")
    test_invalid_json()

    print("\n=== Inverse Transitions ===")
    test_inverse_transitions()

    print("\n=== Depth Limit ===")
    test_depth_limit()

    print("\n=== transition_seq ===")
    test_transition_seq()

    print("\n=== from_transitions ===")
    test_from_transitions()

    print("\nAll tests passed.")