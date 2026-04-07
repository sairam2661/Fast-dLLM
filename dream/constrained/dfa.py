"""
DFA construction and operations for constrained decoding.

Operates at the byte level (like LLGuidance). Tokens are multi-byte sequences;
the token trie bridges the gap by processing one byte at a time.

Usage:
    dfa = build_json_dfa(max_depth=6)
    state = dfa.start_state
    for byte in b'{"key": "value"}':
        state = dfa.transition(state, byte)
    assert dfa.is_accept(state)
"""

from __future__ import annotations
from dataclasses import dataclass, field


# Sentinel for invalid/dead transitions
DEAD = -1


@dataclass
class DFA:
    """
    Deterministic Finite Automaton with precomputed forward and inverse
    transition tables.

    States are integers 0..num_states-1.
    Transitions are indexed by byte (0..255).
    """

    num_states: int
    start_state: int
    accept_states: frozenset[int]

    # forward[state][byte] -> next_state or DEAD
    forward: list[list[int]] = field(repr=False)

    # inverse[state][byte] -> frozenset of predecessor states
    inverse: list[list[frozenset[int]]] = field(repr=False)

    def transition(self, state: int, byte_val: int) -> int:
        """Return next state, or DEAD if transition is invalid."""
        return self.forward[state][byte_val]

    def transition_seq(self, state: int, byte_seq: bytes) -> int:
        """Apply a sequence of byte transitions. Returns DEAD if any step fails."""
        for b in byte_seq:
            state = self.forward[state][b]
            if state == DEAD:
                return DEAD
        return state

    def predecessors(self, state: int, byte_val: int) -> frozenset[int]:
        """Return set of states that transition to `state` via `byte_val`."""
        return self.inverse[state][byte_val]

    def is_accept(self, state: int) -> bool:
        return state in self.accept_states

    @staticmethod
    def from_transitions(
        num_states: int,
        start: int,
        accept: set[int],
        transitions: dict[tuple[int, int], int],
    ) -> "DFA":
        """
        Build a DFA from an explicit transition dict.
        transitions: {(state, byte) -> next_state}
        Missing entries are DEAD.
        """
        forward = [[DEAD] * 256 for _ in range(num_states)]
        for (s, b), ns in transitions.items():
            forward[s][b] = ns

        # Build inverse table
        inverse: list[list[set[int]]] = [[set() for _ in range(256)] for _ in range(num_states)]
        for (s, b), ns in transitions.items():
            if ns != DEAD:
                inverse[ns][b].add(s)

        # Freeze the inverse sets
        inverse_frozen = [
            [frozenset(inverse[s][b]) for b in range(256)]
            for s in range(num_states)
        ]

        return DFA(
            num_states=num_states,
            start_state=start,
            accept_states=frozenset(accept),
            forward=forward,
            inverse=inverse_frozen,
        )


# ---------------------------------------------------------------------------
# Stack-based JSON DFA
# ---------------------------------------------------------------------------
#
# State = (stack, local_context) where stack is a tuple of 'o'/'a' entries
# encoding the nesting of objects and arrays up to max_depth.
#
# We enumerate all reachable states via BFS from the start state, building
# the transition dict as we go.
#
# This captures:
#   - balanced braces and brackets up to max_depth
#   - string keys vs string values (separate states so closing quote
#     returns to the correct context)
#   - basic escape sequences in strings
#   - simplified number and literal syntax
#
# The DFA is intentionally simplified for Phase 1. Full lexer-level
# correctness (Unicode escapes, leading-zero rules, etc.) will be
# handled in Phase 2 with the Earley parser.


class _Lctx:
    """Local context at current nesting level."""
    WANT_VALUE = 0       # expecting a JSON value
    IN_STRING_KEY = 1    # inside a string that is an object key
    IN_STRING_VALUE = 2  # inside a string that is a JSON value
    IN_STRING_ESC_K = 3  # escape inside key string
    IN_STRING_ESC_V = 4  # escape inside value string
    IN_NUMBER = 5        # reading a number
    IN_LITERAL = 6       # reading true/false/null
    OBJ_COLON = 7        # expecting colon after key
    OBJ_NEXT = 8         # expecting , or } after object value
    ARR_NEXT = 9         # expecting , or ] after array value
    OBJ_KEY_OR_CLOSE = 10  # expecting key or } (after { or ,)
    ARR_VAL_OR_CLOSE = 11  # expecting value or ] (after [ or ,)
    DONE = 12            # top-level value complete


def _after_value(stack: tuple) -> int:
    """Determine what local context follows completing a value."""
    if len(stack) == 0:
        return _Lctx.DONE
    elif stack[-1] == 'o':
        return _Lctx.OBJ_NEXT
    elif stack[-1] == 'a':
        return _Lctx.ARR_NEXT
    else:
        raise ValueError(f"Unknown stack entry: {stack[-1]}")


def build_json_dfa(max_depth: int = 6) -> DFA:
    """
    Build a JSON DFA with explicit stack encoding.

    State = (stack_tuple, local_ctx) where stack is a tuple of 'o'/'a'
    entries of length 0..max_depth. States are enumerated by BFS from start.

    Args:
        max_depth: maximum nesting depth (default 6). Deeper nesting is
                   rejected. Increasing this increases state count
                   combinatorially.

    Returns:
        DFA instance with byte-level transitions.
    """
    state_map: dict[tuple[tuple, int], int] = {}
    states_list: list[tuple[tuple, int]] = []
    trans_dict: dict[tuple[int, int], int] = {}

    def get_or_create(stack: tuple, lctx: int) -> int:
        key = (stack, lctx)
        if key not in state_map:
            sid = len(states_list)
            state_map[key] = sid
            states_list.append(key)
        return state_map[key]

    start = get_or_create((), _Lctx.WANT_VALUE)

    WS_BYTES = [ord(c) for c in " \t\n\r"]
    DIGIT_BYTES = list(range(ord('0'), ord('9') + 1))
    LOWER_BYTES = list(range(ord('a'), ord('z') + 1))
    ESCAPE_CHARS = [ord(c) for c in '"\\bfnrt/u']
    # Printable ASCII minus " and \ for string content
    STRING_CONTENT_BYTES = [b for b in range(32, 127)
                            if b != ord('"') and b != ord('\\')]
    STRING_CONTENT_BYTES += list(range(128, 256))  # UTF-8 continuation
    # Number continuation bytes (simplified)
    NUMBER_BYTES = DIGIT_BYTES + [ord(c) for c in '.eE+-']

    queue = [start]
    visited = {start}

    while queue:
        sid = queue.pop(0)
        stack, lctx = states_list[sid]
        depth = len(stack)

        def add_t(byte_val: int, next_stack: tuple, next_lctx: int):
            nsid = get_or_create(next_stack, next_lctx)
            trans_dict[(sid, byte_val)] = nsid
            if nsid not in visited:
                visited.add(nsid)
                queue.append(nsid)

        if lctx == _Lctx.WANT_VALUE:
            for b in WS_BYTES:
                add_t(b, stack, _Lctx.WANT_VALUE)
            add_t(ord('"'), stack, _Lctx.IN_STRING_VALUE)
            if depth < max_depth:
                add_t(ord('{'), stack + ('o',), _Lctx.OBJ_KEY_OR_CLOSE)
                add_t(ord('['), stack + ('a',), _Lctx.ARR_VAL_OR_CLOSE)
            add_t(ord('-'), stack, _Lctx.IN_NUMBER)
            for b in DIGIT_BYTES:
                add_t(b, stack, _Lctx.IN_NUMBER)
            for c in 'tfn':
                add_t(ord(c), stack, _Lctx.IN_LITERAL)

        elif lctx == _Lctx.OBJ_KEY_OR_CLOSE:
            for b in WS_BYTES:
                add_t(b, stack, _Lctx.OBJ_KEY_OR_CLOSE)
            add_t(ord('"'), stack, _Lctx.IN_STRING_KEY)
            if depth > 0:
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord('}'), parent_stack, parent_ctx)

        elif lctx == _Lctx.IN_STRING_KEY:
            for b in STRING_CONTENT_BYTES:
                add_t(b, stack, _Lctx.IN_STRING_KEY)
            add_t(ord('\\'), stack, _Lctx.IN_STRING_ESC_K)
            add_t(ord('"'), stack, _Lctx.OBJ_COLON)

        elif lctx == _Lctx.IN_STRING_VALUE:
            for b in STRING_CONTENT_BYTES:
                add_t(b, stack, _Lctx.IN_STRING_VALUE)
            add_t(ord('\\'), stack, _Lctx.IN_STRING_ESC_V)
            after = _after_value(stack)
            add_t(ord('"'), stack, after)

        elif lctx == _Lctx.IN_STRING_ESC_K:
            for b in ESCAPE_CHARS:
                add_t(b, stack, _Lctx.IN_STRING_KEY)

        elif lctx == _Lctx.IN_STRING_ESC_V:
            for b in ESCAPE_CHARS:
                add_t(b, stack, _Lctx.IN_STRING_VALUE)

        elif lctx == _Lctx.OBJ_COLON:
            for b in WS_BYTES:
                add_t(b, stack, _Lctx.OBJ_COLON)
            add_t(ord(':'), stack, _Lctx.WANT_VALUE)

        elif lctx == _Lctx.OBJ_NEXT:
            for b in WS_BYTES:
                add_t(b, stack, _Lctx.OBJ_NEXT)
            add_t(ord(','), stack, _Lctx.OBJ_KEY_OR_CLOSE)
            if depth > 0:
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord('}'), parent_stack, parent_ctx)

        elif lctx == _Lctx.ARR_VAL_OR_CLOSE:
            for b in WS_BYTES:
                add_t(b, stack, _Lctx.ARR_VAL_OR_CLOSE)
            add_t(ord('"'), stack, _Lctx.IN_STRING_VALUE)
            if depth < max_depth:
                add_t(ord('{'), stack + ('o',), _Lctx.OBJ_KEY_OR_CLOSE)
                add_t(ord('['), stack + ('a',), _Lctx.ARR_VAL_OR_CLOSE)
            add_t(ord('-'), stack, _Lctx.IN_NUMBER)
            for b in DIGIT_BYTES:
                add_t(b, stack, _Lctx.IN_NUMBER)
            for c in 'tfn':
                add_t(ord(c), stack, _Lctx.IN_LITERAL)
            if depth > 0:
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord(']'), parent_stack, parent_ctx)

        elif lctx == _Lctx.ARR_NEXT:
            for b in WS_BYTES:
                add_t(b, stack, _Lctx.ARR_NEXT)
            add_t(ord(','), stack, _Lctx.ARR_VAL_OR_CLOSE)
            if depth > 0:
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord(']'), parent_stack, parent_ctx)

        elif lctx == _Lctx.IN_NUMBER:
            for b in NUMBER_BYTES:
                add_t(b, stack, _Lctx.IN_NUMBER)
            after = _after_value(stack)
            for b in WS_BYTES:
                add_t(b, stack, after)
            if depth > 0 and stack[-1] == 'o':
                add_t(ord(','), stack, _Lctx.OBJ_KEY_OR_CLOSE)
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord('}'), parent_stack, parent_ctx)
            elif depth > 0 and stack[-1] == 'a':
                add_t(ord(','), stack, _Lctx.ARR_VAL_OR_CLOSE)
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord(']'), parent_stack, parent_ctx)

        elif lctx == _Lctx.IN_LITERAL:
            for b in LOWER_BYTES:
                add_t(b, stack, _Lctx.IN_LITERAL)
            after = _after_value(stack)
            for b in WS_BYTES:
                add_t(b, stack, after)
            if depth > 0 and stack[-1] == 'o':
                add_t(ord(','), stack, _Lctx.OBJ_KEY_OR_CLOSE)
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord('}'), parent_stack, parent_ctx)
            elif depth > 0 and stack[-1] == 'a':
                add_t(ord(','), stack, _Lctx.ARR_VAL_OR_CLOSE)
                parent_stack = stack[:-1]
                parent_ctx = _after_value(parent_stack)
                add_t(ord(']'), parent_stack, parent_ctx)

        elif lctx == _Lctx.DONE:
            for b in WS_BYTES:
                add_t(b, stack, _Lctx.DONE)

    # Accept states: DONE, IN_NUMBER, and IN_LITERAL at depth 0
    # (numbers and literals are valid top-level values that can end without
    # a trailing delimiter)
    accept = set()
    accept_lctxs = {_Lctx.DONE, _Lctx.IN_NUMBER, _Lctx.IN_LITERAL}
    for (stack, lctx), sid in state_map.items():
        if len(stack) == 0 and lctx in accept_lctxs:
            accept.add(sid)

    return DFA.from_transitions(
        num_states=len(states_list),
        start=start,
        accept=accept,
        transitions=trans_dict,
    )


def validate_bytes(dfa: DFA, data: bytes) -> bool:
    """Check if a byte sequence is accepted by the DFA."""
    state = dfa.start_state
    for b in data:
        state = dfa.transition(state, b)
        if state == DEAD:
            return False
    return dfa.is_accept(state)