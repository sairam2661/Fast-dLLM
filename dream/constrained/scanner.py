"""
JSON scanner for token-level grammar parsing.

Maps byte-level input to grammar terminal IDs. Handles the fact that BPE
tokens can span lexeme boundaries (e.g., a single BPE token `"name":` emits
T_KEY_name, T_COLON before moving to the next byte).

The scanner is a DFA whose states encode:
  - Where we are in the current lexeme (idle, in-string, in-number, etc.)
  - For strings: whether we're still matching a known key prefix or have
    fallen back to a generic string

Interface:
    scanner = JsonScanner(key_strings=["name", "age"])
    state = scanner.start_state
    for byte in token_bytes:
        state, terminals = scanner.step(state, byte)
        for t in terminals:
            parser_configs = automaton.transition_seq(parser_config, ...)

The `(scanner_state, parser_config)` pair is the complete state for trie
traversal. Scanner states are small integers. Parser configs are BoundedLRAutomaton
config IDs. The trie carries sets of these pairs.

Terminal ID assignment:
    0..255   : single-byte structural terminals ('{', '}', '[', ']', ':', ',')
    256      : T_STRING  — any complete JSON string not matching a key
    257      : T_NUMBER  — JSON number with decimal or exponent
    258      : T_INTEGER — JSON integer (no decimal, no exponent)
    259      : T_TRUE
    260      : T_FALSE
    261      : T_NULL
    262+     : T_KEY_base + key_index, one per schema key (in order given)

Constants T_STRING, T_NUMBER, etc. are exported for use in grammar construction.
"""

from __future__ import annotations
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Terminal ID constants (must match what schema_compiler.py uses)
# ---------------------------------------------------------------------------

T_STRING  = 256
T_NUMBER  = 257
T_INTEGER = 258
T_TRUE    = 259
T_FALSE   = 260
T_NULL    = 261
T_KEY_BASE = 262  # T_KEY_BASE + i for the i-th key in the schema

DEAD = -1  # scanner dead state sentinel


# ---------------------------------------------------------------------------
# Scanner state kinds (internal)
# ---------------------------------------------------------------------------
# We assign integer IDs to scanner states by building them explicitly.
# State kinds (not IDs):
#
#   IDLE          — between lexemes, ready for next token
#   KEY_Q         — just saw opening '"', trying key trie
#   KS(key, pos)  — matched key[0:pos] bytes of key `key` (inside '"')
#   STR           — inside a string, key trie failed or key exhausted non-terminally
#   STR_ESC       — inside a string, after '\'
#   NUM_INT       — in a number, integer part only so far
#   NUM_FLOAT     — in a number, saw '.' or 'e'/'E' (will emit T_NUMBER)
#   NUM_EXP_SIGN  — in number exponent, after 'e'/'E', expecting sign or digit
#   WORD_TRUE_i   — matched i chars of "true"
#   WORD_FALSE_i  — matched i chars of "false"
#   WORD_NULL_i   — matched i chars of "null"

_IDLE = "IDLE"
_KEY_Q = "KEY_Q"
_STR = "STR"
_STR_ESC = "STR_ESC"
_NUM_INT = "NUM_INT"
_NUM_FLOAT = "NUM_FLOAT"
_NUM_EXP_SIGN = "NUM_EXP_SIGN"

def _word_state(word: str, pos: int) -> str:
    return f"WORD_{word}_{pos}"

def _key_state(key_idx: int, pos: int) -> str:
    return f"KS_{key_idx}_{pos}"


# ---------------------------------------------------------------------------
# Scanner DFA construction
# ---------------------------------------------------------------------------

class JsonScanner:
    """
    Byte-level scanner DFA for JSON, schema-aware.

    States are integers. step(state, byte) returns (new_state, terminals)
    where terminals is a list of 0, 1, or 2 terminal IDs emitted by this byte.

    Two terminals are emitted when a byte terminates a pending number or word
    AND is itself a structural terminal (e.g., ',' terminates an integer AND
    is itself T_COMMA).
    """

    def __init__(self, key_strings: list[str]) -> None:
        """
        Args:
            key_strings: list of JSON object key strings (without quotes) that
                         the grammar distinguishes. Order determines terminal IDs:
                         key_strings[i] maps to terminal T_KEY_BASE + i.
        """
        self.key_strings = list(key_strings)
        self.key_terminal = {key: T_KEY_BASE + i for i, key in enumerate(key_strings)}

        # State assignment
        self._state_ids: dict[str, int] = {}
        self._states: list[str] = []

        # Transitions: _trans[state][byte] -> (new_state, terminals_list)
        # terminals_list is a tuple of 0-2 terminal IDs
        self._trans: list[dict[int, tuple[int, tuple[int, ...]]]] = []

        self._build()

        self.num_states = len(self._states)
        self.start_state = self._state_ids[_IDLE]
        self.dead_state = DEAD

    # ------------------------------------------------------------------
    # State management

    def _sid(self, kind: str) -> int:
        """Get or create state ID for a kind string."""
        if kind not in self._state_ids:
            sid = len(self._states)
            self._state_ids[kind] = sid
            self._states.append(kind)
            self._trans.append({})
        return self._state_ids[kind]

    def _add(self, src_kind: str, byte_val: int, dst_kind: str, terminals: tuple[int, ...]) -> None:
        src = self._sid(src_kind)
        dst = self._sid(dst_kind) if dst_kind is not None else DEAD
        self._trans[src][byte_val] = (dst, terminals)

    # ------------------------------------------------------------------
    # Build

    def _build(self) -> None:
        # Ensure all states exist before adding transitions
        self._sid(_IDLE)
        self._sid(_KEY_Q)
        self._sid(_STR)
        self._sid(_STR_ESC)
        self._sid(_NUM_INT)
        self._sid(_NUM_FLOAT)
        self._sid(_NUM_EXP_SIGN)
        for word in ["true", "false", "null"]:
            for pos in range(1, len(word)):
                self._sid(_word_state(word, pos))
        for i, key in enumerate(self.key_strings):
            for pos in range(len(key) + 1):  # pos=len means key fully matched, waiting for '"'
                self._sid(_key_state(i, pos))

        self._build_idle()
        self._build_key_q()   # builds key trie states inline
        self._build_str()
        self._build_str_esc()
        self._build_num_int()
        self._build_num_float()
        self._build_num_exp_sign()
        self._build_words()

    def _structural_transition(self, src_kind: str, pending_terminal: int | None = None) -> None:
        """
        Add transitions from src_kind for all structural bytes.
        If pending_terminal is set, emit it before the structural terminal
        (two-terminal case: number or word terminated by structural char).
        """
        structural = {
            ord('{'): (ord('{'),),
            ord('}'): (ord('}'),),
            ord('['): (ord('['),),
            ord(']'): (ord(']'),),
            ord(':'): (ord(':'),),
            ord(','): (ord(','),),
        }
        ws_bytes = [ord(' '), ord('\t'), ord('\n'), ord('\r')]

        for byte_val, (terminal,) in structural.items():
            if pending_terminal is not None:
                terms = (pending_terminal, terminal)
            else:
                terms = (terminal,)
            self._add(src_kind, byte_val, _IDLE, terms)

        # Whitespace: emits pending (if any), stays IDLE
        for byte_val in ws_bytes:
            if pending_terminal is not None:
                self._add(src_kind, byte_val, _IDLE, (pending_terminal,))
            else:
                self._add(src_kind, byte_val, _IDLE, ())

    def _build_idle(self) -> None:
        src = _IDLE
        # Whitespace: stay idle
        for b in [ord(' '), ord('\t'), ord('\n'), ord('\r')]:
            self._add(src, b, _IDLE, ())
        # Start of string: go to KEY_Q (try key trie)
        self._add(src, ord('"'), _KEY_Q, ())
        # Start of number
        self._add(src, ord('-'), _NUM_INT, ())
        for d in range(ord('0'), ord('9') + 1):
            self._add(src, d, _NUM_INT, ())
        # Start of true/false/null
        self._add(src, ord('t'), _word_state('true', 1), ())
        self._add(src, ord('f'), _word_state('false', 1), ())
        self._add(src, ord('n'), _word_state('null', 1), ())
        # Structural bytes: emit immediately
        for b, t in [(ord('{'), ord('{')), (ord('}'), ord('}')),
                     (ord('['), ord('[')), (ord(']'), ord(']')),
                     (ord(':'), ord(':')), (ord(','), ord(','))]:
            self._add(src, b, _IDLE, (t,))

    def _build_key_q(self) -> None:
        """Just saw opening '"'. Try to match key prefixes."""
        src = _KEY_Q
        # Empty string: '"' immediately -> T_STRING
        self._add(src, ord('"'), _IDLE, (T_STRING,))
        # Escape: fall to generic string
        self._add(src, ord('\\'), _STR_ESC, ())

        # Bytes that match the first char of at least one key -> go to key state
        # Bytes that don't match any key first char -> go to STR
        first_chars: dict[int, list[int]] = {}  # byte -> list of key indices
        for i, key in enumerate(self.key_strings):
            if key:
                b = ord(key[0])
                first_chars.setdefault(b, []).append(i)

        # For each byte: if it matches a key first char, go to KS state(s)
        # Since multiple keys can share a first char, we need to handle ambiguity.
        # Solution: for each byte, go to a single state that tracks ALL matching keys.
        # We handle this by creating "multi-key" states when needed, or by
        # using a simpler approach: if exactly one key starts with this byte, use
        # that key's state. If multiple keys share a first byte, we create a
        # shared state (or just fall to STR for simplicity at cost of no key distinction).
        #
        # Since key collisions are rare in practice and we can always fall back
        # to T_STRING, we use a simple approach: track at most one key at a time.
        # If a byte matches multiple keys' first chars, we pick the first matching
        # key (this is safe: worst case we emit T_STRING for a key we could have
        # matched — but that means the grammar won't accept it, which is conservative).
        #
        # Better: build a proper key trie that tracks ALL simultaneously matching keys.
        # We implement this via a recursive trie structure.
        #
        # For now: full trie. Each key gets its own state chain. At KEY_Q, for each
        # first byte, we transition to the key-state for the FIRST matching key, with
        # the understanding that if that key fails, we fall to STR. Other keys starting
        # with the same byte are missed. This is a known limitation for ambiguous prefixes.
        #
        # The correct solution for ambiguous prefixes is to use a combined trie state
        # (tracking a frozenset of matching keys). We implement this below.

        # Build a combined trie over all keys simultaneously.
        # Each trie node = frozenset of (key_idx, pos) pairs meaning "key_idx matched up to pos"
        # We'll enumerate these as scanner states.
        # For KEY_Q, the starting node = {(i, 0) for all i}

        # Actually, the simpler and cleaner approach: use a proper prefix trie.
        # At any point in key matching, we track which keys are still candidates.
        # Each trie node is a frozenset of (key_idx, pos) where key[0:pos] has been seen.

        # Starting candidates after '"': all keys at position 0
        all_candidates = frozenset((i, 0) for i in range(len(self.key_strings)))
        # KEY_Q is equivalent to the trie node for all_candidates

        # Map trie nodes to scanner state kinds
        # The state kind for a trie node is f"KT_{sorted_items}"
        def trie_state(candidates: frozenset) -> str:
            if not candidates:
                return _STR  # no candidates remaining -> generic string
            items = sorted(candidates)
            return f"KT_{'_'.join(f'{i}p{p}' for i,p in items)}"

        # BFS over trie nodes
        from collections import deque
        trie_queue: deque = deque([all_candidates])
        trie_visited: set = {all_candidates}

        # KEY_Q -> trie node for all_candidates
        # We'll wire KEY_Q's transitions to go to trie states
        # First, collect all trie nodes reachable from KEY_Q

        all_trie_nodes: list[frozenset] = [all_candidates]

        while trie_queue:
            node = trie_queue.popleft()

            # For each possible byte: advance candidates that can take this byte
            # Group candidates by expected next byte
            by_next_byte: dict[int, list[tuple[int, int]]] = {}
            completed_keys: dict[int, int] = {}  # byte -> terminal_id (for keys ending here)

            for key_idx, pos in node:
                key = self.key_strings[key_idx]
                if pos < len(key):
                    b = ord(key[pos])
                    by_next_byte.setdefault(b, []).append((key_idx, pos + 1))
                # If pos == len(key): this key is fully matched, waiting for closing '"'
                # That's handled below as the '"' transition

            # Closing '"' from this trie node:
            # Keys where pos == len(key) are complete
            full_keys = [(key_idx, pos) for key_idx, pos in node if pos == len(self.key_strings[key_idx])]
            if len(full_keys) == 1:
                k_idx, _ = full_keys[0]
                # Emit T_KEY_BASE + k_idx when '"' is seen
                # Store as special: trie node + closing quote
                pass  # handled when building transitions for this node
            elif len(full_keys) > 1:
                # Ambiguous: multiple keys with same content (impossible if keys are distinct)
                pass

            # Advance to child trie nodes
            for byte_val, next_candidates in by_next_byte.items():
                child_node = frozenset(next_candidates)
                if child_node not in trie_visited:
                    trie_visited.add(child_node)
                    trie_queue.append(child_node)
                    all_trie_nodes.append(child_node)

        # Now build the actual scanner state transitions for all trie nodes
        # Map each trie node to a scanner state kind
        node_to_kind: dict[frozenset, str] = {}
        for node in all_trie_nodes:
            if node == all_candidates:
                node_to_kind[node] = _KEY_Q  # KEY_Q is the root trie node
            else:
                node_to_kind[node] = trie_state(node)
                self._sid(trie_state(node))  # ensure state exists

        for node in all_trie_nodes:
            src_kind = node_to_kind[node]

            # Closing '"': check if any key is fully matched
            full_keys = [(key_idx, pos) for key_idx, pos in node
                         if pos == len(self.key_strings[key_idx])]
            if len(full_keys) == 1:
                k_idx = full_keys[0][0]
                # Emit T_KEY_BASE + k_idx
                self._add(src_kind, ord('"'), _IDLE, (T_KEY_BASE + k_idx,))
            else:
                # No complete key, or ambiguous: closing '"' -> T_STRING
                self._add(src_kind, ord('"'), _IDLE, (T_STRING,))

            # Escape: fall to generic STR_ESC
            self._add(src_kind, ord('\\'), _STR_ESC, ())

            # Advance on bytes that match some key's next char
            by_next: dict[int, frozenset] = {}
            for key_idx, pos in node:
                key = self.key_strings[key_idx]
                if pos < len(key):
                    b = ord(key[pos])
                    candidates = by_next.get(b, frozenset())
                    by_next[b] = candidates | frozenset({(key_idx, pos + 1)})

            # For each byte with matching candidates: go to child trie node
            matched_bytes: set[int] = set(by_next.keys())
            for byte_val, child_candidates in by_next.items():
                # Find child node (should be in visited)
                child_kind = node_to_kind.get(child_candidates, _STR)
                self._add(src_kind, byte_val, child_kind, ())

            # For all other string-content bytes: fall to STR
            for byte_val in range(0x20, 0x100):
                if byte_val == ord('"') or byte_val == ord('\\'):
                    continue  # handled above
                if byte_val not in matched_bytes:
                    self._add(src_kind, byte_val, _STR, ())

    def _build_str(self) -> None:
        """Inside a generic string (key trie failed or not applicable)."""
        src = _STR
        # All printable bytes except '"' and '\\': stay in STR
        for b in range(0x20, 0x100):
            if b == ord('"'):
                self._add(src, b, _IDLE, (T_STRING,))
            elif b == ord('\\'):
                self._add(src, b, _STR_ESC, ())
            else:
                self._add(src, b, _STR, ())

    def _build_str_esc(self) -> None:
        """After '\\' inside a string."""
        src = _STR_ESC
        # Valid escape chars -> back to STR
        for b in [ord(c) for c in '"\\bfnrt/']:
            self._add(src, b, _STR, ())
        # 'u' for unicode escape: go to STR (simplified: don't validate hex digits)
        self._add(src, ord('u'), _STR, ())

    def _build_num_int(self) -> None:
        """In a number, integer part only so far."""
        src = _NUM_INT
        # More digits: stay in NUM_INT
        for d in range(ord('0'), ord('9') + 1):
            self._add(src, d, _NUM_INT, ())
        # Decimal point: go to NUM_FLOAT
        self._add(src, ord('.'), _NUM_FLOAT, ())
        # Exponent: go to NUM_EXP_SIGN
        self._add(src, ord('e'), _NUM_EXP_SIGN, ())
        self._add(src, ord('E'), _NUM_EXP_SIGN, ())
        # Structural bytes: emit T_INTEGER then the structural terminal
        self._structural_transition(src, pending_terminal=T_INTEGER)

    def _build_num_float(self) -> None:
        """In a number after seeing '.' or 'e'/'E' (will emit T_NUMBER)."""
        src = _NUM_FLOAT
        for d in range(ord('0'), ord('9') + 1):
            self._add(src, d, _NUM_FLOAT, ())
        self._add(src, ord('e'), _NUM_EXP_SIGN, ())
        self._add(src, ord('E'), _NUM_EXP_SIGN, ())
        self._structural_transition(src, pending_terminal=T_NUMBER)

    def _build_num_exp_sign(self) -> None:
        """In number exponent, after 'e'/'E', expecting optional sign or digit."""
        src = _NUM_EXP_SIGN
        self._add(src, ord('+'), _NUM_FLOAT, ())
        self._add(src, ord('-'), _NUM_FLOAT, ())
        for d in range(ord('0'), ord('9') + 1):
            self._add(src, d, _NUM_FLOAT, ())

    def _build_words(self) -> None:
        """States for recognizing true, false, null."""
        words = {
            'true':  T_TRUE,
            'false': T_FALSE,
            'null':  T_NULL,
        }
        for word, terminal in words.items():
            for pos in range(1, len(word)):
                src = _word_state(word, pos)
                expected_byte = ord(word[pos])
                if pos + 1 == len(word):
                    # Last char: advance to "word complete" -> need to emit on next structural
                    # We use a terminal state WORD_DONE_{word} that emits on structural/ws
                    done_state = f"WORD_DONE_{word}"
                    self._sid(done_state)
                    self._add(src, expected_byte, done_state, ())
                else:
                    self._add(src, expected_byte, _word_state(word, pos + 1), ())
                # Any other byte -> DEAD (invalid keyword)
                # (no transition = DEAD by default)

            # Build WORD_DONE_{word} transitions
            done_state = f"WORD_DONE_{word}"
            self._structural_transition(done_state, pending_terminal=terminal)
            # Whitespace handled by _structural_transition
            # End of input: accept_if_done check is done at parser level

    # ------------------------------------------------------------------
    # Public interface

    def step(self, state: int, byte_val: int) -> tuple[int, tuple[int, ...]]:
        """
        Advance scanner by one byte.

        Returns:
            (new_state, terminals) where:
            - new_state == DEAD means this byte is invalid from current state
            - terminals is a tuple of 0, 1, or 2 terminal IDs emitted by this byte
        """
        if state == DEAD:
            return DEAD, ()
        trans = self._trans[state]
        if byte_val not in trans:
            return DEAD, ()
        new_state, terminals = trans[byte_val]
        return new_state, terminals

    def step_seq(self, state: int, byte_seq: bytes) -> tuple[int, list[int]]:
        """
        Advance scanner through a byte sequence.

        Returns:
            (final_state, all_terminals_emitted)
        Stops early if DEAD is reached. final_state == DEAD on failure.
        """
        all_terminals: list[int] = []
        for b in byte_seq:
            state, terminals = self.step(state, b)
            all_terminals.extend(terminals)
            if state == DEAD:
                return DEAD, all_terminals
        return state, all_terminals

    def pending_terminal(self, state: int) -> int | None:
        """
        If the scanner has accumulated a complete lexeme not yet emitted
        (because no terminating structural byte arrived), return that terminal.

        Used at end-of-input to flush the last token.
        Returns None if no pending lexeme.
        """
        if state == DEAD:
            return None
        kind = self._states[state]
        if kind == _NUM_INT:
            return T_INTEGER
        if kind == _NUM_FLOAT:
            return T_NUMBER
        if kind == "WORD_DONE_true":
            return T_TRUE
        if kind == "WORD_DONE_false":
            return T_FALSE
        if kind == "WORD_DONE_null":
            return T_NULL
        return None

    def state_kind(self, state: int) -> str:
        """Return the kind string for a state (for debugging)."""
        if state == DEAD:
            return "DEAD"
        return self._states[state]

    def describe(self) -> str:
        """Human-readable description of the scanner."""
        lines = [f"JsonScanner: {self.num_states} states"]
        lines.append(f"  start_state: {self.start_state} ({self._states[self.start_state]})")
        for sid, kind in enumerate(self._states):
            trans = self._trans[sid]
            lines.append(f"  State {sid} ({kind}): {len(trans)} transitions")
        return "\n".join(lines)