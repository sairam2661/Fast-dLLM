"""
CFG representation, LR(0) automaton, and bounded stack prefix enumeration.

Replaces dfa.py in the constrained decoding pipeline.

Three layers:
  1. Grammar          -- production rules, terminal/nonterminal sets
  2. LR0Automaton     -- item sets, shift/goto/reduce tables (standard LR(0))
  3. BoundedLRAutomaton -- enumerates stack prefixes up to depth d,
                           assigns integer ConfigIDs, exposes the segment
                           interface that segments.py expects.

Segment interface contract (mirrors DFA):
  - num_configs: int
  - start_config: int
  - accept_configs: frozenset[int]
  - transition_seq(config_id, byte_seq) -> frozenset[int]
  - all_configs() -> range

Key difference from DFA: transition_seq returns a SET (LR parsing is
nondeterministic from a bounded prefix because reductions may or may not
apply depending on context we don't have). The segment operations in
segments.py fan out over this set naturally.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple


# ---------------------------------------------------------------------------
# 1. Grammar
# ---------------------------------------------------------------------------

TERMINAL = "T"
NONTERMINAL = "NT"


class Symbol(NamedTuple):
    kind: str   # TERMINAL or NONTERMINAL
    value: int  # byte value (0-255) for terminals; nonterminal id for NTs


@dataclass
class Grammar:
    """
    A context-free grammar.

    Attributes:
        nonterminals: list of nonterminal names indexed by id.
                      nonterminals[0] is always the start symbol.
        rules: list of (lhs_nt_id, rhs) where rhs is a tuple of Symbols.
               May include epsilon rules where rhs = ().
        start: nonterminal id of the start symbol (always 0).
        augmented_start: set by LR0Automaton after augmentation.
    """
    nonterminals: list[str]
    rules: list[tuple[int, tuple[Symbol, ...]]]
    start: int = 0
    augmented_start: int = field(default=-1, init=False)

    @staticmethod
    def from_rules(
        nonterminals: list[str],
        rules: list[tuple[str, list]],
    ) -> "Grammar":
        """
        Convenience constructor.

        Args:
            nonterminals: list of nonterminal names. Index 0 = start symbol.
            rules: list of (lhs_name, rhs_list) where rhs_list contains
                   either integers (byte terminals) or strings (nonterminal names).

        Example:
            Grammar.from_rules(
                nonterminals=["S"],
                rules=[
                    ("S", [ord("("), "S", ord(")")]),
                    ("S", []),
                ],
            )
        """
        nt_index = {name: i for i, name in enumerate(nonterminals)}
        parsed_rules = []
        for lhs_name, rhs_list in rules:
            lhs = nt_index[lhs_name]
            rhs = []
            for sym in rhs_list:
                if isinstance(sym, int):
                    rhs.append(Symbol(TERMINAL, sym))
                elif isinstance(sym, str):
                    rhs.append(Symbol(NONTERMINAL, nt_index[sym]))
                else:
                    raise ValueError(f"Unknown symbol type: {sym!r}")
            parsed_rules.append((lhs, tuple(rhs)))
        return Grammar(nonterminals=list(nonterminals), rules=parsed_rules)

    def rules_for(self, nt: int) -> list[tuple[int, tuple[Symbol, ...]]]:
        """All rules with lhs == nt. Returns list of (rule_index, rhs)."""
        return [(i, rhs) for i, (lhs, rhs) in enumerate(self.rules) if lhs == nt]

    def nt_name(self, nt_id: int) -> str:
        return self.nonterminals[nt_id]


# ---------------------------------------------------------------------------
# 2. LR(0) Automaton
# ---------------------------------------------------------------------------

class Item(NamedTuple):
    """
    LR(0) item: (rule_index, dot_position).
    dot == len(rhs) means the item is complete (reduce).
    """
    rule_index: int
    dot: int


class LR0Automaton:
    """
    Standard LR(0) automaton.

    States are item sets (frozensets of Items), assigned integer IDs.
    Builds shift, goto, and reduce tables.

    Grammar is augmented automatically: S' -> S is prepended as rule 0,
    shifting original rule indices up by one.
    """

    def __init__(self, grammar: Grammar) -> None:
        # Augment: prepend S' -> S as rule 0
        aug_nt_id = len(grammar.nonterminals)
        self.grammar = Grammar(
            nonterminals=grammar.nonterminals + ["S'"],
            rules=[(aug_nt_id, (Symbol(NONTERMINAL, grammar.start),))]
                  + grammar.rules,
        )
        self.grammar.start = grammar.start
        self.grammar.augmented_start = aug_nt_id

        self._item_sets: list[frozenset[Item]] = []
        self._item_set_ids: dict[frozenset[Item], int] = {}

        # shift[state][byte] -> state
        self.shift: list[dict[int, int]] = []
        # goto[state][nt_id] -> state
        self.goto: list[dict[int, int]] = []
        # reduce[state] -> list of (lhs_nt_id, rhs_length)
        self.reduce: list[list[tuple[int, int]]] = []

        self._build()

    def _closure(self, items: frozenset[Item]) -> frozenset[Item]:
        result: set[Item] = set(items)
        queue: deque[Item] = deque(items)
        while queue:
            item = queue.popleft()
            _, rhs = self.grammar.rules[item.rule_index]
            if item.dot < len(rhs):
                next_sym = rhs[item.dot]
                if next_sym.kind == NONTERMINAL:
                    for i, (lhs, _) in enumerate(self.grammar.rules):
                        if lhs == next_sym.value:
                            new_item = Item(i, 0)
                            if new_item not in result:
                                result.add(new_item)
                                queue.append(new_item)
        return frozenset(result)

    def _goto_set(self, item_set: frozenset[Item], symbol: Symbol) -> frozenset[Item]:
        advanced = {Item(item.rule_index, item.dot + 1)
                    for item in item_set
                    if item.dot < len(self.grammar.rules[item.rule_index][1])
                    and self.grammar.rules[item.rule_index][1][item.dot] == symbol}
        return self._closure(frozenset(advanced)) if advanced else frozenset()

    def _get_or_create(self, item_set: frozenset[Item]) -> tuple[int, bool]:
        if item_set in self._item_set_ids:
            return self._item_set_ids[item_set], False
        sid = len(self._item_sets)
        self._item_sets.append(item_set)
        self._item_set_ids[item_set] = sid
        self.shift.append({})
        self.goto.append({})
        self.reduce.append([])
        return sid, True

    def _build(self) -> None:
        start_items = self._closure(frozenset({Item(0, 0)}))
        start_id, _ = self._get_or_create(start_items)

        queue: deque[int] = deque([start_id])
        visited: set[int] = {start_id}

        while queue:
            sid = queue.popleft()
            item_set = self._item_sets[sid]

            next_terminals: set[int] = set()
            next_nts: set[int] = set()

            for item in item_set:
                _, rhs = self.grammar.rules[item.rule_index]
                if item.dot < len(rhs):
                    sym = rhs[item.dot]
                    if sym.kind == TERMINAL:
                        next_terminals.add(sym.value)
                    else:
                        next_nts.add(sym.value)
                else:
                    lhs, rhs_complete = self.grammar.rules[item.rule_index]
                    if lhs != self.grammar.augmented_start:
                        self.reduce[sid].append((lhs, len(rhs_complete)))

            for byte_val in next_terminals:
                target = self._goto_set(item_set, Symbol(TERMINAL, byte_val))
                if target:
                    tid, is_new = self._get_or_create(target)
                    self.shift[sid][byte_val] = tid
                    if is_new and tid not in visited:
                        visited.add(tid)
                        queue.append(tid)

            for nt in next_nts:
                target = self._goto_set(item_set, Symbol(NONTERMINAL, nt))
                if target:
                    tid, is_new = self._get_or_create(target)
                    self.goto[sid][nt] = tid
                    if is_new and tid not in visited:
                        visited.add(tid)
                        queue.append(tid)

        self.num_states = len(self._item_sets)
        self.start_state = start_id
        accept_item = Item(0, 1)
        self.accept_states = frozenset(
            sid for sid, iset in enumerate(self._item_sets)
            if accept_item in iset
        )

    def has_shift_reduce_conflict(self) -> bool:
        for sid in range(self.num_states):
            if self.reduce[sid] and (self.shift[sid] or self.goto[sid]):
                return True
        return False

    def describe_state(self, sid: int) -> str:
        lines = [f"State {sid}:"]
        for item in sorted(self._item_sets[sid]):
            lhs, rhs = self.grammar.rules[item.rule_index]
            parts = []
            for i, sym in enumerate(rhs):
                if i == item.dot:
                    parts.append("•")
                if sym.kind == TERMINAL:
                    b = sym.value
                    parts.append(repr(chr(b)) if 32 <= b < 127 else f"\\x{b:02x}")
                else:
                    parts.append(self.grammar.nonterminals[sym.value])
            if item.dot == len(rhs):
                parts.append("•")
            lhs_name = self.grammar.nonterminals[lhs]
            lines.append(f"  {lhs_name} -> {' '.join(parts)}")
        for lhs, rhs_len in self.reduce[sid]:
            lines.append(f"  [REDUCE {self.grammar.nonterminals[lhs]}, pop {rhs_len}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. BoundedLRAutomaton
# ---------------------------------------------------------------------------

StackPrefix = tuple[int, ...]  # tuple of LR state IDs; prefix[-1] = top


class BoundedLRAutomaton:
    """
    Segment-interface wrapper around LR0Automaton.

    Enumerates all stack prefixes of depth <= d reachable from the start
    config, assigns integer ConfigIDs, and precomputes byte transitions.

    Drop-in replacement for DFA in segment operations. Key difference:
    transition_seq returns frozenset[int] instead of int.
    """

    def __init__(self, grammar: Grammar, depth: int) -> None:
        """
        Args:
            grammar: the CFG to parse
            depth: stack prefix depth bound. Reductions that would pop more
                   states than exist in the prefix are dropped (that prefix
                   is too shallow to be a valid entry for this token).
        """
        self.lr = LR0Automaton(grammar)
        self.depth = depth

        self._prefixes: list[StackPrefix] = []
        self._prefix_ids: dict[StackPrefix, int] = {}

        # _trans[config_id][byte] -> frozenset[config_id]
        self._trans: list[dict[int, frozenset[int]]] = []

        self._enumerate()

        self.num_configs = len(self._prefixes)
        self.start_config = self._prefix_ids.get((self.lr.start_state,), 0)

        # A config is accepting if, after exhaustively applying all possible
        # reductions (including epsilon reductions), some resulting stack has
        # its top state in lr.accept_states. This handles grammars with epsilon
        # productions where the final state is reached only after a reduction.
        self.accept_configs = frozenset(
            cid for cid, prefix in enumerate(self._prefixes)
            if self._prefix_can_accept(prefix)
        )

    def _prefix_can_accept(self, prefix: StackPrefix) -> bool:
        """True if exhaustive reductions from prefix can reach an accept state."""
        for stack in self._apply_reductions(prefix):
            if stack and stack[-1] in self.lr.accept_states:
                return True
        return False

    # ------------------------------------------------------------------
    # Reduction simulation

    def _apply_one_reduce(
        self, stack: tuple[int, ...], lhs: int, pop_len: int
    ) -> tuple[int, ...] | None:
        """
        Apply one reduction. Returns new stack or None if too shallow.

        Pops pop_len states, looks up GOTO on exposed top, pushes result.
        """
        if len(stack) < pop_len:
            return None
        new_stack = stack[:-pop_len] if pop_len > 0 else stack
        if not new_stack:
            return None
        exposed = new_stack[-1]
        goto_state = self.lr.goto[exposed].get(lhs)
        if goto_state is None:
            return None
        return new_stack + (goto_state,)

    def _apply_reductions(self, stack: tuple[int, ...]) -> set[tuple[int, ...]]:
        """
        Exhaustively apply all reduction chains from stack via BFS.

        A config is included in results if:
          (a) its top state has no reduces (purely a shift state), OR
          (b) it has shifts available — because the next byte may need the
              shift path, even if reduces are also possible (shift/reduce conflict), OR
          (c) all its reductions were dropped (too shallow) — nowhere to go.

        We also continue exploring reductions from (b) to find all reachable
        post-reduction configs.

        This handles grammars with epsilon productions correctly: from a state
        with both "shift '('" and "reduce S->ε", we output the current config
        (to allow future shifts) AND the epsilon-reduced config (to allow ')').
        """
        results: set[tuple[int, ...]] = set()
        queue: deque[tuple[int, ...]] = deque([stack])
        visited: set[tuple[int, ...]] = {stack}

        while queue:
            current = queue.popleft()
            top = current[-1]
            reduces = self.lr.reduce[top]

            # Include this config if it has available shifts (next byte can use them)
            if self.lr.shift[top]:
                results.add(current)

            if not reduces:
                # No reductions — if no shifts either, still add (dead end for bytes,
                # but we need it to represent the config exists)
                if not self.lr.shift[top]:
                    results.add(current)
                continue

            any_succeeded = False
            for lhs, pop_len in reduces:
                new_stack = self._apply_one_reduce(current, lhs, pop_len)
                if new_stack is not None and new_stack not in visited:
                    visited.add(new_stack)
                    queue.append(new_stack)
                    any_succeeded = True

            if not any_succeeded:
                # All reductions needed more context — include as-is
                results.add(current)

        return results

    def _transition_prefix(self, prefix: StackPrefix, byte_val: int) -> set[StackPrefix]:
        """
        All stack prefixes reachable from prefix after shifting byte_val.

        1. Shift: push lr.shift[top][byte_val] onto working stack.
        2. Reduce: exhaustively apply all reduction chains.
        3. Truncate each result to depth d (keep rightmost d elements).
        """
        top = prefix[-1]
        shifted = self.lr.shift[top].get(byte_val)
        if shifted is None:
            return set()

        working = prefix + (shifted,)
        terminal_stacks = self._apply_reductions(working)

        result: set[StackPrefix] = set()
        for stack in terminal_stacks:
            if not stack:
                continue
            truncated = stack[-self.depth:] if len(stack) > self.depth else stack
            result.add(truncated)

        return result

    # ------------------------------------------------------------------
    # BFS enumeration

    def _get_or_create(self, prefix: StackPrefix) -> tuple[int, bool]:
        if prefix in self._prefix_ids:
            return self._prefix_ids[prefix], False
        cid = len(self._prefixes)
        self._prefixes.append(prefix)
        self._prefix_ids[prefix] = cid
        self._trans.append({})
        return cid, True

    def _enumerate(self) -> None:
        start_prefix = (self.lr.start_state,)
        start_id, _ = self._get_or_create(start_prefix)

        queue: deque[int] = deque([start_id])
        visited: set[int] = {start_id}

        while queue:
            cid = queue.popleft()
            prefix = self._prefixes[cid]
            top = prefix[-1]

            for byte_val in self.lr.shift[top]:
                target_prefixes = self._transition_prefix(prefix, byte_val)
                if not target_prefixes:
                    continue

                target_ids: set[int] = set()
                for tp in target_prefixes:
                    tid, is_new = self._get_or_create(tp)
                    target_ids.add(tid)
                    if is_new and tid not in visited:
                        visited.add(tid)
                        queue.append(tid)

                self._trans[cid][byte_val] = frozenset(target_ids)

    # ------------------------------------------------------------------
    # Segment interface

    def transition_seq(self, config_id: int, byte_seq: bytes) -> frozenset[int]:
        """
        All config IDs reachable from config_id after byte_seq.
        Returns empty frozenset if the sequence is dead from this config.
        """
        current: frozenset[int] = frozenset({config_id})
        for byte_val in byte_seq:
            nxt: set[int] = set()
            for cid in current:
                nxt.update(self._trans[cid].get(byte_val, frozenset()))
            if not nxt:
                return frozenset()
            current = frozenset(nxt)
        return current

    def all_configs(self) -> range:
        """All valid config IDs. Replaces range(dfa.num_states)."""
        return range(self.num_configs)

    def config_prefix(self, config_id: int) -> StackPrefix:
        """Stack prefix for a config ID (for debugging)."""
        return self._prefixes[config_id]

    def describe_config(self, config_id: int) -> str:
        prefix = self._prefixes[config_id]
        return f"Config {config_id}: [{', '.join(f'LR{s}' for s in prefix)}]"


# ---------------------------------------------------------------------------
# Convenience constructors for testing
# ---------------------------------------------------------------------------

def balanced_parens_grammar() -> Grammar:
    """
    S -> ( S ) | S S | ε

    Note: S -> S S introduces an LR(0) conflict (this grammar is ambiguous).
    Use the non-ambiguous version for cleaner testing:
      S -> ( S ) S | ε
    which is equivalent but unambiguous for the purpose of acceptance.
    """
    return Grammar.from_rules(
        nonterminals=["S"],
        rules=[
            ("S", [ord("("), "S", ord(")"), "S"]),
            ("S", []),
        ],
    )


def simple_arithmetic_grammar() -> Grammar:
    """
    E -> E + T | T
    T -> T * F | F
    F -> ( E ) | d    (d = byte ord('d'))
    """
    return Grammar.from_rules(
        nonterminals=["E", "T", "F"],
        rules=[
            ("E", ["E", ord("+"), "T"]),
            ("E", ["T"]),
            ("T", ["T", ord("*"), "F"]),
            ("T", ["F"]),
            ("F", [ord("("), "E", ord(")")]),
            ("F", [ord("d")]),
        ],
    )