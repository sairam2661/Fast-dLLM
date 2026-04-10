"""
Token trie for fast valid-set computation.

Organizes the vocabulary into a byte prefix tree. Valid-set computation
traverses the trie with a set of composite (scanner_state, parser_config)
pairs, pruning entire subtrees when all pairs die.

Two backends are supported via duck typing on the `automaton` argument:

  DFA backend (legacy):
    - automaton is a DFA instance
    - scanner is None
    - composite states are plain DFA state integers
    - traversal calls dfa.transition(state, byte)

  Scanner+LR backend (new):
    - automaton is a BoundedLRAutomaton
    - scanner is a JsonScanner
    - composite state = scanner_state * automaton.num_configs + parser_config
    - traversal calls scanner.step(sc_state, byte) -> (new_sc, terminals)
      then advances parser configs through each terminal

The public interface is unchanged: compute_valid_set takes left_exits and
right_entries as frozenset[int] of composite IDs (or plain DFA state ints
in legacy mode), and returns a set of valid token IDs.

Usage (new backend):
    trie = TokenTrie(token_to_bytes={...})
    valid = trie.compute_valid_set(
        left_exits=frozenset({composite_id, ...}),
        right_entries=frozenset({composite_id, ...}),
        automaton=blr,
        scanner=json_scanner,
    )
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING


class TokenTrie:
    """
    Prefix tree over vocabulary tokens, organized by byte content.

    Flat array representation for cache-friendly traversal:
    - byte_vals[i]: byte value of node i (-1 for root)
    - token_ids[i]: token ID if node i completes a token, else -1
    - first_child[i]: index of first child, or -1 if leaf
    - next_sibling[i]: index of next sibling, or -1 if last
    """

    def __init__(self, token_to_bytes: dict[int, bytes]):
        self.vocab_size = max(token_to_bytes.keys()) + 1 if token_to_bytes else 0
        self._build(token_to_bytes)

    def _build(self, token_to_bytes: dict[int, bytes]):
        root = [-1, -1, {}]
        for token_id, byte_seq in token_to_bytes.items():
            if len(byte_seq) == 0:
                if root[1] == -1:
                    root[1] = token_id
                continue
            node = root
            for b in byte_seq:
                if b not in node[2]:
                    node[2][b] = [-1, -1, {}]
                node = node[2][b]
            node[1] = token_id

        self._byte_vals: list[int] = []
        self._token_ids: list[int] = []
        self._first_child: list[int] = []
        self._next_sibling: list[int] = []
        self._flatten(root)

    def _flatten(self, tree_node):
        idx = len(self._byte_vals)
        self._byte_vals.append(tree_node[0])
        self._token_ids.append(tree_node[1])
        self._first_child.append(-1)
        self._next_sibling.append(-1)
        children = sorted(tree_node[2].items())
        prev_child_idx = -1
        for byte_val, child_node in children:
            child_node[0] = byte_val
            child_idx = len(self._byte_vals)
            if prev_child_idx == -1:
                self._first_child[idx] = child_idx
            else:
                self._next_sibling[prev_child_idx] = child_idx
            self._flatten(child_node)
            prev_child_idx = child_idx

    @property
    def num_nodes(self) -> int:
        return len(self._byte_vals)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_valid_set(
        self,
        left_exits: frozenset[int],
        right_entries: "frozenset[int] | None",
        automaton,
        scanner=None,
    ) -> set[int]:
        """
        Compute set of valid token IDs.

        Args:
            left_exits: composite state IDs (or DFA states) from left context.
            right_entries: composite state IDs required by right context,
                          or None if unconstrained.
            automaton: CompositeAutomaton (new), BoundedLRAutomaton (direct), or DFA (legacy).
            scanner: JsonScanner instance, or None for DFA legacy mode.
                     If automaton is a CompositeAutomaton, scanner is extracted automatically.

        Returns:
            Set of valid token IDs.
        """
        valid: set[int] = set()

        # Unwrap CompositeAutomaton to get the raw lr and scanner
        lr_automaton = automaton
        sc = scanner
        if hasattr(automaton, '_lr') and hasattr(automaton, '_sc'):
            # CompositeAutomaton: extract internals for the trie traversal
            # The trie carries (scanner_state, parser_config) composite IDs and
            # needs direct access to lr._trans and scanner.step
            lr_automaton = automaton._lr
            sc = automaton._sc
            num_configs = automaton.num_configs
        else:
            num_configs = getattr(automaton, 'num_configs', None)

        # Empty-byte token at root
        root_tid = self._token_ids[0]
        if root_tid != -1:
            if right_entries is None or (left_exits & right_entries):
                valid.add(root_tid)

        if sc is not None:
            child = self._first_child[0]
            while child != -1:
                self._traverse_lr(
                    child, left_exits, right_entries, valid,
                    lr_automaton=lr_automaton, scanner=sc, num_configs=num_configs,
                )
                child = self._next_sibling[child]
        else:
            child = self._first_child[0]
            while child != -1:
                self._traverse_dfa(child, left_exits, right_entries, valid, dfa=automaton)
                child = self._next_sibling[child]

        return valid

    # ------------------------------------------------------------------
    # DFA traversal (legacy, unchanged logic)
    # ------------------------------------------------------------------

    def _traverse_dfa(
        self,
        node_idx: int,
        current_states: frozenset[int],
        right_entries: "frozenset[int] | None",
        valid: set[int],
        dfa,
    ):
        from constrained.dfa import DEAD
        byte_val = self._byte_vals[node_idx]

        next_states: set[int] = set()
        for q in current_states:
            nq = dfa.transition(q, byte_val)
            if nq != DEAD:
                next_states.add(nq)

        if not next_states:
            return

        next_frozen = frozenset(next_states)
        tid = self._token_ids[node_idx]
        if tid != -1:
            if right_entries is None or (next_frozen & right_entries):
                valid.add(tid)

        child = self._first_child[node_idx]
        while child != -1:
            self._traverse_dfa(child, next_frozen, right_entries, valid, dfa)
            child = self._next_sibling[child]

    # ------------------------------------------------------------------
    # Scanner + LR traversal (new)
    # ------------------------------------------------------------------

    def _traverse_lr(
        self,
        node_idx: int,
        current_pairs: frozenset[int],
        right_entries: "frozenset[int] | None",
        valid: set[int],
        lr_automaton,
        scanner,
        num_configs: int,
    ):
        """
        Trie traversal carrying composite (scanner_state, parser_config) pairs
        encoded as: composite = sc * num_configs + pc.

        lr_automaton is the raw BoundedLRAutomaton (._trans accessed directly).
        scanner is the JsonScanner.
        num_configs is lr_automaton.num_configs.
        """
        byte_val = self._byte_vals[node_idx]
        SCANNER_DEAD = scanner.dead_state  # -1

        next_pairs: set[int] = set()
        for composite in current_pairs:
            sc = composite // num_configs
            pc = composite % num_configs

            new_sc, terminals = scanner.step(sc, byte_val)
            if new_sc == SCANNER_DEAD:
                continue

            # Advance parser config through all emitted terminals
            current_pcs: frozenset[int] = frozenset({pc})
            for terminal in terminals:
                nxt: set[int] = set()
                for c in current_pcs:
                    nxt.update(lr_automaton._trans[c].get(terminal, frozenset()))
                if not nxt:
                    current_pcs = frozenset()
                    break
                current_pcs = frozenset(nxt)

            if not current_pcs:
                continue

            for new_pc in current_pcs:
                next_pairs.add(new_sc * num_configs + new_pc)

        if not next_pairs:
            return

        next_frozen = frozenset(next_pairs)

        tid = self._token_ids[node_idx]
        if tid != -1:
            if right_entries is None or (next_frozen & right_entries):
                valid.add(tid)

        child = self._first_child[node_idx]
        while child != -1:
            self._traverse_lr(child, next_frozen, right_entries, valid,
                              lr_automaton=lr_automaton, scanner=scanner, num_configs=num_configs)
            child = self._next_sibling[child]

    # ------------------------------------------------------------------
    # Legacy mask interface (DFA only)
    # ------------------------------------------------------------------

    def compute_valid_mask(
        self,
        left_exits: frozenset[int],
        right_entries: "frozenset[int] | None",
        automaton,
        scanner=None,
    ) -> list[bool]:
        valid = self.compute_valid_set(left_exits, right_entries, automaton, scanner)
        mask = [False] * self.vocab_size
        for tid in valid:
            mask[tid] = True
        return mask

    def stats(self) -> dict:
        num_tokens = sum(1 for t in self._token_ids if t != -1)
        num_leaves = sum(
            1 for i in range(len(self._first_child))
            if self._first_child[i] == -1
        )
        max_depth = 0
        def depth(idx, d):
            nonlocal max_depth
            max_depth = max(max_depth, d)
            child = self._first_child[idx]
            while child != -1:
                depth(child, d + 1)
                child = self._next_sibling[child]
        depth(0, 0)
        return {
            'num_nodes': self.num_nodes,
            'num_tokens': num_tokens,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'vocab_size': self.vocab_size,
        }