"""
Token trie for fast valid-set computation.

Organizes the vocabulary into a byte prefix tree. Valid-set computation
traverses the trie with a set of DFA states, pruning entire subtrees
when no valid transition exists.

This is the LLGuidance approach extended to segment-based constraints:
instead of traversing with a single DFA state, we traverse with a set
of exit states from the left segment and filter completions against
entry states from the right segment.

Usage:
    trie = TokenTrie(token_to_bytes={0: b'hello', 1: b'{', ...})

    # Get valid tokens between two segments
    valid = trie.compute_valid_set(
        left_exits=frozenset({3, 7}),
        right_entries=frozenset({12}),
        dfa=dfa,
    )
"""

from __future__ import annotations
from typing import Optional

from constrained.dfa import DFA, DEAD


class TokenTrie:
    """
    Prefix tree over vocabulary tokens, organized by byte content.

    Enables fast valid-set computation by pruning entire subtrees
    when a byte transition leads to dead states.

    Internally stored as flat arrays for cache-friendly traversal:
    - byte_vals[i]: byte value of node i (-1 for root)
    - token_ids[i]: token ID if node i completes a token, else -1
    - first_child[i]: index of first child, or -1 if leaf
    - next_sibling[i]: index of next sibling, or -1 if last
    """

    def __init__(self, token_to_bytes: dict[int, bytes]):
        """
        Build trie from token-to-bytes mapping.

        Args:
            token_to_bytes: dict mapping token_id -> byte sequence.
                           Empty byte sequences are skipped.
        """
        self.vocab_size = max(token_to_bytes.keys()) + 1 if token_to_bytes else 0

        # Build tree structure, then flatten
        self._build(token_to_bytes)

    def _build(self, token_to_bytes: dict[int, bytes]):
        """Build flat array representation via temporary tree."""
        # Temporary tree nodes: [byte_val, token_id, children_dict]
        root = [-1, -1, {}]

        for token_id, byte_seq in token_to_bytes.items():
            if len(byte_seq) == 0:
                # Empty token — valid from any state to the same state
                # Store at root level
                if root[1] == -1:
                    root[1] = token_id
                continue

            node = root
            for b in byte_seq:
                if b not in node[2]:
                    node[2][b] = [-1, -1, {}]
                node = node[2][b]
            node[1] = token_id  # mark as completing a token

        # Flatten via DFS
        self._byte_vals: list[int] = []
        self._token_ids: list[int] = []
        self._first_child: list[int] = []
        self._next_sibling: list[int] = []

        self._flatten(root)

    def _flatten(self, tree_node):
        """DFS flatten tree into arrays."""
        idx = len(self._byte_vals)
        self._byte_vals.append(tree_node[0])
        self._token_ids.append(tree_node[1])
        self._first_child.append(-1)
        self._next_sibling.append(-1)

        # Sort children by byte value for deterministic traversal
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

    def compute_valid_set(
        self,
        left_exits: frozenset[int],
        right_entries: frozenset[int] | None,
        dfa: DFA,
    ) -> set[int]:
        """
        Compute set of valid token IDs.

        Traverses the trie with the set of left exit states.
        At each byte, computes next states via DFA transitions.
        Prunes subtrees where the state set becomes empty.
        A token is valid if, after processing all its bytes, some
        resulting state is in right_entries.

        Args:
            left_exits: DFA states from the left context.
            right_entries: DFA states required by right context.
                          If None, all non-DEAD states are accepted.
            dfa: the constraint DFA.

        Returns:
            Set of valid token IDs.
        """
        valid = set()
        # Start traversal from root's children
        # (root itself has byte_val=-1 and represents the empty prefix)

        # Check if root completes a token (empty byte sequence token)
        root_tid = self._token_ids[0]
        if root_tid != -1:
            # Empty token: state doesn't change. Valid if some left exit
            # is in right entries.
            if right_entries is None or (left_exits & right_entries):
                valid.add(root_tid)

        child = self._first_child[0]
        while child != -1:
            self._traverse(child, left_exits, right_entries, dfa, valid)
            child = self._next_sibling[child]

        return valid

    def _traverse(
        self,
        node_idx: int,
        current_states: frozenset[int],
        right_entries: frozenset[int] | None,
        dfa: DFA,
        valid: set[int],
    ):
        """Recursive trie traversal with state-set pruning."""
        byte_val = self._byte_vals[node_idx]

        # Advance all current states through this byte
        next_states = set()
        for q in current_states:
            nq = dfa.transition(q, byte_val)
            if nq != DEAD:
                next_states.add(nq)

        if not next_states:
            return  # prune: no states survive this byte

        next_frozen = frozenset(next_states)

        # Check if this node completes a token
        tid = self._token_ids[node_idx]
        if tid != -1:
            if right_entries is None or (next_frozen & right_entries):
                valid.add(tid)

        # Recurse into children
        child = self._first_child[node_idx]
        while child != -1:
            self._traverse(child, next_frozen, right_entries, dfa, valid)
            child = self._next_sibling[child]

    def compute_valid_mask(
        self,
        left_exits: frozenset[int],
        right_entries: frozenset[int] | None,
        dfa: DFA,
    ) -> list[bool]:
        """
        Compute boolean mask over vocabulary.
        mask[token_id] = True if the token is valid.
        """
        valid = self.compute_valid_set(left_exits, right_entries, dfa)
        mask = [False] * self.vocab_size
        for tid in valid:
            mask[tid] = True
        return mask

    def stats(self) -> dict:
        """Trie statistics for debugging."""
        num_tokens = sum(1 for t in self._token_ids if t != -1)
        num_leaves = sum(
            1 for i in range(len(self._first_child))
            if self._first_child[i] == -1
        )
        # Compute max depth
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