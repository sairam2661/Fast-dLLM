"""
Segment Manager for order-agnostic incremental parsing.

Maintains the collection of segments across the generation region.
Routes token revelations to the correct primitive operation (create,
extend_left, extend_right, merge_with_bridge).
Provides valid-token-set queries for masked positions.

Usage:
	mgr = SegmentManager(dfa=dfa, gen_start=0, gen_length=128,
						 token_to_bytes=tokenizer_lookup)
	# As the diffusion model unmasks tokens:
	mgr.reveal_token(position=42, token_id=1537)
	# Before committing a token, check what's valid:
	valid = mgr.get_valid_tokens(position=43, vocab_size=32000)
"""

from __future__ import annotations
from typing import Optional, Callable

from constrained.dfa import DFA, DEAD
from constrained.segments import (
	Segment, create, extend_right, extend_left, merge, merge_with_bridge,
)


class SegmentManager:
	"""
	Manages segments across the generation region.

	Segments are stored in a list sorted by start position.
	The number of segments is bounded by gen_length and typically
	much smaller (segments merge as gaps fill in), so linear scans
	are fine.
	"""

	def __init__(
		self,
		dfa: DFA,
		gen_start: int,
		gen_length: int,
		token_to_bytes: Callable[[int], bytes],
	):
		"""
		Args:
			dfa: The DFA for constraint checking.
			gen_start: Start position of the generation region.
			gen_length: Number of positions in the generation region.
			token_to_bytes: Function mapping token_id -> byte sequence.
		"""
		self.dfa = dfa
		self.gen_start = gen_start
		self.gen_length = gen_length
		self.gen_end = gen_start + gen_length - 1  # inclusive
		self.token_to_bytes = token_to_bytes

		# Segments sorted by start position
		self._segments: list[Segment] = []

		# Position -> token_id for committed positions
		self.committed: dict[int, int] = {}
  
		# index in segments
		self._end_to_idx: dict[int, int] = {} 
		self._start_to_idx: dict[int, int] = {} 

	@property
	def num_segments(self) -> int:
		return len(self._segments)

	@property
	def num_committed(self) -> int:
		return len(self.committed)

	@property
	def num_masked(self) -> int:
		return self.gen_length - len(self.committed)

	def get_segments(self) -> list[Segment]:
		"""Return a copy of the segment list."""
		return list(self._segments)

	# ------------------------------------------------------------------
	# Segment lookup (linear scan — segments list is small)
	# ------------------------------------------------------------------

	def _find_seg_ending_at(self, pos: int) -> Optional[int]:
		"""Index of segment whose end == pos, or None."""
		for i, seg in enumerate(self._segments):
			if seg.end == pos:
				return i
		return None

	def _find_seg_starting_at(self, pos: int) -> Optional[int]:
		"""Index of segment whose start == pos, or None."""
		for i, seg in enumerate(self._segments):
			if seg.start == pos:
				return i
		return None

	def _insert_segment(self, seg: Segment):
		"""Insert segment maintaining sorted order by start position."""
		idx = len(self._segments)
		for i, existing in enumerate(self._segments):
			if existing.start > seg.start:
				idx = i
				break
		self._segments.insert(idx, seg)

	def _remove_indices(self, *indices: int):
		"""Remove segments at the given indices (in any order)."""
		for i in sorted(indices, reverse=True):
			self._segments.pop(i)

	# ------------------------------------------------------------------
	# Core operation: reveal a token
	# ------------------------------------------------------------------
	def _find_seg_ending_at(self, pos):
		return self._end_to_idx.get(pos)

	# Replace _find_seg_starting_at:
	def _find_seg_starting_at(self, pos):
		return self._start_to_idx.get(pos)

	# Add a helper to rebuild the index dicts (call after any segment mutation):
	def _rebuild_index(self):
		self._end_to_idx = {seg.end: i for i, seg in enumerate(self._segments)}
		self._start_to_idx = {seg.start: i for i, seg in enumerate(self._segments)}

	# Then modify reveal_token to call _rebuild_index() at the end,
	# after the segment list is modified.

	# Full replacement of reveal_token with index maintenance:
	def reveal_token(self, position, token_id):
		assert position not in self.committed
		assert self.gen_start <= position <= self.gen_end

		token_bytes = self.token_to_bytes(token_id)
		self.committed[position] = token_id

		left_idx = self._end_to_idx.get(position - 1)
		right_idx = self._start_to_idx.get(position + 1)

		if left_idx is None and right_idx is None:
			seg = create(position, token_bytes, self.dfa)
			self._insert_segment(seg)
		elif left_idx is not None and right_idx is None:
			left_seg = self._segments[left_idx]
			seg = extend_right(left_seg, position, token_bytes, self.dfa)
			self._segments[left_idx] = seg
		elif left_idx is None and right_idx is not None:
			right_seg = self._segments[right_idx]
			seg = extend_left(right_seg, position, token_bytes, self.dfa)
			self._segments[right_idx] = seg
		else:
			left_seg = self._segments[left_idx]
			right_seg = self._segments[right_idx]
			seg = merge_with_bridge(left_seg, position, token_bytes, right_seg, self.dfa)
			self._remove_indices(left_idx, right_idx)
			self._insert_segment(seg)

		self._rebuild_index()
	# ------------------------------------------------------------------
	# Valid token queries
	# ------------------------------------------------------------------

	def get_valid_tokens(self, position: int, vocab_size: int) -> set[int]:
		"""
		Compute the set of valid token IDs at a masked position.

		A token is valid if it can bridge the left context (exit states
		of the segment ending just before this position) to the right
		context (entry states of the segment starting just after).

		This is the naive O(|V| * |Q|) implementation — loops over the
		full vocabulary. Production use should use a token trie for
		pruning (see trie.py, to be implemented).

		Args:
			position: a currently masked position
			vocab_size: total vocabulary size

		Returns:
			Set of valid token IDs.
		"""
		assert position not in self.committed, (
			f"Position {position} is already committed"
		)

		left_exits = self._left_exit_states(position)
		right_entries = self._right_entry_states(position)

		valid = set()
		for token_id in range(vocab_size):
			token_bytes = self.token_to_bytes(token_id)
			for q in left_exits:
				result = self.dfa.transition_seq(q, token_bytes)
				if result != DEAD and result in right_entries:
					valid.add(token_id)
					break
		return valid

	def is_token_valid(self, position: int, token_id: int) -> bool:
		"""
		Check if a specific token is valid at a masked position.

		Cheaper than get_valid_tokens when you only need to check one.
		"""
		assert position not in self.committed, (
			f"Position {position} is already committed"
		)

		left_exits = self._left_exit_states(position)
		right_entries = self._right_entry_states(position)
		token_bytes = self.token_to_bytes(token_id)

		for q in left_exits:
			result = self.dfa.transition_seq(q, token_bytes)
			if result != DEAD and result in right_entries:
				return True
		return False

	def _left_exit_states(self, position: int) -> frozenset[int]:
		"""
		DFA states that could be active just before `position`.

		- If a segment ends at position-1: its exit states, filtered
		  by the gen_start constraint if the segment starts at gen_start.
		- If position is the start of generation: {dfa.start_state}.
		- Otherwise (gap to the left): all states (over-approximation).
		"""
		idx = self._find_seg_ending_at(position - 1)
		if idx is not None:
			seg = self._segments[idx]
			if seg.start <= self.gen_start:
				# This segment covers gen_start, so its entry must be
				# start_state. Filter pairs accordingly.
				return frozenset(
					x for e, x in seg.pairs if e == self.dfa.start_state
				)
			return seg.exit_states()
		elif position == self.gen_start:
			return frozenset({self.dfa.start_state})
		else:
			return frozenset(range(self.dfa.num_states))

	def _right_entry_states(self, position: int) -> frozenset[int]:
		"""
		DFA states required as entry by the right context at `position`.

		- If a segment starts at position+1: its entry states, filtered
		  by the gen_end constraint if the segment ends at gen_end.
		- If position is the end of generation: accept states.
		- Otherwise (gap to the right): all states (over-approximation).
		"""
		idx = self._find_seg_starting_at(position + 1)
		if idx is not None:
			seg = self._segments[idx]
			if seg.end >= self.gen_end:
				# This segment covers gen_end, so its exit must be an
				# accept state. Filter pairs accordingly.
				return frozenset(
					e for e, x in seg.pairs if x in self.dfa.accept_states
				)
			return seg.entry_states()
		elif position == self.gen_end:
			return self.dfa.accept_states
		else:
			return frozenset(range(self.dfa.num_states))

	# ------------------------------------------------------------------
	# Global consistency
	# ------------------------------------------------------------------

	def has_empty_segment(self) -> bool:
		"""
		Check if any segment has an empty transition relation.

		An empty relation means the committed tokens in that segment
		are irreconcilable — no DFA path can traverse them. This is
		an irrecoverable error.
		"""
		return any(len(seg.pairs) == 0 for seg in self._segments)

	def is_valid_complete(self) -> bool:
		"""
		Check if the generation is fully committed and valid.

		True iff all positions are committed, there is exactly one
		segment, and it has a (start_state, accept_state) pair.
		"""
		if self.num_masked > 0:
			return False
		if len(self._segments) != 1:
			return False
		seg = self._segments[0]
		return any(
			e == self.dfa.start_state and x in self.dfa.accept_states
			for e, x in seg.pairs
		)

	# ------------------------------------------------------------------
	# Initialization helpers
	# ------------------------------------------------------------------

	def init_with_prompt(self, prompt_bytes: list[bytes]):
		"""
		Initialize with a committed prompt preceding the generation region.

		The prompt occupies positions gen_start - len(prompt_bytes) .. gen_start - 1.
		It becomes a single segment providing left context for the
		generation region.

		Args:
			prompt_bytes: list of byte sequences for each prompt token,
						  in left-to-right order.
		"""
		if not prompt_bytes:
			return

		prompt_start = self.gen_start - len(prompt_bytes)
		seg = create(prompt_start, prompt_bytes[0], self.dfa)
		for i, tok_bytes in enumerate(prompt_bytes[1:], start=1):
			seg = extend_right(seg, prompt_start + i, tok_bytes, self.dfa)

		self._insert_segment(seg)

	def __repr__(self) -> str:
		segs = ", ".join(
			f"[{s.start}-{s.end}]({len(s.pairs)}p)"
			for s in self._segments
		)
		return (
			f"SegmentManager(committed={self.num_committed}/{self.gen_length}, "
			f"segments=[{segs}])"
		)