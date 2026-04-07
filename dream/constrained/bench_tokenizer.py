"""
Benchmark: trie-based valid-set computation on Dream's real tokenizer.

Usage:
    python bench_tokenizer.py
"""

import time
import transformers

from dfa import build_json_dfa, DEAD
from segments import create, extend_right
from manager import SegmentManager
from trie import TokenTrie


def load_tokenizer():
    print("Loading Dream tokenizer...")
    tok = transformers.AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True
    )
    print(f"  Vocab size: {tok.vocab_size}")
    return tok


def build_token_to_bytes(tok) -> dict[int, bytes]:
    """
    Extract token_id -> byte sequence from tokenizer.

    Uses the tokenizer's byte_decoder directly: each character in the
    BPE vocab string maps to a byte via byte_decoder (the GPT2-style
    bytes_to_unicode mapping). This is lossless — no decode/encode
    round-trip needed.
    """
    byte_decoder = tok.byte_decoder
    t2b = {}
    num_special = 0

    for token_id in range(tok.vocab_size):
        token_str = tok.convert_ids_to_tokens(token_id)
        if token_str is None:
            t2b[token_id] = b""
            num_special += 1
            continue
        try:
            t2b[token_id] = bytes(byte_decoder[c] for c in token_str)
        except KeyError:
            # Special tokens (e.g. <|endoftext|>) have chars not in byte_decoder
            t2b[token_id] = b""
            num_special += 1

    lengths = [len(v) for v in t2b.values() if len(v) > 0]
    print(f"  Tokens mapped: {len(t2b)} ({num_special} special/empty)")
    if lengths:
        print(f"  Byte lengths: min={min(lengths)}, max={max(lengths)}, "
              f"median={sorted(lengths)[len(lengths)//2]}")

    # Sanity check: round-trip
    for test_str in ['{"key": 42}', 'Hello world', '{"a": [1, 2]}']:
        ids = tok.encode(test_str, add_special_tokens=False)
        reconstructed = b''.join(t2b[tid] for tid in ids)
        assert reconstructed == test_str.encode('utf-8'), (
            f"Round-trip failed for {test_str!r}:\n"
            f"  expected:      {test_str.encode()!r}\n"
            f"  reconstructed: {reconstructed!r}"
        )
    print("  Round-trip sanity check: OK")

    return t2b


def brute_force_valid(left_exits, right_entries, dfa, token_to_bytes):
    """Reference implementation."""
    valid = set()
    for tid, tbytes in token_to_bytes.items():
        if len(tbytes) == 0:
            continue
        for q in left_exits:
            result = dfa.transition_seq(q, tbytes)
            if result != DEAD:
                if right_entries is None or result in right_entries:
                    valid.add(tid)
                    break
    return valid


def main():
    tok = load_tokenizer()
    t2b = build_token_to_bytes(tok)

    print("\nBuilding JSON DFA (max_depth=6)...")
    t0 = time.perf_counter()
    dfa = build_json_dfa(max_depth=6)
    dt = time.perf_counter() - t0
    print(f"  States: {dfa.num_states}, built in {dt:.3f}s")

    print("\nBuilding token trie...")
    t0 = time.perf_counter()
    trie = TokenTrie(t2b)
    dt = time.perf_counter() - t0
    s = trie.stats()
    print(f"  Nodes: {s['num_nodes']}, Depth: {s['max_depth']}, "
          f"Tokens: {s['num_tokens']}, built in {dt:.3f}s")

    # --- Constraint scenarios ---
    scenarios = []

    # 1. Start of generation
    start_exits = frozenset({dfa.start_state})
    scenarios.append(("start, no right", start_exits, None))
    scenarios.append(("start, accept right", start_exits, dfa.accept_states))

    # 2. After '{"key": '
    state = dfa.start_state
    for b in b'{"key": ':
        state = dfa.transition(state, b)
        assert state != DEAD
    scenarios.append(("after '{\"key\": ', no right", frozenset({state}), None))

    # 3. Tight: between '{"key"' and '42}'
    left_state = dfa.start_state
    for b in b'{"key"':
        left_state = dfa.transition(left_state, b)
    left_exits_tight = frozenset({left_state})

    right_entries_tight = set()
    for q in range(dfa.num_states):
        result = dfa.transition_seq(q, b'42}')
        if result != DEAD and dfa.is_accept(result):
            right_entries_tight.add(q)
    scenarios.append((
        "between '{\"key\"' and '42}' (tight)",
        left_exits_tight, frozenset(right_entries_tight),
    ))

    # 4. After a real tokenized prefix
    prefix_ids = tok.encode('{"name": "John", "age": ', add_special_tokens=False)
    state = dfa.start_state
    for tid in prefix_ids:
        for b in t2b[tid]:
            state = dfa.transition(state, b)
            if state == DEAD:
                break
        if state == DEAD:
            break
    if state != DEAD:
        scenarios.append((
            f"after real prefix ({len(prefix_ids)} tokens), no right",
            frozenset({state}), None,
        ))

    # 5. Deep nesting
    seg_state = dfa.start_state
    for b in b'{"items": [{"id": ':
        seg_state = dfa.transition(seg_state, b)
    if seg_state != DEAD:
        scenarios.append((
            "deep nesting exit, no right",
            frozenset({seg_state}), None,
        ))

    # --- Correctness ---
    print("\n=== Correctness (trie vs brute-force) ===")
    for label, left_exits, right_entries in scenarios:
        trie_result = trie.compute_valid_set(left_exits, right_entries, dfa)
        bf_result = brute_force_valid(left_exits, right_entries, dfa, t2b)
        match = trie_result == bf_result
        status = "PASS" if match else "FAIL"
        print(f"  {status}: {label} -> {len(trie_result)} valid")
        if not match:
            only_trie = trie_result - bf_result
            only_bf = bf_result - trie_result
            print(f"    trie-only: {len(only_trie)}, bf-only: {len(only_bf)}")

    # --- Speed ---
    print("\n=== Speed benchmark ===")
    N_WARMUP = 3
    N_RUNS = 20

    for label, left_exits, right_entries in scenarios:
        for _ in range(N_WARMUP):
            trie.compute_valid_set(left_exits, right_entries, dfa)

        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            trie.compute_valid_set(left_exits, right_entries, dfa)
        trie_ms = (time.perf_counter() - t0) / N_RUNS * 1000

        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            brute_force_valid(left_exits, right_entries, dfa, t2b)
        bf_ms = (time.perf_counter() - t0) / N_RUNS * 1000

        speedup = bf_ms / trie_ms if trie_ms > 0 else float('inf')
        print(f"  {label}:")
        print(f"    trie: {trie_ms:8.2f} ms  |  brute: {bf_ms:8.2f} ms  |  {speedup:.1f}x")

    # --- Integration: mask each position ---
    print("\n=== SegmentManager + Trie integration ===")
    json_target = '{"name": "test"}'
    target_ids = tok.encode(json_target, add_special_tokens=False)
    print(f"  Target: {json_target}")
    print(f"  Tokens: {[tok.decode([t]) for t in target_ids]}")

    n = len(target_ids)
    for mask_pos in range(n):
        mgr = SegmentManager(dfa, gen_start=0, gen_length=n,
                             token_to_bytes=lambda t: t2b.get(t, b''))
        for i in range(n):
            if i != mask_pos:
                mgr.reveal_token(i, target_ids[i])

        left_exits = mgr._left_exit_states(mask_pos)
        right_entries = mgr._right_entry_states(mask_pos)
        trie_valid = trie.compute_valid_set(left_exits, right_entries, dfa)

        original_tok = target_ids[mask_pos]
        original_in_valid = original_tok in trie_valid
        status = "OK" if original_in_valid else "MISS"
        tok_str = tok.decode([original_tok])
        print(f"  pos {mask_pos} ({tok_str!r:>8s}): {len(trie_valid):5d} valid, "
              f"original in set: {status}")

        # Debug MISS
        if not original_in_valid:
            target_bytes = t2b[original_tok]
            print(f"    target bytes: {target_bytes!r}")
            print(f"    left_exits ({len(left_exits)}): {sorted(left_exits)[:5]}...")
            print(f"    right_entries ({len(right_entries)}): {sorted(right_entries)[:5]}...")
            # Check target token transitions
            for q in sorted(left_exits)[:3]:
                result = dfa.transition_seq(q, target_bytes)
                in_right = result in right_entries if result != DEAD else "DEAD"
                print(f"    from {q}: -> {result}, in_right={in_right}")
            # Compare with raw (unfiltered) right entries
            right_idx = mgr._find_seg_starting_at(mask_pos + 1)
            if right_idx is not None:
                raw = mgr._segments[right_idx].entry_states()
                print(f"    raw right entries: {len(raw)} states")
                for q in sorted(left_exits)[:3]:
                    result = dfa.transition_seq(q, target_bytes)
                    if result != DEAD:
                        print(f"    from {q}: -> {result}, in RAW={result in raw}")


if __name__ == "__main__":
    main()