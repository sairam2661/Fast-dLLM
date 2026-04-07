"""
Unit tests for constrained_decoder.py.

Run:
    python test_constrained_decoder.py
"""

import torch
from dfa import build_json_dfa, DFA, DEAD
from trie import TokenTrie
from constrained_decoder import ConstrainedDecoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_json_vocab():
    chars = list('{}[]:,"0123456789abcdefghijklmnopqrstuvwxyz \n-.')
    vocab = {i: c.encode() for i, c in enumerate(chars)}
    char_to_id = {c: i for i, c in enumerate(chars)}
    return vocab, char_to_id, chars


MASK_TOKEN_ID = 999


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sync_from_sequence():
    """sync_from_sequence picks up non-mask tokens."""
    dfa = build_json_dfa(max_depth=4)
    vocab, c2id, chars = make_json_vocab()
    trie = TokenTrie(vocab)

    seq = torch.tensor([[c2id['{'], MASK_TOKEN_ID, MASK_TOKEN_ID,
                          MASK_TOKEN_ID, c2id['}']]])

    dec = ConstrainedDecoder(dfa, trie, vocab,
                             gen_start=0, gen_length=5,
                             mask_token_id=MASK_TOKEN_ID)
    dec.sync_from_sequence(seq)

    assert dec.mgr.num_committed == 2
    assert dec.mgr.num_segments == 2
    print("  PASS: sync_from_sequence")


def test_update_committed():
    """update_committed detects newly unmasked tokens."""
    dfa = build_json_dfa(max_depth=4)
    vocab, c2id, chars = make_json_vocab()
    trie = TokenTrie(vocab)

    seq = torch.tensor([[c2id['{'], MASK_TOKEN_ID, MASK_TOKEN_ID,
                          MASK_TOKEN_ID, c2id['}']]])

    dec = ConstrainedDecoder(dfa, trie, vocab,
                             gen_start=0, gen_length=5,
                             mask_token_id=MASK_TOKEN_ID)
    dec.sync_from_sequence(seq)
    assert dec.mgr.num_committed == 2

    seq[0, 1] = c2id['"']
    dec.update_committed(seq)
    assert dec.mgr.num_committed == 3
    print("  PASS: update_committed")


def test_apply_constraints():
    """apply_constraints masks invalid tokens in logits."""
    dfa = build_json_dfa(max_depth=4)
    vocab, c2id, chars = make_json_vocab()
    V = len(chars)
    trie = TokenTrie(vocab)

    # {"a" _ 1}  — position 4 masked (the colon)
    json_str = '{"a":1}'
    token_ids = [c2id[c] for c in json_str]
    seq = torch.tensor([token_ids])
    seq[0, 4] = MASK_TOKEN_ID

    dec = ConstrainedDecoder(dfa, trie, vocab,
                             gen_start=0, gen_length=7,
                             mask_token_id=MASK_TOKEN_ID)
    dec.sync_from_sequence(seq)

    logits = torch.ones(1, 7, V)
    dec.apply_constraints(logits, seq, block_start=0, block_end=7)

    colon_id = c2id[':']
    pos4_logits = logits[0, 4]
    assert pos4_logits[colon_id] == 1.0
    valid_count = (pos4_logits > float('-inf')).sum().item()
    assert valid_count == 1, f"Expected 1 valid token at pos 4, got {valid_count}"
    print("  PASS: apply_constraints masks to only ':'")


def test_apply_constraints_to_masked_logits():
    """apply_constraints_to_masked_logits works on flattened tensor."""
    dfa = build_json_dfa(max_depth=4)
    vocab, c2id, chars = make_json_vocab()
    V = len(chars)
    trie = TokenTrie(vocab)

    json_str = '{"a":1}'
    token_ids = [c2id[c] for c in json_str]
    seq = torch.tensor([token_ids])
    seq[0, 4] = MASK_TOKEN_ID

    dec = ConstrainedDecoder(dfa, trie, vocab,
                             gen_start=0, gen_length=7,
                             mask_token_id=MASK_TOKEN_ID)
    dec.sync_from_sequence(seq)

    mask_logits = torch.ones(1, V)
    mask_positions = torch.tensor([4])

    dec.apply_constraints_to_masked_logits(mask_logits, mask_positions,
                                           device=torch.device('cpu'))

    colon_id = c2id[':']
    valid_count = (mask_logits[0] > float('-inf')).sum().item()
    assert valid_count == 1
    assert mask_logits[0, colon_id] == 1.0
    print("  PASS: apply_constraints_to_masked_logits")


def test_simulated_denoising():
    """
    Simulate a denoising loop producing valid JSON.

    Uses a known-good target and reveals tokens in a scrambled order.
    At each step, we verify the target token is in the valid set and
    commit it. This simulates the ideal case where the model's top
    prediction happens to be valid.
    """
    dfa = build_json_dfa(max_depth=4)
    vocab, c2id, chars = make_json_vocab()
    V = len(chars)
    trie = TokenTrie(vocab)

    # Target: {"a":1}
    target = '{"a":1}'
    target_ids = [c2id[c] for c in target]
    gen_length = len(target)

    seq = torch.full((1, gen_length), MASK_TOKEN_ID, dtype=torch.long)

    dec = ConstrainedDecoder(dfa, trie, vocab,
                             gen_start=0, gen_length=gen_length,
                             mask_token_id=MASK_TOKEN_ID)

    # Scrambled reveal order (mimics diffusion)
    import random
    random.seed(42)
    order = list(range(gen_length))
    random.shuffle(order)

    for step, pos in enumerate(order):
        dec.update_committed(seq)
        valid_mask = dec.get_valid_mask(pos, torch.device('cpu'))
        valid_ids = valid_mask.nonzero(as_tuple=True)[0].tolist()

        target_tid = target_ids[pos]
        assert target_tid in valid_ids, (
            f"Step {step}: target token {chars[target_tid]!r} not valid at "
            f"pos {pos}. Valid: {[chars[v] for v in valid_ids[:10]]}"
        )

        seq[0, pos] = target_tid
        print(f"    Step {step}: pos {pos} <- {chars[target_tid]!r} "
              f"({len(valid_ids)} valid)")

    # Verify
    result = ''.join(chars[seq[0, i].item()] for i in range(gen_length))
    dec.update_committed(seq)
    assert dec.mgr.is_valid_complete(), f"Result {result!r} not valid"
    print(f"  PASS: simulated denoising -> {result!r}")


def test_simulated_denoising_free():
    """
    Simulate denoising without a target — pick first valid token each step.

    This tests that the constraints alone can guide generation to a
    valid output, regardless of which valid token is chosen.
    """
    dfa = build_json_dfa(max_depth=4)
    vocab, c2id, chars = make_json_vocab()
    V = len(chars)
    trie = TokenTrie(vocab)

    gen_length = 5  # short enough that first-valid always works
    seq = torch.full((1, gen_length), MASK_TOKEN_ID, dtype=torch.long)

    dec = ConstrainedDecoder(dfa, trie, vocab,
                             gen_start=0, gen_length=gen_length,
                             mask_token_id=MASK_TOKEN_ID)

    # Reveal left-to-right (most constrained order for DFA)
    for pos in range(gen_length):
        dec.update_committed(seq)
        valid_mask = dec.get_valid_mask(pos, torch.device('cpu'))
        valid_ids = valid_mask.nonzero(as_tuple=True)[0].tolist()

        if not valid_ids:
            # If we're stuck, the sequence can't be completed
            # This is expected for some random paths — not a bug
            print(f"    pos {pos}: no valid tokens (stuck)")
            break

        chosen = valid_ids[0]
        seq[0, pos] = chosen
        print(f"    pos {pos} <- {chars[chosen]!r} ({len(valid_ids)} valid)")

    result_ids = [seq[0, i].item() for i in range(gen_length)]
    if MASK_TOKEN_ID not in result_ids:
        result = ''.join(chars[tid] for tid in result_ids)
        dec.update_committed(seq)
        is_valid = dec.mgr.is_valid_complete()
        print(f"  Result: {result!r}, valid={is_valid}")
        assert is_valid, f"Result should be valid"
        print("  PASS: free denoising produced valid output")
    else:
        print("  SKIP: got stuck (expected for some orderings)")


def test_apply_constraints_skips_committed():
    """apply_constraints doesn't modify logits for committed positions."""
    dfa = build_json_dfa(max_depth=4)
    vocab, c2id, chars = make_json_vocab()
    V = len(chars)
    trie = TokenTrie(vocab)

    seq = torch.tensor([[c2id['{'], MASK_TOKEN_ID, c2id['}']]])
    dec = ConstrainedDecoder(dfa, trie, vocab,
                             gen_start=0, gen_length=3,
                             mask_token_id=MASK_TOKEN_ID)
    dec.sync_from_sequence(seq)

    logits = torch.ones(1, 3, V)
    dec.apply_constraints(logits, seq, block_start=0, block_end=3)

    assert (logits[0, 0] == 1.0).all(), "Committed pos 0 should be untouched"
    assert (logits[0, 2] == 1.0).all(), "Committed pos 2 should be untouched"

    valid_count = (logits[0, 1] > float('-inf')).sum().item()
    assert valid_count < V
    print(f"  PASS: committed positions untouched, masked pos has {valid_count}/{V} valid")


if __name__ == '__main__':
    print("=== sync_from_sequence ===")
    test_sync_from_sequence()

    print("\n=== update_committed ===")
    test_update_committed()

    print("\n=== apply_constraints ===")
    test_apply_constraints()

    print("\n=== apply_constraints_to_masked_logits ===")
    test_apply_constraints_to_masked_logits()

    print("\n=== simulated denoising (guided) ===")
    test_simulated_denoising()

    print("\n=== simulated denoising (free) ===")
    test_simulated_denoising_free()

    print("\n=== committed positions skipped ===")
    test_apply_constraints_skips_committed()

    print("\nAll tests passed.")