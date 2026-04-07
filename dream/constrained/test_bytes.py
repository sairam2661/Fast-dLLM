"""
Debug: verify that build_token_to_bytes produces byte sequences that
match what the tokenizer actually encodes.

The key test: for a string s, does concatenating t2b[tid] for each
tid in tok.encode(s) give back s.encode('utf-8')?

Run:
    python debug_bytes.py
"""

import transformers


def build_token_to_bytes_v1(tok):
    """Original: decode then re-encode (lossy for some tokens)."""
    t2b = {}
    for token_id in range(tok.vocab_size):
        try:
            s = tok.decode([token_id])
            t2b[token_id] = s.encode("utf-8")
        except Exception:
            t2b[token_id] = b""
    return t2b


def build_token_to_bytes_v2(tok):
    """Current: convert_ids_to_tokens with <0xHH> handling."""
    t2b = {}
    for token_id in range(tok.vocab_size):
        token_str = tok.convert_ids_to_tokens(token_id)
        if token_str is None:
            t2b[token_id] = b""
            continue
        if "<0x" in token_str:
            result = bytearray()
            i = 0
            while i < len(token_str):
                if (token_str[i:i+3] == "<0x"
                        and i + 5 <= len(token_str)
                        and token_str[i+4:i+5] == ">"):
                    try:
                        result.append(int(token_str[i+3:i+5], 16))
                        i += 5
                        continue
                    except ValueError:
                        pass
                result.extend(token_str[i].encode("utf-8"))
                i += 1
            t2b[token_id] = bytes(result)
        else:
            t2b[token_id] = token_str.encode("utf-8")
    return t2b


def build_token_to_bytes_v3(tok):
    """
    New approach: use the tokenizer's own convert_tokens_to_string or
    decode for each token, but also try to get raw bytes from the
    internal vocabulary.
    """
    t2b = {}

    # Try to get the raw vocabulary
    vocab = tok.get_vocab()  # {string: id}
    id_to_str = {v: k for k, v in vocab.items()}

    for token_id in range(tok.vocab_size):
        # Method: decode single token, encode back to utf-8
        # But use add_special_tokens=False and clean_up_tokenization_spaces=False
        try:
            s = tok.decode([token_id], clean_up_tokenization_spaces=False)
            t2b[token_id] = s.encode("utf-8")
        except Exception:
            t2b[token_id] = b""

    return t2b


def main():
    tok = transformers.AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True
    )

    test_strings = [
        '{"name": "test"}',
        '{"key": 42}',
        '{"a": [1, 2, 3]}',
        '{"nested": {"inner": true}}',
        'Hello world',
        '{"emoji": "\\u2603"}',
    ]

    for version, builder in [("v1 (decode)", build_token_to_bytes_v1),
                              ("v2 (convert_ids)", build_token_to_bytes_v2),
                              ("v3 (decode clean)", build_token_to_bytes_v3)]:
        print(f"\n=== {version} ===")
        t2b = builder(tok)

        for s in test_strings:
            target = s.encode("utf-8")
            token_ids = tok.encode(s, add_special_tokens=False)
            reconstructed = b''.join(t2b[tid] for tid in token_ids)
            match = reconstructed == target
            status = "OK" if match else "MISMATCH"
            print(f"  {status}: {s!r}")
            if not match:
                print(f"    target:       {target!r}")
                print(f"    reconstructed: {reconstructed!r}")
                # Show per-token breakdown
                for tid in token_ids:
                    tok_str = tok.convert_ids_to_tokens(tid)
                    decoded = tok.decode([tid])
                    print(f"    id={tid}: convert={tok_str!r}, "
                          f"decode={decoded!r}, t2b={t2b[tid]!r}")

    # Also check: what does the Qwen tokenizer use internally?
    print("\n=== Tokenizer internals ===")
    print(f"Type: {type(tok).__name__}")
    print(f"Has sp_model: {hasattr(tok, 'sp_model')}")
    print(f"Has byte_encoder: {hasattr(tok, 'byte_encoder')}")
    print(f"Has byte_decoder: {hasattr(tok, 'byte_decoder')}")

    # Check a few tokens
    for tid in [0, 1, 2, 100, 1000]:
        raw = tok.convert_ids_to_tokens(tid)
        dec = tok.decode([tid])
        print(f"  id={tid}: convert={raw!r}, decode={dec!r}")


if __name__ == "__main__":
    main()