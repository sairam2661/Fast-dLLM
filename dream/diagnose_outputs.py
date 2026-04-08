#!/usr/bin/env python3
"""
Diagnostic: Compare constrained vs unconstrained outputs side-by-side.
Analyzes failing instances to understand what's going wrong.

Usage:
  python diagnose_outputs.py \
      --model_path Dream-org/Dream-v0-Instruct-7B \
      --instances 3 \
      --steps 32
"""

import argparse
import json
import os
import time
import types

import torch
import transformers
from datasets import load_dataset

from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from constrained.constrained_decoder import build_constrained_decoder, ConstrainedDecoder


def load_model_with_mixin(model_path, mixin_module, device="cuda"):
    model = DreamModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    model.diffusion_generate = types.MethodType(mixin_module.DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(mixin_module.DreamGenerationMixin._sample, model)
    return model


def format_prompt(schema_str, input_text, tokenizer):
    system_msg = (
        "You are a helpful assistant that answers in JSON. "
        "Here is the JSON schema you must adhere to:\n"
        f"<schema>\n{schema_str}\n</schema>"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": input_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    return inputs


def generate_one(model, tokenizer, inputs, steps, temperature, device,
                 constrained_decoder=None):
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            constrained_decoder=constrained_decoder,
        )

    full_ids = output.sequences[0].tolist()
    gen_ids = full_ids[prompt_len:]
    gen_text = tokenizer.decode(gen_ids).split(tokenizer.eos_token)[0]

    # Also get raw token strings for detailed analysis
    raw_tokens = []
    for tid in gen_ids:
        tok_str = tokenizer.convert_ids_to_tokens(tid)
        raw_tokens.append((tid, tok_str))

    return {
        "gen_text": gen_text,
        "gen_ids": gen_ids,
        "raw_tokens": raw_tokens,
        "prompt_len": prompt_len,
    }


def find_json_region(text):
    """Find the first { and last matching } in the text."""
    start = text.find('{')
    if start == -1:
        return None, None, "no '{' found"

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return start, i + 1, None

    return start, None, f"unbalanced braces (depth={depth} at end)"


def diagnose_json(text):
    """Try to parse as JSON and give detailed error info."""
    result = {
        "text_length": len(text),
        "has_open_brace": '{' in text,
        "has_close_brace": '}' in text,
    }

    start, end, region_err = find_json_region(text)
    result["json_region_start"] = start
    result["json_region_end"] = end
    result["json_region_error"] = region_err

    if start is not None and end is not None:
        candidate = text[start:end]
        result["candidate_length"] = len(candidate)
        try:
            parsed = json.loads(candidate)
            result["valid_json"] = True
            result["parsed_keys"] = list(parsed.keys()) if isinstance(parsed, dict) else f"[array len={len(parsed)}]"
        except json.JSONDecodeError as e:
            result["valid_json"] = False
            result["json_error"] = str(e)
            result["error_pos"] = e.pos
            result["error_lineno"] = e.lineno
            result["error_colno"] = e.colno
            # Show context around the error
            err_start = max(0, e.pos - 30)
            err_end = min(len(candidate), e.pos + 30)
            result["error_context"] = (
                f"...{candidate[err_start:e.pos]}<<<HERE>>>{candidate[e.pos:err_end]}..."
            )
    else:
        result["valid_json"] = False
        result["candidate_length"] = 0

    return result


def show_token_diff(tokens_a, tokens_b, label_a="Unconstrained", label_b="Constrained"):
    """Show where two token sequences differ."""
    max_len = max(len(tokens_a), len(tokens_b))
    diffs = []
    for i in range(max_len):
        tid_a = tokens_a[i][0] if i < len(tokens_a) else None
        tid_b = tokens_b[i][0] if i < len(tokens_b) else None
        if tid_a != tid_b:
            tok_a = tokens_a[i][1] if i < len(tokens_a) else "<missing>"
            tok_b = tokens_b[i][1] if i < len(tokens_b) else "<missing>"
            diffs.append((i, tid_a, tok_a, tid_b, tok_b))
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--instances", type=int, nargs="+", default=[3])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load dataset
    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    print(f"Dataset has {len(ds)} instances")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # Load model with UNCONSTRAINED mixin
    print("Loading model (unconstrained)...", flush=True)
    from model.generation_utils_block import DreamGenerationMixin as UnconstrainedMixin
    model = DreamModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(args.device)

    # Build constrained decoder
    print("Building constrained decoder...", flush=True)
    cd = None
    dfa, trie, t2b = build_constrained_decoder(tokenizer, max_depth=6)
    cd = ConstrainedDecoder(
        dfa=dfa, trie=trie, token_to_bytes=t2b,
        gen_start=0, gen_length=1,
        mask_token_id=tokenizer.convert_tokens_to_ids("<|mask|>"),
    )
    print(f"Precomputing state masks...", flush=True)
    cd.precompute_state_masks(args.device)

    for inst_idx in args.instances:
        row = ds[inst_idx]
        print(f"\n{'='*80}")
        print(f"Instance {inst_idx}: {row['instance_id']}")
        print(f"{'='*80}")

        schema = json.loads(row["schema"])
        print(f"\nSchema keys: {list(schema.get('properties', {}).keys())}")
        print(f"Required: {schema.get('required', [])}")
        print(f"Input: {row['input'][:200]}...")

        inputs = format_prompt(row["schema"], row["input"], tokenizer)
        prompt_len = inputs.input_ids.shape[1]
        print(f"Prompt length: {prompt_len} tokens")

        # --- Run UNCONSTRAINED ---
        print(f"\n--- UNCONSTRAINED (seed={args.seed}) ---")
        torch.manual_seed(args.seed)
        model.diffusion_generate = types.MethodType(UnconstrainedMixin.diffusion_generate, model)
        model._sample = types.MethodType(UnconstrainedMixin._sample, model)
        result_unc = generate_one(model, tokenizer, inputs, args.steps,
                                   args.temperature, args.device)

        print(f"Generated text ({len(result_unc['gen_text'])} chars):")
        print(f"  {repr(result_unc['gen_text'][:])}")
        diag_unc = diagnose_json(result_unc["gen_text"])
        print(f"JSON diagnosis: {json.dumps({k:v for k,v in diag_unc.items() if k != 'parsed_keys'}, indent=2)}")
        if 'parsed_keys' in diag_unc:
            print(f"  Parsed keys: {diag_unc['parsed_keys']}")

        # --- Run CONSTRAINED ---
        print(f"\n--- CONSTRAINED (seed={args.seed}) ---")
        torch.manual_seed(args.seed)
        from model.generation_utils_block_constrained import DreamGenerationMixin as ConstrainedMixin
        model.diffusion_generate = types.MethodType(ConstrainedMixin.diffusion_generate, model)
        model._sample = types.MethodType(ConstrainedMixin._sample, model)
        result_con = generate_one(model, tokenizer, inputs, args.steps,
                                   args.temperature, args.device,
                                   constrained_decoder=cd)

        print(f"Generated text ({len(result_con['gen_text'])} chars):")
        print(f"  {repr(result_con['gen_text'][:])}")
        diag_con = diagnose_json(result_con["gen_text"])
        print(f"JSON diagnosis: {json.dumps({k:v for k,v in diag_con.items() if k != 'parsed_keys'}, indent=2)}")
        if 'parsed_keys' in diag_con:
            print(f"  Parsed keys: {diag_con['parsed_keys']}")

        # --- DIFF ---
        print(f"\n--- TOKEN DIFF ---")
        diffs = show_token_diff(result_unc["raw_tokens"], result_con["raw_tokens"])
        if not diffs:
            print("  Outputs are IDENTICAL (no token differences)")
        else:
            print(f"  {len(diffs)} positions differ out of {max(len(result_unc['raw_tokens']), len(result_con['raw_tokens']))}")
            for pos, tid_a, tok_a, tid_b, tok_b in diffs[:30]:
                marker = ""
                if tid_a is not None and t2b.get(tid_a, b'') == b'':
                    marker += " [unc=empty-bytes]"
                if tid_b is not None and t2b.get(tid_b, b'') == b'':
                    marker += " [con=empty-bytes]"
                print(f"  pos {pos:3d}: unc=({tid_a}, {repr(tok_a)}) vs con=({tid_b}, {repr(tok_b)}){marker}")

        # --- Reference output ---
        print(f"\n--- REFERENCE ---")
        ref = row["output"]
        try:
            ref_parsed = json.loads(ref)
            print(f"  Keys: {list(ref_parsed.keys()) if isinstance(ref_parsed, dict) else 'array'}")
            print(f"  {repr(ref[:300])}")
        except:
            print(f"  {repr(ref[:300])}")

        # --- Mask token analysis ---
        mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
        unc_masks = sum(1 for tid, _ in result_unc["raw_tokens"] if tid == mask_id)
        con_masks = sum(1 for tid, _ in result_con["raw_tokens"] if tid == mask_id)
        unc_empty = sum(1 for tid, _ in result_unc["raw_tokens"] if t2b.get(tid, b'') == b'' and tid != mask_id)
        con_empty = sum(1 for tid, _ in result_con["raw_tokens"] if t2b.get(tid, b'') == b'' and tid != mask_id)
        unc_oov = sum(1 for tid, _ in result_unc["raw_tokens"] if tid >= len(t2b))
        con_oov = sum(1 for tid, _ in result_con["raw_tokens"] if tid >= len(t2b))
        print(f"\n--- SPECIAL TOKEN COUNTS ---")
        print(f"  Unconstrained: {unc_masks} mask, {unc_empty} empty-byte, {unc_oov} out-of-vocab")
        print(f"  Constrained:   {con_masks} mask, {con_empty} empty-byte, {con_oov} out-of-vocab")


if __name__ == "__main__":
    main()