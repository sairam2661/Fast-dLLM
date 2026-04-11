#!/usr/bin/env python3
"""
Evaluation: Dream-Instruct with LR grammar-constrained decoding
on eth-sri/json-mode-eval-extended.

Drop-in replacement for eval_json_constrained.py that uses the
per-schema scanner+LR backend instead of the global JSON DFA.

Key differences from the DFA version:
  - build_grammar_decoder_lr(tokenizer, schema, device) builds a fresh
    CompositeAutomaton per schema and precomputes its state masks.
  - Schema-string cache avoids rebuilding for repeated schemas.
  - The decoder is reset (gen_start/gen_length) by _sample as before;
    the decoder object itself is re-created per schema (or reused from cache).

Usage:
  python eval/eval_json_constrained_lr.py \
      --model_path Dream-org/Dream-v0-Instruct-7B \
      --steps 128 \
      --num_instances 50 \
      --constrained
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import jsonschema
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from constrained.constrained_decoder import ConstrainedDecoder
from constrained.manager import CompositeAutomaton
from constrained.trie import TokenTrie
from constrained.cfg import BoundedLRAutomaton
from constrained.scanner import JsonScanner
from constrained.schema_compiler import compile_schema, recommended_depth


# ---------------------------------------------------------------------------
# Schema complexity check (mirrors eval_quick.py)
# ---------------------------------------------------------------------------

def schema_stats(schema):
    """(nesting_depth, n_optional_max)"""
    if not isinstance(schema, dict):
        return 0, 0
    typ = schema.get("type")
    if typ == "object":
        props = schema.get("properties", {})
        req = set(schema.get("required", []))
        n_opt = len(props) - len(req)
        sub_d, sub_o = 0, 0
        for v in props.values():
            sd, so = schema_stats(v)
            sub_d = max(sub_d, sd)
            sub_o = max(sub_o, so)
        return 1 + sub_d, max(n_opt, sub_o)
    if typ == "array":
        return schema_stats(schema.get("items", {}))
    for k in ("anyOf", "oneOf"):
        if k in schema:
            results = [schema_stats(s) for s in schema[k]]
            if results:
                return max(r[0] for r in results), max(r[1] for r in results)
    return 0, 0


# ---------------------------------------------------------------------------
# Per-schema decoder factory with cache
# ---------------------------------------------------------------------------

_decoder_cache: dict[str, ConstrainedDecoder] = {}
_t2b_cache: dict[str, dict[int, bytes]] = {}  # tokenizer vocab, built once


def _build_t2b(tokenizer) -> dict[int, bytes]:
    """Build token-to-bytes mapping. Done once per tokenizer."""
    global _t2b_cache
    key = getattr(tokenizer, 'name_or_path', str(type(tokenizer)))
    if key in _t2b_cache:
        return _t2b_cache[key]
    byte_decoder = tokenizer.byte_decoder
    t2b: dict[int, bytes] = {}
    for token_id in range(tokenizer.vocab_size):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        if token_str is None:
            t2b[token_id] = b""
            continue
        try:
            t2b[token_id] = bytes(byte_decoder[c] for c in token_str)
        except KeyError:
            t2b[token_id] = b""
    _t2b_cache[key] = t2b
    print(f"  [t2b] Built token→bytes for {len(t2b)} tokens", flush=True)
    return t2b


def build_grammar_decoder_lr(
    tokenizer,
    schema: dict,
    device: torch.device,
    t2b: dict[int, bytes],
    trie: TokenTrie,
) -> ConstrainedDecoder:
    """
    Build a ConstrainedDecoder for a single JSON Schema.

    Uses a schema-string cache so repeated schemas are free.
    The trie is shared across all schemas (it only depends on the vocabulary).

    Args:
        tokenizer: HuggingFace tokenizer.
        schema: JSON Schema dict.
        device: torch device for precomputed masks.
        t2b: pre-built token→bytes mapping (from _build_t2b).
        trie: pre-built TokenTrie (from TokenTrie(t2b)).

    Returns:
        ConstrainedDecoder with precomputed state masks, ready to use.
    """
    schema_str = json.dumps(schema, sort_keys=True)
    if schema_str in _decoder_cache:
        return _decoder_cache[schema_str]

    t0 = time.time()

    # Compile grammar
    key_strings, grammar = compile_schema(schema)
    depth = recommended_depth(schema)

    # Build automaton
    lr = BoundedLRAutomaton(grammar, depth=depth)
    sc = JsonScanner(key_strings=key_strings)
    composite = CompositeAutomaton(lr, sc)

    nesting, n_opt = schema_stats(schema)
    t_build = time.time() - t0
    print(f"  [lr-build] nesting={nesting}, opt={n_opt}, depth={depth}, "
          f"lr_configs={lr.num_configs}, composite_states={composite.num_states}, "
          f"build={t_build:.3f}s", flush=True)

    # Build decoder (gen_start/gen_length set by _sample per call)
    mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    decoder = ConstrainedDecoder(
        automaton=composite,
        trie=trie,
        token_to_bytes=t2b,
        gen_start=0,
        gen_length=1,  # placeholder; reset by _sample
        mask_token_id=mask_token_id,
        scanner=sc,
    )
    # Masks are computed lazily on first query (~20ms each, ~30-50 per generation).
    # No upfront precomputation needed — the composite state space is too large.

    total = time.time() - t0
    print(f"  [lr-total] {total:.2f}s (grammar+automaton build only; masks are lazy)",
          flush=True)

    _decoder_cache[schema_str] = decoder
    return decoder


# ---------------------------------------------------------------------------
# Model loading (identical to DFA version)
# ---------------------------------------------------------------------------

def load_model(model_path, device="cuda", constrained=False):
    if constrained:
        from model.generation_utils_block_constrained import DreamGenerationMixin
    else:
        from model.generation_utils_block import DreamGenerationMixin

    print(f"Loading model from {model_path}...", flush=True)
    t0 = time.time()
    model = DreamModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    model.diffusion_generate = types.MethodType(
        DreamGenerationMixin.diffusion_generate, model
    )
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    model._cd_commit_position = types.MethodType(
        DreamGenerationMixin._cd_commit_position, model
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt / generation / eval helpers (identical to DFA version)
# ---------------------------------------------------------------------------

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


def generate(model, tokenizer, inputs, steps, max_new_tokens=256,
             temperature=0.2, device="cuda", output_history=False,
             constrained_decoder=None, instance_idx=None):
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    if constrained_decoder is not None:
        print(f"\n[GEN] Instance {instance_idx}: prompt_len={prompt_len}, "
              f"max_new_tokens={max_new_tokens}, steps={steps}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=output_history,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            constrained_decoder=constrained_decoder,
        )
    elapsed = time.time() - t0

    gen_text = tokenizer.decode(
        output.sequences[0][prompt_len:].tolist()
    ).split(tokenizer.eos_token)[0]

    if constrained_decoder is not None:
        print(f"[GEN] Instance {instance_idx} done in {elapsed:.2f}s, "
              f"gen_len={len(output.sequences[0]) - prompt_len}", flush=True)

    return {
        "text": gen_text,
        "prompt_len": prompt_len,
        "gen_len": len(output.sequences[0]) - prompt_len,
        "time_seconds": elapsed,
    }


def extract_json(text):
    import re
    text = text.strip()
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    md = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if md:
        try:
            json.loads(md.group(1).strip())
            return md.group(1).strip()
        except (json.JSONDecodeError, ValueError):
            pass
    for open_ch, close_ch in [('{', '}'), ('[', ']')]:
        start = text.find(open_ch)
        if start == -1:
            continue
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
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


def check_output(gen_text, schema_str):
    extracted = extract_json(gen_text)
    result = {
        "has_json": extracted is not None,
        "syntax_valid": False,
        "schema_valid": False,
        "extracted": extracted,
        "error": None,
    }
    if extracted is None:
        result["error"] = "no JSON found"
        return result
    try:
        json.loads(extracted)
        result["syntax_valid"] = True
    except json.JSONDecodeError as e:
        result["error"] = f"syntax: {e}"
        return result
    schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
    try:
        jsonschema.validate(json.loads(extracted), schema)
        result["schema_valid"] = True
    except jsonschema.ValidationError as e:
        result["error"] = f"schema: {e.message}"
    except jsonschema.SchemaError as e:
        result["error"] = f"bad schema: {e.message}"
    return result


def check_exact_match(gen_text, ref_output):
    extracted = extract_json(gen_text)
    if extracted is None:
        return False
    try:
        gen_parsed = json.loads(extracted)
        ref_parsed = json.loads(ref_output)
        return (json.dumps(gen_parsed, sort_keys=True) ==
                json.dumps(ref_parsed, sort_keys=True))
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[128])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/constrained_lr")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--constrained", action="store_true",
                        help="Enable grammar-constrained decoding (LR backend)")
    args = parser.parse_args()

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir if args.constrained else "results/unconstrained_lr"
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances", flush=True)

    model, tokenizer = load_model(args.model_path, args.device,
                                  constrained=args.constrained)

    # Build shared vocabulary structures once (independent of schema)
    t2b = _build_t2b(tokenizer) if args.constrained else None
    shared_trie = TokenTrie(t2b) if args.constrained else None
    if args.constrained:
        print(f"  [trie] {shared_trie.num_nodes} nodes, vocab={shared_trie.vocab_size}",
              flush=True)

    all_results = {"config": vars(args), "step_results": {}}

    for step_budget in args.steps:
        print(f"\n{'='*60}", flush=True)
        print(f"Steps: {step_budget} "
              f"({'LR-constrained' if args.constrained else 'baseline'})", flush=True)
        print(f"{'='*60}", flush=True)

        instances = []
        syntax_ok = schema_ok = exact_ok = 0
        total_time = 0.0
        cache_hits = 0

        for idx in tqdm(range(len(ds)), desc=f"steps={step_budget}"):
            row = ds[idx]
            instance_id = row["instance_id"]
            schema_str = row["schema"]
            input_text = row["input"]
            ref_output = row["output"]
            schema = (json.loads(schema_str)
                      if isinstance(schema_str, str) else schema_str)

            # Build per-schema decoder (cached if schema repeats)
            constrained_decoder = None
            if args.constrained:
                schema_key = json.dumps(schema, sort_keys=True)
                was_cached = schema_key in _decoder_cache
                t_pre = time.time()
                constrained_decoder = build_grammar_decoder_lr(
                    tokenizer, schema, args.device, t2b, shared_trie
                )
                dt_pre = time.time() - t_pre
                if was_cached:
                    cache_hits += 1
                    print(f"  [cache-hit] instance {idx}, schema reuse", flush=True)
                else:
                    print(f"  [new-schema] instance {idx}, build={dt_pre:.2f}s",
                          flush=True)

            inputs = format_prompt(schema_str, input_text, tokenizer)
            gen = generate(
                model, tokenizer, inputs,
                steps=step_budget,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=args.device,
                constrained_decoder=constrained_decoder,
                instance_idx=idx,
            )

            check = check_output(gen["text"], schema_str)
            exact = check_exact_match(gen["text"], ref_output)

            if check["syntax_valid"]:
                syntax_ok += 1
            if check["schema_valid"]:
                schema_ok += 1
            if exact:
                exact_ok += 1
            total_time += gen["time_seconds"]

            n_done = idx + 1
            if args.constrained or idx % 10 == 9:
                print(
                    f"[EVAL] {n_done}/{len(ds)}: "
                    f"syntax={syntax_ok}/{n_done} ({syntax_ok/n_done:.0%}), "
                    f"schema={schema_ok}/{n_done} ({schema_ok/n_done:.0%}), "
                    f"avg_gen={total_time/n_done:.2f}s, "
                    f"cache_hits={cache_hits}"
                    f"{', error: ' + check['error'] if check['error'] else ''}",
                    flush=True,
                )

            instances.append({
                "instance_id": instance_id,
                "syntax_valid": check["syntax_valid"],
                "schema_valid": check["schema_valid"],
                "exact_match": exact,
                "error": check["error"],
                "gen_len": gen["gen_len"],
                "time_seconds": gen["time_seconds"],
                "generated_text": gen["text"][:500],
                "extracted_json": (check["extracted"][:500]
                                   if check["extracted"] else None),
            })

        n = len(ds)
        summary = {
            "steps": step_budget,
            "num_instances": n,
            "constrained": args.constrained,
            "syntax_valid_rate": syntax_ok / n,
            "schema_valid_rate": schema_ok / n,
            "exact_match_rate": exact_ok / n,
            "avg_time_seconds": total_time / n,
            "total_precomp_seconds": precomp_time,
            "cache_hits": cache_hits,
            "unique_schemas_built": n - cache_hits,
        }

        print(f"\nResults for steps={step_budget}:", flush=True)
        print(f"  Syntax valid:  {syntax_ok}/{n} ({summary['syntax_valid_rate']:.1%})",
              flush=True)
        print(f"  Schema valid:  {schema_ok}/{n} ({summary['schema_valid_rate']:.1%})",
              flush=True)
        print(f"  Exact match:   {exact_ok}/{n} ({summary['exact_match_rate']:.1%})",
              flush=True)
        print(f"  Avg gen time:  {summary['avg_time_seconds']:.2f}s/inst", flush=True)
        print(f"  Total precomp: {precomp_time:.1f}s "
              f"({n - cache_hits} unique schemas, {cache_hits} cache hits)", flush=True)

        all_results["step_results"][step_budget] = {
            "summary": summary,
            "instances": instances,
        }

    out_path = os.path.join(output_dir, "json_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    print(f"\n{'Steps':>6} | {'Syntax':>8} | {'Schema':>8} | {'Exact':>8} | "
          f"{'Time/inst':>10} | {'Precomp':>10}", flush=True)
    print("-" * 65, flush=True)
    for step_budget in args.steps:
        s = all_results["step_results"][step_budget]["summary"]
        print(
            f"{s['steps']:>6} | {s['syntax_valid_rate']:>7.1%} | "
            f"{s['schema_valid_rate']:>7.1%} | {s['exact_match_rate']:>7.1%} | "
            f"{s['avg_time_seconds']:>9.2f}s | "
            f"{s['total_precomp_seconds']:>9.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()