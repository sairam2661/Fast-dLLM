#!/usr/bin/env python3
"""
Evaluation: Dream-Instruct with grammar-constrained decoding
on eth-sri/json-mode-eval-extended.

Compares baseline (no constraints) vs constrained decoding across
step budgets.

Usage:
  # Baseline
  python eval/eval_json_constrained.py \
      --model_path Dream-org/Dream-v0-Instruct-7B \
      --steps 32 64 128 256 \
      --num_instances 50 \
      --output_dir results/baseline

  # Constrained
  python eval/eval_json_constrained.py \
      --model_path Dream-org/Dream-v0-Instruct-7B \
      --steps 32 64 128 256 \
      --num_instances 50 \
      --output_dir results/constrained \
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
from constrained.constrained_decoder import build_constrained_decoder, ConstrainedDecoder


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
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    return model, tokenizer


def build_grammar_decoder(tokenizer, device="cuda"):
    """Build the constrained decoder (DFA + trie). One-time cost."""

    print("Building grammar decoder...", flush=True)
    t0 = time.time()
    dfa, trie, t2b = build_constrained_decoder(tokenizer, max_depth=6)
    elapsed = time.time() - t0
    print(f"  DFA states: {dfa.num_states}, Trie nodes: {trie.num_nodes}", flush=True)
    print(f"  Built in {elapsed:.1f}s", flush=True)

    # Create a decoder instance (gen_start/gen_length set per generation)
    decoder = ConstrainedDecoder(
        dfa=dfa, trie=trie, token_to_bytes=t2b,
        gen_start=0, gen_length=1,  # placeholder, reset per call
        mask_token_id=tokenizer.convert_tokens_to_ids("<|mask|>"),
    )

    # Precompute state masks upfront so it doesn't block the first instance
    print(f"Precomputing state masks ({dfa.num_states} states × {len(t2b)} tokens)...",
          flush=True)
    t1 = time.time()
    decoder.precompute_state_masks(device)
    print(f"  State masks precomputed in {time.time() - t1:.1f}s", flush=True)

    return decoder


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

    result = {
        "text": gen_text,
        "prompt_len": prompt_len,
        "gen_len": len(output.sequences[0]) - prompt_len,
        "time_seconds": elapsed,
    }
    return result


def extract_json(text):
    """Try to extract a JSON object/array from generated text."""
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
                    candidate = text[start:i+1]
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
        parsed = json.loads(extracted)
        result["syntax_valid"] = True
    except json.JSONDecodeError as e:
        result["error"] = f"syntax: {e}"
        return result
    schema = json.loads(schema_str)
    try:
        jsonschema.validate(parsed, schema)
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
        return json.dumps(gen_parsed, sort_keys=True) == json.dumps(ref_parsed, sort_keys=True)
    except (json.JSONDecodeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/constrained")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--constrained", action="store_true",
                        help="Enable grammar-constrained decoding")
    args = parser.parse_args()

    # Force unbuffered output so logs appear immediately
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances", flush=True)

    model, tokenizer = load_model(args.model_path, args.device,
                                   constrained=args.constrained)

    constrained_decoder = None
    if args.constrained:
        constrained_decoder = build_grammar_decoder(tokenizer, args.device)

    all_results = {
        "config": vars(args),
        "step_results": {},
    }

    for step_budget in args.steps:
        print(f"\n{'='*60}", flush=True)
        print(f"Steps: {step_budget} ({'constrained' if args.constrained else 'baseline'})",
              flush=True)
        print(f"{'='*60}", flush=True)

        instances = []
        syntax_ok = 0
        schema_ok = 0
        exact_ok = 0
        total_time = 0.0

        for idx in tqdm(range(len(ds)), desc=f"steps={step_budget}"):
            row = ds[idx]
            instance_id = row["instance_id"]
            schema_str = row["schema"]
            input_text = row["input"]
            ref_output = row["output"]

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

            # Print running stats every instance when constrained
            if args.constrained or idx % 10 == 9:
                n_done = idx + 1
                print(f"[EVAL] {n_done}/{len(ds)} done: "
                      f"syntax={syntax_ok}/{n_done} ({syntax_ok/n_done:.0%}), "
                      f"schema={schema_ok}/{n_done} ({schema_ok/n_done:.0%}), "
                      f"avg_time={total_time/n_done:.2f}s"
                      f"{' | error: ' + check['error'] if check['error'] else ''}",
                      flush=True)

            instance_result = {
                "instance_id": instance_id,
                "syntax_valid": check["syntax_valid"],
                "schema_valid": check["schema_valid"],
                "exact_match": exact,
                "error": check["error"],
                "gen_len": gen["gen_len"],
                "time_seconds": gen["time_seconds"],
                "generated_text": gen["text"][:500],
                "extracted_json": check["extracted"][:500] if check["extracted"] else None,
            }
            instances.append(instance_result)

        n = len(ds)
        summary = {
            "steps": step_budget,
            "num_instances": n,
            "constrained": args.constrained,
            "syntax_valid_rate": syntax_ok / n,
            "schema_valid_rate": schema_ok / n,
            "exact_match_rate": exact_ok / n,
            "avg_time_seconds": total_time / n,
        }

        print(f"\nResults for steps={step_budget}:", flush=True)
        print(f"  Syntax valid:  {syntax_ok}/{n} ({summary['syntax_valid_rate']:.1%})", flush=True)
        print(f"  Schema valid:  {schema_ok}/{n} ({summary['schema_valid_rate']:.1%})", flush=True)
        print(f"  Exact match:   {exact_ok}/{n} ({summary['exact_match_rate']:.1%})", flush=True)
        print(f"  Avg time/inst: {summary['avg_time_seconds']:.2f}s", flush=True)

        all_results["step_results"][step_budget] = {
            "summary": summary,
            "instances": instances,
        }

    out_path = os.path.join(args.output_dir, "json_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    print(f"\n{'Steps':>6} | {'Syntax':>8} | {'Schema':>8} | {'Exact':>8} | {'Time/inst':>10}",
          flush=True)
    print("-" * 50, flush=True)
    for step_budget in args.steps:
        s = all_results["step_results"][step_budget]["summary"]
        print(f"{s['steps']:>6} | {s['syntax_valid_rate']:>7.1%} | {s['schema_valid_rate']:>7.1%} | "
              f"{s['exact_match_rate']:>7.1%} | {s['avg_time_seconds']:>9.2f}s", flush=True)


if __name__ == "__main__":
    main()