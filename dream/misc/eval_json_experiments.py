#!/usr/bin/env python3
"""
Experiments 1-3 & 5: Denoising history analysis and anchor scaling.

Exp 1 (Phase Transition): When do structural errors become unrecoverable?
Exp 2 (Error Cascade): Do early errors cause cascading failures?
Exp 3 (Critical Positions): Which positions matter most for validity?
Exp 5 (Anchor Scaling): How does validity improve with more anchored tokens?

Usage:
  # Collect histories (Exp 1-3):
  python eval/eval_json_experiments.py history \
      --steps 256 --num_instances 50 --output_dir results/history

  # Anchor scaling (Exp 5):
  python eval/eval_json_experiments.py anchor \
      --steps 32 --num_instances 50 --output_dir results/anchor

  # Step budget scaling (Exp 4) - just use existing eval_json.py:
  python eval/eval_json.py --steps 16 32 48 64 96 128 192 256 384 512
"""

import argparse
import json
import os
import time
import types

import jsonschema
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from model.generation_utils import DreamGenerationMixin

JSON_STRUCTURAL_CHARS = set('{}[],:\"')


def load_model(model_path, device="cuda"):
    model = DreamModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(device)
    model.diffusion_generate = types.MethodType(
        DreamGenerationMixin.diffusion_generate, model
    )
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    return model, tokenizer


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
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


def check_output(gen_text, schema_str):
    extracted = extract_json(gen_text)
    result = {"syntax_valid": False, "schema_valid": False, "error": None}
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


# ── Structural analysis utilities ──────────────────────────────────────

def check_partial_structure(token_ids, tokenizer, mask_token_id):
    """
    Check bracket/brace consistency of unmasked tokens.
    Returns dict with error count, stack state, and details.
    """
    stack = []
    errors = []
    structural_count = 0
    content_count = 0

    for pos, tid in enumerate(token_ids):
        if tid == mask_token_id:
            continue
        decoded = tokenizer.decode([tid])
        for c in decoded:
            if c in '{[':
                stack.append((c, pos))
                structural_count += 1
            elif c == '}':
                structural_count += 1
                if not stack or stack[-1][0] != '{':
                    errors.append({"type": "mismatch", "char": "}", "pos": pos,
                                   "expected": "{" if stack else "nothing"})
                else:
                    stack.pop()
            elif c == ']':
                structural_count += 1
                if not stack or stack[-1][0] != '[':
                    errors.append({"type": "mismatch", "char": "]", "pos": pos,
                                   "expected": "[" if stack else "nothing"})
                else:
                    stack.pop()
            elif c in ',:':
                structural_count += 1
            elif c == '"':
                structural_count += 1
            elif not c.isspace():
                content_count += 1

    return {
        "errors": len(errors),
        "error_details": errors,
        "open_stack": len(stack),
        "structural_count": structural_count,
        "content_count": content_count,
    }


def analyze_history(history, tokenizer, mask_token_id, prompt_len):
    """
    Analyze full denoising history for Experiments 1-3.
    Returns per-step analysis.
    """
    steps = []
    prev_unmasked = set()

    for step_idx, seq in enumerate(history):
        gen_region = seq[prompt_len:]
        total_pos = len(gen_region)
        masked = sum(1 for t in gen_region if t == mask_token_id)
        unmasked = total_pos - masked
        mask_ratio = masked / total_pos if total_pos > 0 else 0

        # which positions were newly unmasked this step
        current_unmasked = set()
        newly_unmasked = []
        for pos, tid in enumerate(gen_region):
            if tid != mask_token_id:
                current_unmasked.add(pos)
                if pos not in prev_unmasked:
                    decoded = tokenizer.decode([tid])
                    is_structural = any(c in JSON_STRUCTURAL_CHARS for c in decoded)
                    newly_unmasked.append({
                        "pos": pos,
                        "token_id": tid,
                        "token_str": decoded,
                        "is_structural": is_structural,
                    })

        # structural consistency check
        structure = check_partial_structure(gen_region, tokenizer, mask_token_id)

        steps.append({
            "step": step_idx,
            "mask_ratio": mask_ratio,
            "unmasked_count": unmasked,
            "newly_unmasked_count": len(newly_unmasked),
            "newly_unmasked_structural": sum(1 for n in newly_unmasked if n["is_structural"]),
            "newly_unmasked_content": sum(1 for n in newly_unmasked if not n["is_structural"]),
            "structural_errors": structure["errors"],
            "open_stack": structure["open_stack"],
            "total_structural": structure["structural_count"],
            "total_content": structure["content_count"],
            "newly_unmasked_details": newly_unmasked,
        })

        prev_unmasked = current_unmasked

    return steps


def find_first_error_step(step_analyses):
    """Find the first step where a structural error appears."""
    for s in step_analyses:
        if s["structural_errors"] > 0:
            return s["step"], s["mask_ratio"]
    return None, None


# ── Experiment: History collection (Exp 1-3) ──────────────────────────

def run_history_experiment(args):
    model, tokenizer = load_model(args.model_path, args.device)
    mask_token_id = tokenizer.mask_token_id

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances")

    results = []

    for idx in tqdm(range(len(ds)), desc="Collecting histories"):
        row = ds[idx]
        inputs = format_prompt(row["schema"], row["input"], tokenizer)
        input_ids = inputs.input_ids.to(args.device)
        attention_mask = inputs.attention_mask.to(args.device)
        prompt_len = input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                output_history=True,
                return_dict_in_generate=True,
                steps=args.steps[0],
                temperature=args.temperature,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.0,
            )
        elapsed = time.time() - t0

        gen_text = tokenizer.decode(
            output.sequences[0][prompt_len:].tolist()
        ).split(tokenizer.eos_token)[0]
        check = check_output(gen_text, row["schema"])

        # analyze history
        history_ids = [h[0].tolist() for h in output.history] if output.history else []
        step_analyses = analyze_history(history_ids, tokenizer, mask_token_id, prompt_len)
        first_err_step, first_err_mask_ratio = find_first_error_step(step_analyses)

        # compute summary stats
        structural_unmask_order = []
        for s in step_analyses:
            for tok in s["newly_unmasked_details"]:
                structural_unmask_order.append({
                    "step": s["step"],
                    "mask_ratio": s["mask_ratio"],
                    "pos": tok["pos"],
                    "is_structural": tok["is_structural"],
                    "token_str": tok["token_str"],
                })

        # compact step-level data (drop per-token details for storage)
        step_summary = [{
            "step": s["step"],
            "mask_ratio": round(s["mask_ratio"], 4),
            "structural_errors": s["structural_errors"],
            "open_stack": s["open_stack"],
            "newly_structural": s["newly_unmasked_structural"],
            "newly_content": s["newly_unmasked_content"],
            "total_structural": s["total_structural"],
            "total_content": s["total_content"],
        } for s in step_analyses]

        results.append({
            "instance_id": row["instance_id"],
            "final_syntax_valid": check["syntax_valid"],
            "final_schema_valid": check["schema_valid"],
            "final_error": check["error"],
            "first_error_step": first_err_step,
            "first_error_mask_ratio": first_err_mask_ratio,
            "total_steps": len(step_analyses),
            "time_seconds": elapsed,
            "step_summary": step_summary,
            "generated_text": gen_text[:500],
        })

    # save raw results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "history_data.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw data saved to {out_path}")

    # print Experiment 1 summary: phase transition
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Phase Transition")
    print(f"{'='*60}")
    valid = [r for r in results if r["final_syntax_valid"]]
    invalid = [r for r in results if not r["final_syntax_valid"]]
    print(f"Valid: {len(valid)}/{len(results)}, Invalid: {len(invalid)}/{len(results)}")

    if invalid:
        err_steps = [r["first_error_step"] for r in invalid if r["first_error_step"] is not None]
        err_ratios = [r["first_error_mask_ratio"] for r in invalid if r["first_error_mask_ratio"] is not None]
        if err_steps:
            print(f"\nInvalid outputs — first structural error:")
            print(f"  Earliest step: {min(err_steps)}")
            print(f"  Median step:   {sorted(err_steps)[len(err_steps)//2]}")
            print(f"  Latest step:   {max(err_steps)}")
            print(f"  Median mask ratio at first error: {sorted(err_ratios)[len(err_ratios)//2]:.3f}")

    if valid:
        valid_with_transient = [r for r in valid
                                if any(s["structural_errors"] > 0 for s in r["step_summary"])]
        print(f"\nValid outputs with transient errors: {len(valid_with_transient)}/{len(valid)}")

    # print Experiment 2 summary: error cascade
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Error Cascade")
    print(f"{'='*60}")
    for r in invalid[:5]:
        if r["first_error_step"] is None:
            continue
        err_progression = [(s["step"], s["structural_errors"])
                          for s in r["step_summary"] if s["structural_errors"] > 0]
        if err_progression:
            print(f"\n  {r['instance_id']} (first error at step {r['first_error_step']}):")
            for step, errs in err_progression[:10]:
                print(f"    Step {step:>3d}: {errs} error(s)")

    # print Experiment 3 summary: early vs late structural tokens
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Structural Token Timing")
    print(f"{'='*60}")
    # aggregate across all instances: when are structural tokens unmasked?
    valid_structural_steps = []
    invalid_structural_steps = []
    for r in results:
        bucket = valid_structural_steps if r["final_syntax_valid"] else invalid_structural_steps
        for s in r["step_summary"]:
            if s["newly_structural"] > 0:
                bucket.extend([s["mask_ratio"]] * s["newly_structural"])

    if valid_structural_steps:
        vs = sorted(valid_structural_steps)
        print(f"\nValid outputs — structural tokens unmasked at mask_ratio:")
        print(f"  25th percentile: {vs[len(vs)//4]:.3f}")
        print(f"  Median:          {vs[len(vs)//2]:.3f}")
        print(f"  75th percentile: {vs[3*len(vs)//4]:.3f}")
    if invalid_structural_steps:
        ivs = sorted(invalid_structural_steps)
        print(f"\nInvalid outputs — structural tokens unmasked at mask_ratio:")
        print(f"  25th percentile: {ivs[len(ivs)//4]:.3f}")
        print(f"  Median:          {ivs[len(ivs)//2]:.3f}")
        print(f"  75th percentile: {ivs[3*len(ivs)//4]:.3f}")


# ── Experiment 5: Anchor scaling ──────────────────────────────────────

def find_structural_positions(ref_output, tokenizer):
    ref_ids = tokenizer.encode(ref_output, add_special_tokens=False)
    structural = []
    for i, tok_id in enumerate(ref_ids):
        decoded = tokenizer.decode([tok_id]).strip()
        if decoded and decoded[0] in JSON_STRUCTURAL_CHARS:
            structural.append((i, tok_id))
    return ref_ids, structural


def run_anchor_experiment(args):
    model, tokenizer = load_model(args.model_path, args.device)

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances")

    pin_counts = [0, 1, 2, 3, 5, 8, 10, 999]  # 999 = all
    step_budget = args.steps[0]

    all_results = {}

    for max_pins in pin_counts:
        label = "all" if max_pins == 999 else str(max_pins)
        print(f"\n{'='*60}")
        print(f"Anchor scaling: max_pins={label}, steps={step_budget}")
        print(f"{'='*60}")

        syntax_ok = 0
        schema_ok = 0

        for idx in tqdm(range(len(ds)), desc=f"pins={label}"):
            row = ds[idx]
            inputs = format_prompt(row["schema"], row["input"], tokenizer)
            input_ids = inputs.input_ids.to(args.device)
            attention_mask = inputs.attention_mask.to(args.device)
            prompt_len = input_ids.shape[1]

            ref_ids, structural_positions = find_structural_positions(row["output"], tokenizer)
            pins = structural_positions if max_pins == 999 else structural_positions[:max_pins]

            def make_hook(pins_local, pl):
                def hook(step, x, logits):
                    for offset, tok_id in pins_local:
                        pos = pl + offset
                        if pos < x.shape[1]:
                            x[0, pos] = tok_id
                    return x
                return hook

            hook = make_hook(pins, prompt_len) if pins else (lambda step, x, logits: x)

            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=step_budget,
                    temperature=args.temperature,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.0,
                    generation_tokens_hook_func=hook,
                )

            gen_text = tokenizer.decode(
                output.sequences[0][prompt_len:].tolist()
            ).split(tokenizer.eos_token)[0]
            check = check_output(gen_text, row["schema"])

            if check["syntax_valid"]:
                syntax_ok += 1
            if check["schema_valid"]:
                schema_ok += 1

        n = len(ds)
        all_results[label] = {
            "max_pins": max_pins,
            "syntax_rate": syntax_ok / n,
            "schema_rate": schema_ok / n,
        }
        print(f"  Syntax: {syntax_ok}/{n} ({syntax_ok/n:.1%})  Schema: {schema_ok}/{n} ({schema_ok/n:.1%})")

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "anchor_scaling.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'Pins':>6} | {'Syntax':>8} | {'Schema':>8}")
    print("-" * 30)
    for label, r in all_results.items():
        print(f"{label:>6} | {r['syntax_rate']:>7.1%} | {r['schema_rate']:>7.1%}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["history", "anchor"],
                        help="'history' for Exp 1-3, 'anchor' for Exp 5")
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[256])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/experiments")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.mode == "history":
        run_history_experiment(args)
    elif args.mode == "anchor":
        run_anchor_experiment(args)


if __name__ == "__main__":
    main()