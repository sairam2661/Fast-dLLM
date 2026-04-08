#!/usr/bin/env python3
"""
Experiment 6: Simple constraint propagation via re-masking.
Experiment 7: Anchor + propagation composition.

Usage:
  # Exp 6: re-masking only
  python eval/eval_json_constraints.py remasking \
      --steps 32 64 --num_instances 50 --output_dir results/remasking

  # Exp 7: anchoring + re-masking
  python eval/eval_json_constraints.py composed \
      --steps 32 64 --num_instances 50 --output_dir results/composed
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


def find_structural_positions(ref_output, tokenizer):
    ref_ids = tokenizer.encode(ref_output, add_special_tokens=False)
    structural = []
    for i, tok_id in enumerate(ref_ids):
        decoded = tokenizer.decode([tok_id]).strip()
        if decoded and decoded[0] in JSON_STRUCTURAL_CHARS:
            structural.append((i, tok_id))
    return ref_ids, structural


# ── Re-masking logic ──────────────────────────────────────────────────

def find_bracket_violations(gen_region, tokenizer, mask_token_id):
    """
    Scan unmasked tokens left-to-right. Return positions of tokens that
    create bracket/brace mismatches.
    """
    stack = []
    violations = []

    for pos, tid in enumerate(gen_region):
        if tid == mask_token_id:
            continue
        decoded = tokenizer.decode([tid])
        for c in decoded:
            if c in '{[':
                stack.append((c, pos))
            elif c == '}':
                if not stack or stack[-1][0] != '{':
                    violations.append(pos)
                else:
                    stack.pop()
            elif c == ']':
                if not stack or stack[-1][0] != '[':
                    violations.append(pos)
                else:
                    stack.pop()

    return violations


def make_remasking_hook(tokenizer, mask_token_id, prompt_len, anchor_positions=None):
    """
    Create a hook that:
    1. Optionally pins anchor tokens
    2. Checks bracket consistency and re-masks violating tokens
    """
    stats = {"total_remasked": 0, "steps_with_remasking": 0}

    def hook(step, x, logits):
        # pin anchors first
        if anchor_positions:
            for offset, tok_id in anchor_positions:
                pos = prompt_len + offset
                if pos < x.shape[1]:
                    x[0, pos] = tok_id

        # check bracket consistency in generation region
        gen_region = x[0, prompt_len:].tolist()
        violations = find_bracket_violations(gen_region, tokenizer, mask_token_id)

        if violations:
            # don't re-mask anchor positions
            anchor_offsets = set(off for off, _ in anchor_positions) if anchor_positions else set()
            remasked = 0
            for pos in violations:
                if pos not in anchor_offsets:
                    x[0, prompt_len + pos] = mask_token_id
                    remasked += 1
            if remasked > 0:
                stats["total_remasked"] += remasked
                stats["steps_with_remasking"] += 1

        return x

    return hook, stats


# ── Run experiments ───────────────────────────────────────────────────

def run_experiment(args, use_anchors=False):
    model, tokenizer = load_model(args.model_path, args.device)
    mask_token_id = tokenizer.mask_token_id

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances")

    anchor_counts = [0, 1, 3] if use_anchors else [0]
    all_results = {}

    for step_budget in args.steps:
        for num_anchors in anchor_counts:
            label = f"steps={step_budget}_anchors={num_anchors}"
            print(f"\n{'='*60}")
            print(f"{label}")
            print(f"{'='*60}")

            syntax_ok = 0
            schema_ok = 0
            total_remasked = 0
            total_steps_with_remasking = 0
            instances = []

            for idx in tqdm(range(len(ds)), desc=label):
                row = ds[idx]
                inputs = format_prompt(row["schema"], row["input"], tokenizer)
                input_ids = inputs.input_ids.to(args.device)
                attention_mask = inputs.attention_mask.to(args.device)
                prompt_len = input_ids.shape[1]

                # set up anchors
                anchor_positions = None
                if num_anchors > 0:
                    _, structural = find_structural_positions(row["output"], tokenizer)
                    anchor_positions = structural[:num_anchors]

                hook, stats = make_remasking_hook(
                    tokenizer, mask_token_id, prompt_len, anchor_positions
                )

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
                total_remasked += stats["total_remasked"]
                total_steps_with_remasking += stats["steps_with_remasking"]

                instances.append({
                    "instance_id": row["instance_id"],
                    "syntax_valid": check["syntax_valid"],
                    "schema_valid": check["schema_valid"],
                    "error": check["error"],
                    "remasked_tokens": stats["total_remasked"],
                    "steps_with_remasking": stats["steps_with_remasking"],
                })

            n = len(ds)
            summary = {
                "steps": step_budget,
                "num_anchors": num_anchors,
                "syntax_rate": syntax_ok / n,
                "schema_rate": schema_ok / n,
                "avg_remasked_per_instance": total_remasked / n,
                "avg_steps_with_remasking": total_steps_with_remasking / n,
            }
            all_results[label] = {"summary": summary, "instances": instances}

            print(f"  Syntax: {syntax_ok}/{n} ({syntax_ok/n:.1%})")
            print(f"  Schema: {schema_ok}/{n} ({schema_ok/n:.1%})")
            print(f"  Avg tokens re-masked per instance: {total_remasked/n:.1f}")
            print(f"  Avg steps with re-masking: {total_steps_with_remasking/n:.1f}")

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # print summary table
    print(f"\n{'Config':>30} | {'Syntax':>8} | {'Schema':>8} | {'Remasked':>10}")
    print("-" * 65)
    for label, r in all_results.items():
        s = r["summary"]
        print(f"{label:>30} | {s['syntax_rate']:>7.1%} | {s['schema_rate']:>7.1%} | {s['avg_remasked_per_instance']:>9.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["remasking", "composed"],
                        help="'remasking' for Exp 6, 'composed' for Exp 7")
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/constraints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.mode == "remasking":
        run_experiment(args, use_anchors=False)
    elif args.mode == "composed":
        run_experiment(args, use_anchors=True)


if __name__ == "__main__":
    main()