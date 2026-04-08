#!/usr/bin/env python3
"""
Experiment 8: Length-aware structural planning.

At each denoising step, check whether there are enough remaining masked
positions to close all open brackets/braces. If not, force-unmask closing
tokens to prevent uncloseable states.

Usage:
  python eval/eval_json_lengthaware.py \
      --steps 32 64 --num_instances 50 --output_dir results/lengthaware
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
CLOSE_FOR_OPEN = {'{': '}', '[': ']'}


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


def make_lengthaware_hook(tokenizer, mask_token_id, prompt_len):
    """
    After each step, check if open structures can still be closed given
    remaining masked positions. If not, force-close from the end.
    """
    # pre-compute token ids for closing characters
    close_token_ids = {}
    vocab = tokenizer.get_vocab()
    for tok_str, tok_id in vocab.items():
        decoded = tokenizer.decode([tok_id]).strip()
        if decoded == '}':
            close_token_ids['}'] = tok_id
        elif decoded == ']':
            close_token_ids[']'] = tok_id
    # fallback: encode single characters
    if '}' not in close_token_ids:
        close_token_ids['}'] = tokenizer.encode('}', add_special_tokens=False)[0]
    if ']' not in close_token_ids:
        close_token_ids[']'] = tokenizer.encode(']', add_special_tokens=False)[0]

    stats = {"interventions": 0, "tokens_forced": 0}

    def hook(step, x, logits):
        gen_region = x[0, prompt_len:]

        # scan for open structures
        stack = []
        for pos, tid in enumerate(gen_region.tolist()):
            if tid == mask_token_id:
                continue
            decoded = tokenizer.decode([tid])
            for c in decoded:
                if c in '{[':
                    stack.append(c)
                elif c == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                elif c == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()

        if not stack:
            return x

        # count remaining masked positions
        mask_positions = []
        for pos in range(len(gen_region)):
            if gen_region[pos].item() == mask_token_id:
                mask_positions.append(pos)

        remaining_masks = len(mask_positions)
        open_count = len(stack)

        # if we're running out of room, force-close from the end
        if open_count >= remaining_masks and remaining_masks > 0:
            forced = 0
            # fill from the last masked positions
            for i in range(min(open_count, remaining_masks)):
                closer = stack.pop()
                close_char = CLOSE_FOR_OPEN[closer]
                close_tid = close_token_ids[close_char]
                pos = mask_positions[-(i + 1)]  # last masked positions
                x[0, prompt_len + pos] = close_tid
                forced += 1

            if forced > 0:
                stats["interventions"] += 1
                stats["tokens_forced"] += forced

        return x

    return hook, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/lengthaware")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(args.model_path, args.device)
    mask_token_id = tokenizer.mask_token_id

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances")

    all_results = {}

    for step_budget in args.steps:
        print(f"\n{'='*60}")
        print(f"Steps: {step_budget}")
        print(f"{'='*60}")

        syntax_ok = 0
        schema_ok = 0
        total_interventions = 0
        total_forced = 0
        instances = []

        for idx in tqdm(range(len(ds)), desc=f"steps={step_budget}"):
            row = ds[idx]
            inputs = format_prompt(row["schema"], row["input"], tokenizer)
            input_ids = inputs.input_ids.to(args.device)
            attention_mask = inputs.attention_mask.to(args.device)
            prompt_len = input_ids.shape[1]

            hook, stats = make_lengthaware_hook(tokenizer, mask_token_id, prompt_len)

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
            total_interventions += stats["interventions"]
            total_forced += stats["tokens_forced"]

            instances.append({
                "instance_id": row["instance_id"],
                "syntax_valid": check["syntax_valid"],
                "schema_valid": check["schema_valid"],
                "error": check["error"],
                "interventions": stats["interventions"],
                "tokens_forced": stats["tokens_forced"],
            })

        n = len(ds)
        summary = {
            "steps": step_budget,
            "syntax_rate": syntax_ok / n,
            "schema_rate": schema_ok / n,
            "avg_interventions": total_interventions / n,
            "avg_tokens_forced": total_forced / n,
        }
        all_results[step_budget] = {"summary": summary, "instances": instances}

        print(f"  Syntax: {syntax_ok}/{n} ({syntax_ok/n:.1%})")
        print(f"  Schema: {schema_ok}/{n} ({schema_ok/n:.1%})")
        print(f"  Avg interventions per instance: {total_interventions/n:.1f}")
        print(f"  Avg tokens forced per instance: {total_forced/n:.1f}")

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'Steps':>6} | {'Syntax':>8} | {'Schema':>8} | {'Interventions':>14} | {'Forced':>8}")
    print("-" * 55)
    for step_budget in args.steps:
        s = all_results[step_budget]["summary"]
        print(f"{s['steps']:>6} | {s['syntax_rate']:>7.1%} | {s['schema_rate']:>7.1%} | {s['avg_interventions']:>13.1f} | {s['avg_tokens_forced']:>7.1f}")


if __name__ == "__main__":
    main()