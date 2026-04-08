#!/usr/bin/env python3
"""
Experiment 11: One-step greedy lookahead at structural boundaries.

When a structural token is unmasked at position i, check position i+1's
logits. If i+1's top prediction is incompatible, replace it with the
highest-probability compatible token. No extra forward pass needed.

Usage:
  python eval/eval_json_lookahead.py \
      --steps 32 64 --num_instances 50 --output_dir results/lookahead
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


def build_compatible_token_sets(tokenizer):
    """
    For each structural character, pre-compute which token IDs are
    compatible as the next token.
    """
    vocab = tokenizer.get_vocab()
    vocab_size = max(vocab.values()) + 1

    # for each rule, build a mask of compatible token ids
    def tokens_starting_with(chars):
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for tok_str, tok_id in vocab.items():
            decoded = tokenizer.decode([tok_id]).lstrip()
            if decoded and decoded[0] in chars:
                mask[tok_id] = True
        return mask

    value_starters = '"{}[]-0123456789tfn'

    rules = {
        '{': tokens_starting_with('"{}'),       # after {: must be " or } or {
        '[': tokens_starting_with(value_starters + ']'),  # after [: value start or ]
        ':': tokens_starting_with(value_starters),  # after :: value start
        ',': tokens_starting_with('"{['),        # after , in object: " or { or [
    }

    return rules


def make_lookahead_hook(tokenizer, mask_token_id, prompt_len, compatible_sets, device):
    """
    After each step, find newly unmasked structural tokens.
    Check if their right neighbor (position i+1) is compatible.
    If not, and if we have logits, replace with best compatible token.
    """
    stats = {"corrections": 0, "checks": 0}
    structural_chars = set('{[:,')  # chars that trigger a right-neighbor check

    # track what was masked last step to detect newly unmasked
    prev_masked = None

    def hook(step, x, logits):
        nonlocal prev_masked

        gen_region = x[0, prompt_len:]
        gen_len = len(gen_region)

        current_masked = set()
        for pos in range(gen_len):
            if gen_region[pos].item() == mask_token_id:
                current_masked.add(pos)

        # on first call, just record state
        if prev_masked is None:
            prev_masked = current_masked
            return x

        # find newly unmasked positions
        newly_unmasked = prev_masked - current_masked

        for pos in sorted(newly_unmasked):
            tid = gen_region[pos].item()
            decoded = tokenizer.decode([tid]).strip()
            if not decoded:
                continue

            # check last char of this token
            trigger_char = decoded[-1]
            if trigger_char not in structural_chars:
                continue

            # check position pos+1
            next_pos = pos + 1
            if next_pos >= gen_len:
                continue

            next_tid = gen_region[next_pos].item()
            if next_tid != mask_token_id:
                # neighbor still masked — skip, it'll be predicted later
                continue

            # position i+1 is still masked — pre-fill with best compatible token
            stats["checks"] += 1

            compat_mask = compatible_sets.get(trigger_char)
            if compat_mask is None:
                continue

            if logits is None:
                continue

            logit_pos = prompt_len + next_pos
            if logit_pos >= logits.shape[1]:
                continue

            next_logits = logits[0, logit_pos].clone()

            logit_size = next_logits.shape[0]
            if compat_mask.shape[0] < logit_size:
                padded = torch.zeros(logit_size, dtype=torch.bool)
                padded[:compat_mask.shape[0]] = compat_mask
                compat_mask = padded
            next_logits[~compat_mask.to(next_logits.device)] = float('-inf')

            best_tid = next_logits.argmax().item()
            x[0, prompt_len + next_pos] = best_tid
            stats["corrections"] += 1

        prev_masked = current_masked
        return x

    return hook, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/lookahead")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(args.model_path, args.device)
    mask_token_id = tokenizer.mask_token_id

    compatible_sets = build_compatible_token_sets(tokenizer)
    print("Compatible token sets built:")
    for char, mask in compatible_sets.items():
        print(f"  After '{char}': {mask.sum().item()} compatible tokens")

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
        total_corrections = 0
        total_checks = 0
        instances = []

        for idx in tqdm(range(len(ds)), desc=f"steps={step_budget}"):
            row = ds[idx]
            inputs = format_prompt(row["schema"], row["input"], tokenizer)
            input_ids = inputs.input_ids.to(args.device)
            attention_mask = inputs.attention_mask.to(args.device)
            prompt_len = input_ids.shape[1]

            hook, stats = make_lookahead_hook(
                tokenizer, mask_token_id, prompt_len,
                compatible_sets, args.device
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
            total_corrections += stats["corrections"]
            total_checks += stats["checks"]

            instances.append({
                "instance_id": row["instance_id"],
                "syntax_valid": check["syntax_valid"],
                "schema_valid": check["schema_valid"],
                "error": check["error"],
                "corrections": stats["corrections"],
                "checks": stats["checks"],
            })

        n = len(ds)
        summary = {
            "steps": step_budget,
            "syntax_rate": syntax_ok / n,
            "schema_rate": schema_ok / n,
            "avg_corrections": total_corrections / n,
            "avg_checks": total_checks / n,
        }
        all_results[step_budget] = {"summary": summary, "instances": instances}

        print(f"  Syntax: {syntax_ok}/{n} ({syntax_ok/n:.1%})")
        print(f"  Schema: {schema_ok}/{n} ({schema_ok/n:.1%})")
        print(f"  Avg boundary checks: {total_checks/n:.1f}")
        print(f"  Avg corrections: {total_corrections/n:.1f}")

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'Steps':>6} | {'Syntax':>8} | {'Schema':>8} | {'Checks':>8} | {'Corrections':>12}")
    print("-" * 55)
    for step_budget in args.steps:
        s = all_results[step_budget]["summary"]
        print(f"{s['steps']:>6} | {s['syntax_rate']:>7.1%} | {s['schema_rate']:>7.1%} | {s['avg_checks']:>7.1f} | {s['avg_corrections']:>11.1f}")


if __name__ == "__main__":
    main()