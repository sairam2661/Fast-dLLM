#!/usr/bin/env python3
"""
Experiment 9: Boundary token enforcement.

After each denoising step, scan for structural tokens. Check whether
their immediate neighbors form valid local sequences. Re-mask offending
neighbors.

Usage:
  python eval/eval_json_boundary.py \
      --steps 32 64 --num_instances 50 --output_dir results/boundary
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


def make_boundary_hook(tokenizer, mask_token_id, prompt_len):
    """
    After each step, check local validity around structural tokens.
    Re-mask neighbors that create invalid local sequences.

    Rules based on observed failure modes:
    - After { : next non-mask token must start with "  or be { [ or }
    - After [ : next non-mask token must start with " { [ ] or be a value start
    - After : : next non-mask token must start with " { [ or be a value start (digit, t, f, n)
    - After , : next non-mask token must start with " { [
    - No doubled delimiters: ":": or {" "{ etc.
    - After " that closes a key, next must be :
    """
    stats = {"total_remasked": 0, "steps_with_remasking": 0, "rule_counts": {}}

    def decode_token(tid):
        return tokenizer.decode([tid])

    def get_leading_char(tid):
        """Get the first non-whitespace character of a token."""
        decoded = decode_token(tid)
        stripped = decoded.lstrip()
        return stripped[0] if stripped else None

    def is_value_start(ch):
        """Could this character start a JSON value?"""
        if ch is None:
            return False
        return ch in '"{}[]-0123456789tfn'

    def record_rule(rule_name):
        stats["rule_counts"][rule_name] = stats["rule_counts"].get(rule_name, 0) + 1

    def hook(step, x, logits):
        gen_region = x[0, prompt_len:]
        gen_len = len(gen_region)
        remasked_this_step = 0

        positions_to_remask = set()

        for pos in range(gen_len - 1):
            tid = gen_region[pos].item()
            next_tid = gen_region[pos + 1].item()

            # only check pairs where BOTH positions are unmasked
            if tid == mask_token_id or next_tid == mask_token_id:
                continue

            text = decode_token(tid)
            next_text = decode_token(next_tid)
            stripped = text.strip()
            next_stripped = next_text.lstrip()

            if not stripped or not next_stripped:
                continue

            # last char of current token, first char of next token
            cur_end = stripped[-1]
            next_start = next_stripped[0]

            # Rule 1: { followed directly by non-quote, non-bracket
            if cur_end == '{' and next_start not in '"{}[]':
                positions_to_remask.add(pos + 1)
                record_rule("after_{_not_quote")

            # Rule 2: [ followed directly by invalid value start
            if cur_end == '[' and not is_value_start(next_start) and next_start != ']':
                positions_to_remask.add(pos + 1)
                record_rule("after_[_not_value")

            # Rule 3: : followed directly by non-value
            if cur_end == ':' and not is_value_start(next_start):
                positions_to_remask.add(pos + 1)
                record_rule("after_:_not_value")

            # Rule 4: , followed directly by non-key/non-value start
            if cur_end == ',' and next_start not in '"{[':
                positions_to_remask.add(pos + 1)
                record_rule("after_,_not_key")

            # Rule 5: doubled colon ::
            if cur_end == ':' and next_start == ':':
                positions_to_remask.add(pos + 1)
                record_rule("doubled_colon")

            # Rule 6: doubled quote "" after { or : or , or [
            if cur_end == '"' and next_start == '"':
                # check if prev unmasked char is a delimiter
                prev_end = None
                if pos > 0:
                    prev_tid = gen_region[pos - 1].item()
                    if prev_tid != mask_token_id:
                        prev_stripped = decode_token(prev_tid).strip()
                        if prev_stripped:
                            prev_end = prev_stripped[-1]
                if prev_end is not None and prev_end in '{:,[':
                    positions_to_remask.add(pos)
                    record_rule("doubled_quote")

        # apply re-masking
        if positions_to_remask:
            for pos in positions_to_remask:
                x[0, prompt_len + pos] = mask_token_id
                remasked_this_step += 1
            stats["total_remasked"] += remasked_this_step
            stats["steps_with_remasking"] += 1

        return x

    return hook, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/boundary")
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
        total_remasked = 0
        total_steps_with_remasking = 0
        aggregate_rule_counts = {}
        instances = []

        for idx in tqdm(range(len(ds)), desc=f"steps={step_budget}"):
            row = ds[idx]
            inputs = format_prompt(row["schema"], row["input"], tokenizer)
            input_ids = inputs.input_ids.to(args.device)
            attention_mask = inputs.attention_mask.to(args.device)
            prompt_len = input_ids.shape[1]

            hook, stats = make_boundary_hook(tokenizer, mask_token_id, prompt_len)

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
            for rule, count in stats["rule_counts"].items():
                aggregate_rule_counts[rule] = aggregate_rule_counts.get(rule, 0) + count

            instances.append({
                "instance_id": row["instance_id"],
                "syntax_valid": check["syntax_valid"],
                "schema_valid": check["schema_valid"],
                "error": check["error"],
                "remasked_tokens": stats["total_remasked"],
                "steps_with_remasking": stats["steps_with_remasking"],
                "rule_counts": stats["rule_counts"],
            })

        n = len(ds)
        summary = {
            "steps": step_budget,
            "syntax_rate": syntax_ok / n,
            "schema_rate": schema_ok / n,
            "avg_remasked": total_remasked / n,
            "avg_steps_with_remasking": total_steps_with_remasking / n,
            "aggregate_rule_counts": aggregate_rule_counts,
        }
        all_results[step_budget] = {"summary": summary, "instances": instances}

        print(f"  Syntax: {syntax_ok}/{n} ({syntax_ok/n:.1%})")
        print(f"  Schema: {schema_ok}/{n} ({schema_ok/n:.1%})")
        print(f"  Avg tokens re-masked: {total_remasked/n:.1f}")
        print(f"  Avg steps with re-masking: {total_steps_with_remasking/n:.1f}")
        print(f"  Rule trigger counts:")
        for rule, count in sorted(aggregate_rule_counts.items(), key=lambda x: -x[1]):
            print(f"    {rule}: {count}")

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'Steps':>6} | {'Syntax':>8} | {'Schema':>8} | {'Remasked':>10} | {'Remask steps':>13}")
    print("-" * 60)
    for step_budget in args.steps:
        s = all_results[step_budget]["summary"]
        print(f"{s['steps']:>6} | {s['syntax_rate']:>7.1%} | {s['schema_rate']:>7.1%} | {s['avg_remasked']:>9.1f} | {s['avg_steps_with_remasking']:>12.1f}")


if __name__ == "__main__":
    main()