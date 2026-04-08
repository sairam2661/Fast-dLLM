#!/usr/bin/env python3
"""
Oracle experiment: pre-fill JSON structural tokens from the reference output
before denoising, then let the model fill in content tokens only.

If this dramatically improves validity, then "structure first" matters and
building a parser-based scorer (Tier 2) is worthwhile.

Usage:
  python eval/eval_json_oracle.py \
      --steps 32 64 128 256 \
      --num_instances 50 \
      --output_dir results/oracle
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


def find_structural_positions(ref_output, tokenizer):
    """Tokenize the reference output and find which token positions are structural."""
    ref_ids = tokenizer.encode(ref_output, add_special_tokens=False)
    structural = []
    for i, tok_id in enumerate(ref_ids):
        decoded = tokenizer.decode([tok_id]).strip()
        if decoded and decoded[0] in JSON_STRUCTURAL_CHARS:
            structural.append((i, tok_id))
    return ref_ids, structural

def generate_with_oracle(model, tokenizer, inputs, ref_output, steps,
                         max_new_tokens=256, temperature=0.2, device="cuda",
                         max_pins=None):
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    ref_ids, structural_positions = find_structural_positions(ref_output, tokenizer)

    if max_pins is not None:
        structural_positions = structural_positions[:max_pins]

    def skeleton_hook(step, x, logits):
        for offset, tok_id in structural_positions:
            pos = prompt_len + offset
            if pos < x.shape[1]:
                x[0, pos] = tok_id
        return x

    t0 = time.time()
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            generation_tokens_hook_func=skeleton_hook,
        )
    elapsed = time.time() - t0

    gen_text = tokenizer.decode(
        output.sequences[0][prompt_len:].tolist()
    ).split(tokenizer.eos_token)[0]

    return {
        "text": gen_text,
        "prompt_len": prompt_len,
        "time_seconds": elapsed,
        "pinned_tokens": len(structural_positions),
        "total_ref_tokens": len(ref_ids),
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
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


def check_output(gen_text, schema_str):
    extracted = extract_json(gen_text)
    result = {"has_json": extracted is not None, "syntax_valid": False,
              "schema_valid": False, "extracted": extracted, "error": None}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/oracle")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_pins", type=int, default=None,
                    help="Max structural tokens to pin. None = all (full oracle)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances")

    model, tokenizer = load_model(args.model_path, args.device)

    all_results = {"config": vars(args), "step_results": {}}

    for step_budget in args.steps:
        print(f"\n{'='*60}")
        print(f"Steps: {step_budget}")
        print(f"{'='*60}")

        instances = []
        syntax_ok = 0
        schema_ok = 0
        total_time = 0.0

        for idx in tqdm(range(len(ds)), desc=f"steps={step_budget}"):
            row = ds[idx]
            instance_id = row["instance_id"]
            schema_str = row["schema"]
            input_text = row["input"]
            ref_output = row["output"]

            inputs = format_prompt(schema_str, input_text, tokenizer)
            gen = generate_with_oracle(
                model, tokenizer, inputs, ref_output,
                steps=step_budget,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=args.device,
                max_pins=args.max_pins,
            )

            check = check_output(gen["text"], schema_str)

            if check["syntax_valid"]:
                syntax_ok += 1
            if check["schema_valid"]:
                schema_ok += 1
            total_time += gen["time_seconds"]

            instances.append({
                "instance_id": instance_id,
                "syntax_valid": check["syntax_valid"],
                "schema_valid": check["schema_valid"],
                "error": check["error"],
                "pinned_tokens": gen["pinned_tokens"],
                "total_ref_tokens": gen["total_ref_tokens"],
                "time_seconds": gen["time_seconds"],
                "generated_text": gen["text"][:500],
            })

        n = len(ds)
        summary = {
            "steps": step_budget,
            "num_instances": n,
            "syntax_valid_rate": syntax_ok / n,
            "schema_valid_rate": schema_ok / n,
            "avg_time_seconds": total_time / n,
        }

        print(f"\nResults for steps={step_budget}:")
        print(f"  Syntax valid:  {syntax_ok}/{n} ({summary['syntax_valid_rate']:.1%})")
        print(f"  Schema valid:  {schema_ok}/{n} ({summary['schema_valid_rate']:.1%})")

        all_results["step_results"][step_budget] = {
            "summary": summary,
            "instances": instances,
        }

    out_path = os.path.join(args.output_dir, "json_eval_oracle.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'Steps':>6} | {'Syntax':>8} | {'Schema':>8} | {'Time/inst':>10}")
    print("-" * 42)
    for step_budget in args.steps:
        s = all_results["step_results"][step_budget]["summary"]
        print(f"{s['steps']:>6} | {s['syntax_valid_rate']:>7.1%} | {s['schema_valid_rate']:>7.1%} | {s['avg_time_seconds']:>9.2f}s")


if __name__ == "__main__":
    main()