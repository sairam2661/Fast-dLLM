#!/usr/bin/env python3
"""
Baseline evaluation: Dream-Instruct on eth-sri/json-mode-eval-extended.
Measures JSON syntax validity and schema conformance across step budgets.

Usage:
  python eval/eval_json.py \
      --model_path Dream-org/Dream-v0-Instruct-7B \
      --steps 32 64 128 256 \
      --num_instances 50 \
      --output_dir results/baseline
"""

import argparse
import json
import os
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


def load_model(model_path, device="cuda", use_grammar=False):
    if use_grammar:
        from model.generation_utils_block_grammar import DreamGenerationMixin
    else:
        from model.generation_utils_block import DreamGenerationMixin

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
    """Replicate the Mündler et al. prompt format."""
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
             grammar_scorer=None, grammar_lambda=0.0):
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    prompt_len = input_ids.shape[1]

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
            grammar_scorer=grammar_scorer,
            grammar_lambda=grammar_lambda,
        )
    elapsed = time.time() - t0

    gen_text = tokenizer.decode(
        output.sequences[0][prompt_len:].tolist()
    ).split(tokenizer.eos_token)[0]

    result = {
        "text": gen_text,
        "prompt_len": prompt_len,
        "gen_len": len(output.sequences[0]) - prompt_len,
        "time_seconds": elapsed,
    }

    if output_history and hasattr(output, "history"):
        result["history"] = [
            h[0][prompt_len:].tolist() for h in output.history
        ]

    return result


def extract_json(text):
    """Try to extract a JSON object/array from generated text."""
    import re
    text = text.strip()

    # try whole text
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # strip markdown fences
    md = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if md:
        try:
            json.loads(md.group(1).strip())
            return md.group(1).strip()
        except (json.JSONDecodeError, ValueError):
            pass

    # find first { or [ with matching close
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
    """Check syntax validity and schema conformance."""
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
    """Check if generated JSON matches reference after normalization."""
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
    parser.add_argument("--output_dir", type=str, default="results/baseline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grammar_lambda", type=float, default=0.0,
                        help="Grammar criticality weight. 0 = baseline (no grammar awareness)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances")

    # load model
    use_grammar = args.grammar_lambda > 0
    model, tokenizer = load_model(args.model_path, args.device, use_grammar=use_grammar)

    # init grammar scorer if needed
    grammar_scorer = None
    if use_grammar:
        from grammar.scorer import GrammarCriticalityScorer
        grammar_scorer = GrammarCriticalityScorer(tokenizer, device=args.device)
        print(f"Grammar-aware scheduling enabled (lambda={args.grammar_lambda})")

    all_results = {
        "config": vars(args),
        "step_results": {},
    }

    for step_budget in args.steps:
        print(f"\n{'='*60}")
        print(f"Steps: {step_budget}")
        print(f"{'='*60}")

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
                grammar_scorer=grammar_scorer,
                grammar_lambda=args.grammar_lambda,
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
            "syntax_valid_rate": syntax_ok / n,
            "schema_valid_rate": schema_ok / n,
            "exact_match_rate": exact_ok / n,
            "avg_time_seconds": total_time / n,
            "tokens_per_second": (args.max_new_tokens * n) / total_time if total_time > 0 else 0,
        }

        print(f"\nResults for steps={step_budget}:")
        print(f"  Syntax valid:  {syntax_ok}/{n} ({summary['syntax_valid_rate']:.1%})")
        print(f"  Schema valid:  {schema_ok}/{n} ({summary['schema_valid_rate']:.1%})")
        print(f"  Exact match:   {exact_ok}/{n} ({summary['exact_match_rate']:.1%})")
        print(f"  Avg time/inst: {summary['avg_time_seconds']:.2f}s")

        all_results["step_results"][step_budget] = {
            "summary": summary,
            "instances": instances,
        }

    # save
    out_path = os.path.join(args.output_dir, "json_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # print comparison table
    print(f"\n{'Steps':>6} | {'Syntax':>8} | {'Schema':>8} | {'Exact':>8} | {'Time/inst':>10}")
    print("-" * 50)
    for step_budget in args.steps:
        s = all_results["step_results"][step_budget]["summary"]
        print(f"{s['steps']:>6} | {s['syntax_valid_rate']:>7.1%} | {s['schema_valid_rate']:>7.1%} | {s['exact_match_rate']:>7.1%} | {s['avg_time_seconds']:>9.2f}s")


if __name__ == "__main__":
    main()