#!/usr/bin/env python3
"""
Evaluation: Dream-Instruct with re-prompting on failure.

Runs unconstrained generation, checks syntax/schema validity.
On failure, re-prompts with the failed output asking for correction,
up to --max_retries times. Tracks how many attempts each instance needs.

Usage:
  python eval/eval_json_reprompt.py \
      --model_path Dream-org/Dream-v0-Instruct-7B \
      --steps 128 \
      --num_instances 50 \
      --max_retries 3
"""

import argparse
import json
import os
import re
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
from model.generation_utils_block import DreamGenerationMixin


def load_model(model_path, device="cuda"):
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


def format_retry_prompt(schema_str, input_text, failed_output, tokenizer):
    system_msg = (
        "You are a helpful assistant that answers in JSON. "
        "Here is the JSON schema you must adhere to:\n"
        f"<schema>\n{schema_str}\n</schema>"
    )
    retry_user_msg = (
        f"{input_text}\n\n"
        f"Your previous response was:\n"
        f"```\n{failed_output}\n```\n"
        f"This was not valid JSON conforming to the schema. "
        f"Please output only the corrected JSON, nothing else."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": retry_user_msg},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    return inputs


def generate(model, tokenizer, inputs, steps, max_new_tokens=256,
             temperature=0.2, device="cuda"):
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    prompt_len = input_ids.shape[1]

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
        )
    elapsed = time.time() - t0

    gen_text = tokenizer.decode(
        output.sequences[0][prompt_len:].tolist()
    ).split(tokenizer.eos_token)[0]

    return {
        "text": gen_text,
        "prompt_len": prompt_len,
        "gen_len": len(output.sequences[0]) - prompt_len,
        "time_seconds": elapsed,
    }


def extract_json(text):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

    torch.manual_seed(args.seed)
    output_dir = f"results/reprompt_{args.max_retries}retries"
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))
    print(f"Loaded {len(ds)} instances", flush=True)

    model, tokenizer = load_model(args.model_path, args.device)

    instances = []
    # Track results by attempt number
    passed_at_attempt = {i: 0 for i in range(args.max_retries + 1)}  # 0 = first try
    syntax_at_attempt = {i: 0 for i in range(args.max_retries + 1)}
    still_failing = 0
    total_time = 0.0
    total_attempts = 0

    for idx in tqdm(range(len(ds)), desc="Evaluating"):
        row = ds[idx]
        instance_id = row["instance_id"]
        schema_str = row["schema"]
        input_text = row["input"]

        attempt_history = []
        passed = False

        for attempt in range(args.max_retries + 1):
            if attempt == 0:
                inputs = format_prompt(schema_str, input_text, tokenizer)
            else:
                # Re-prompt with the failed output
                inputs = format_retry_prompt(
                    schema_str, input_text,
                    attempt_history[-1]["text"], tokenizer
                )

            gen = generate(
                model, tokenizer, inputs,
                steps=args.steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=args.device,
            )
            check = check_output(gen["text"], schema_str)
            total_time += gen["time_seconds"]
            total_attempts += 1

            attempt_record = {
                "attempt": attempt,
                "text": gen["text"][:500],
                "extracted": check["extracted"][:500] if check["extracted"] else None,
                "syntax_valid": check["syntax_valid"],
                "schema_valid": check["schema_valid"],
                "error": check["error"],
                "time_seconds": gen["time_seconds"],
            }
            attempt_history.append(attempt_record)

            if check["syntax_valid"] and attempt == 0:
                syntax_at_attempt[0] += 1
            elif check["syntax_valid"] and not any(a["syntax_valid"] for a in attempt_history[:-1]):
                syntax_at_attempt[attempt] += 1

            if check["schema_valid"]:
                passed_at_attempt[attempt] += 1
                passed = True
                break

            print(f"  [{idx}] attempt {attempt} failed: {check['error']}", flush=True)

        if not passed:
            still_failing += 1

        instance_result = {
            "instance_id": instance_id,
            "num_attempts": len(attempt_history),
            "final_schema_valid": passed,
            "attempts": attempt_history,
        }
        instances.append(instance_result)

        # Running stats
        n_done = idx + 1
        cum_passed = sum(passed_at_attempt.values())
        print(f"[{n_done}/{len(ds)}] "
              f"schema_valid={cum_passed}/{n_done} ({cum_passed/n_done:.0%}) | "
              f"attempt breakdown: {dict(passed_at_attempt)} | "
              f"still_failing={still_failing}",
              flush=True)

    # Final summary
    n = len(ds)
    cum_schema = 0
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: steps={args.steps}, max_retries={args.max_retries}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Attempt':>10} | {'Newly passed':>14} | {'Cumulative':>12}", flush=True)
    print("-" * 45, flush=True)
    for attempt in range(args.max_retries + 1):
        cum_schema += passed_at_attempt[attempt]
        label = "initial" if attempt == 0 else f"retry {attempt}"
        print(f"{label:>10} | {passed_at_attempt[attempt]:>6} ({passed_at_attempt[attempt]/n:.0%})"
              f"        | {cum_schema:>4} ({cum_schema/n:.0%})", flush=True)
    print(f"{'failed':>10} | {still_failing:>6} ({still_failing/n:.0%})        |", flush=True)
    print(f"\nTotal attempts: {total_attempts} ({total_attempts/n:.1f} avg/instance)", flush=True)
    print(f"Total time: {total_time:.1f}s ({total_time/n:.2f}s avg/instance)", flush=True)

    all_results = {
        "config": vars(args),
        "summary": {
            "passed_at_attempt": passed_at_attempt,
            "still_failing": still_failing,
            "total_attempts": total_attempts,
            "total_time": total_time,
        },
        "instances": instances,
    }
    out_path = os.path.join(output_dir, "reprompt_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()