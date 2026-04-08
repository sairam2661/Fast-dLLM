#!/usr/bin/env python3
"""
Experiment 10: Are boundary errors caused by temporal separation or factorization?

Analyze existing denoising histories. For each invalid output, find the
error position (from json.loads), then check whether the bad token and
its neighbor were unmasked in the same step or different steps.

Usage:
  python eval/eval_json_temporal.py results/history_64/history_data.json
"""

import json
import sys
import transformers
import torch

def load_tokenizer(model_path="Dream-org/Dream-v0-Instruct-7B"):
    return transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def find_unmask_step(history, pos, mask_token_id):
    """Find the step at which position pos was first unmasked."""
    for step_idx, seq in enumerate(history):
        if seq[pos] != mask_token_id:
            return step_idx
    return None


def analyze(history_path):
    with open(history_path) as f:
        results = json.load(f)

    tokenizer = load_tokenizer()
    mask_token_id = tokenizer.mask_token_id

    invalid = [r for r in results if not r["final_syntax_valid"]]
    print(f"Total: {len(results)}, Invalid: {len(invalid)}\n")

    # we need the raw history token ids — check if they're in the data
    # The history experiment saved step_summary but not raw token ids.
    # We need to re-derive from the generated text and error position.

    # Actually, let's check if history_ids were saved
    if invalid and "step_summary" in invalid[0]:
        # We have step summaries with newly_unmasked details per step
        # but not full token-level history. We can use the newly_unmasked_details
        # to build a map of (position -> step_unmasked)
        pass

    same_step_count = 0
    diff_step_count = 0
    unclear_count = 0
    details = []

    for r in invalid:
        instance_id = r["instance_id"]
        gen_text = r["generated_text"]

        # find the json error position
        # try to extract and parse to get the error location
        text = gen_text.strip()
        start_idx = None
        for i, ch in enumerate(text):
            if ch in '{[':
                start_idx = i
                break

        if start_idx is None:
            unclear_count += 1
            continue

        json_text = text[start_idx:]
        try:
            json.loads(json_text)
            unclear_count += 1
            continue
        except json.JSONDecodeError as e:
            error_char_pos = e.pos  # character position in json_text
        except Exception:
            unclear_count += 1
            continue

        # build position -> unmask step map from step_summary
        unmask_step_map = {}
        for s in r["step_summary"]:
            step = s["step"]
            # step_summary doesn't have per-position details in compact form
            # but the full data might have newly_unmasked_details
            # Let's check if we have it
            pass

        # We need the raw history to do this properly.
        # Let's try a different approach: tokenize the generated text
        # and find which token positions correspond to the error.
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)

        # map character position to token position
        char_to_tok = {}
        current_char = 0
        for tok_idx, tid in enumerate(gen_tokens):
            decoded = tokenizer.decode([tid])
            for c_offset in range(len(decoded)):
                char_to_tok[current_char + c_offset] = tok_idx
            current_char += len(decoded)

        # the error is at character position error_char_pos in json_text
        # which is at error_char_pos + start_idx in gen_text
        abs_char_pos = error_char_pos + start_idx
        error_tok_pos = char_to_tok.get(abs_char_pos)
        if error_tok_pos is None:
            # try nearby
            for delta in range(-2, 3):
                error_tok_pos = char_to_tok.get(abs_char_pos + delta)
                if error_tok_pos is not None:
                    break

        if error_tok_pos is None:
            unclear_count += 1
            continue

        # get the neighbor token position
        prev_tok_pos = error_tok_pos - 1 if error_tok_pos > 0 else None
        next_tok_pos = error_tok_pos + 1 if error_tok_pos < len(gen_tokens) - 1 else None

        # now find unmask steps from the step_summary newly_unmasked data
        # The step_summary has newly_unmasked_count but not positions
        # unless the full data was saved.

        # Let's try to reconstruct from step_summary
        # We know how many tokens were unmasked at each step
        # but not which ones. We need the raw history for exact analysis.

        # Print what we can determine
        error_context_start = max(0, abs_char_pos - 15)
        error_context_end = min(len(gen_text), abs_char_pos + 15)
        error_char = gen_text[abs_char_pos] if abs_char_pos < len(gen_text) else '?'

        detail = {
            "instance_id": instance_id,
            "error_char_pos": abs_char_pos,
            "error_char": error_char,
            "error_tok_pos": error_tok_pos,
            "context": gen_text[error_context_start:error_context_end],
        }

        # If the error token and its predecessor are known,
        # show their token strings
        if error_tok_pos is not None and error_tok_pos < len(gen_tokens):
            detail["error_token"] = tokenizer.decode([gen_tokens[error_tok_pos]])
        if prev_tok_pos is not None and prev_tok_pos < len(gen_tokens):
            detail["prev_token"] = tokenizer.decode([gen_tokens[prev_tok_pos]])

        details.append(detail)

    # Since we don't have per-position unmask timing in the saved data,
    # print what we can and note what additional data we need
    print("="*70)
    print("Error position analysis (from token-level inspection):")
    print("="*70)

    for d in details:
        print(f"\n  {d['instance_id']}:")
        print(f"    Error at char {d['error_char_pos']}: '{d['error_char']}'")
        print(f"    Context: ...{repr(d['context'])}...")
        if 'prev_token' in d:
            print(f"    Token before error: {repr(d['prev_token'])}")
        if 'error_token' in d:
            print(f"    Error token:        {repr(d['error_token'])}")

    print(f"\n{'='*70}")
    print("NOTE: To determine same-step vs different-step unmasking,")
    print("we need full per-position history. Re-run the history experiment")
    print("with output_history=True and save the raw token IDs per step.")
    print(f"Analyzed {len(details)} error positions, {unclear_count} unclear.")
    print(f"{'='*70}")

    # Better approach: re-run history collection saving full position data
    print(f"\nTo get the temporal separation data, re-run:")
    print(f"  python eval/eval_json_temporal.py --collect --steps 64 --num_instances 50")


def collect_and_analyze(args):
    """Re-collect histories with full position tracking, then analyze."""
    import time
    import types
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from model.configuration_dream import DreamConfig
    from model.modeling_dream import DreamModel
    from model.generation_utils import DreamGenerationMixin

    model_path = args.model_path
    model = DreamModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(args.device)
    model.diffusion_generate = types.MethodType(
        DreamGenerationMixin.diffusion_generate, model
    )
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    mask_token_id = tokenizer.mask_token_id

    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    if args.num_instances < len(ds):
        ds = ds.select(range(args.num_instances))

    step_budget = args.steps[0]
    print(f"Collecting histories: {len(ds)} instances, {step_budget} steps\n")

    same_step = 0
    diff_step = 0
    unclear = 0
    all_details = []

    for idx in tqdm(range(len(ds)), desc="Collecting"):
        row = ds[idx]

        system_msg = (
            "You are a helpful assistant that answers in JSON. "
            "Here is the JSON schema you must adhere to:\n"
            f"<schema>\n{row['schema']}\n</schema>"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": row["input"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(args.device)
        attention_mask = inputs.attention_mask.to(args.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                output_history=True,
                return_dict_in_generate=True,
                steps=step_budget,
                temperature=0.2,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.0,
            )

        gen_text = tokenizer.decode(
            output.sequences[0][prompt_len:].tolist()
        ).split(tokenizer.eos_token)[0]

        # check validity
        try:
            # extract json
            text = gen_text.strip()
            start_idx = None
            for i, ch in enumerate(text):
                if ch in '{[':
                    start_idx = i
                    break
            if start_idx is not None:
                json.loads(text[start_idx:])
                continue  # valid, skip
        except json.JSONDecodeError as e:
            error_char_pos = e.pos
        except:
            unclear += 1
            continue

        if start_idx is None:
            unclear += 1
            continue

        # build per-position unmask step map from history
        history = output.history if output.history else []
        if not history:
            unclear += 1
            continue

        # map: gen_region position -> step when first unmasked
        gen_len = len(history[0][0]) - prompt_len
        pos_to_step = {}
        for step_idx, h in enumerate(history):
            gen_region = h[0][prompt_len:].tolist()
            for pos in range(gen_len):
                if pos not in pos_to_step and gen_region[pos] != mask_token_id:
                    pos_to_step[pos] = step_idx

        # tokenize gen_text to find error token position
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)

        # map char pos -> token pos
        char_to_tok = {}
        current_char = 0
        for tok_idx, tid in enumerate(gen_tokens):
            decoded = tokenizer.decode([tid])
            for c_off in range(len(decoded)):
                char_to_tok[current_char + c_off] = tok_idx
            current_char += len(decoded)

        abs_char_pos = error_char_pos + start_idx
        error_tok_pos = char_to_tok.get(abs_char_pos)
        if error_tok_pos is None:
            for delta in range(-3, 4):
                error_tok_pos = char_to_tok.get(abs_char_pos + delta)
                if error_tok_pos is not None:
                    break

        if error_tok_pos is None or error_tok_pos >= gen_len:
            unclear += 1
            continue

        prev_tok_pos = error_tok_pos - 1 if error_tok_pos > 0 else None

        error_step = pos_to_step.get(error_tok_pos)
        prev_step = pos_to_step.get(prev_tok_pos) if prev_tok_pos is not None else None

        if error_step is not None and prev_step is not None:
            if error_step == prev_step:
                same_step += 1
                temporal = "SAME step"
            else:
                diff_step += 1
                temporal = f"DIFFERENT steps (prev={prev_step}, error={error_step}, gap={abs(error_step - prev_step)})"
        else:
            unclear += 1
            temporal = "unknown"

        error_context_start = max(0, abs_char_pos - 20)
        error_context_end = min(len(gen_text), abs_char_pos + 20)

        detail = {
            "instance_id": row["instance_id"],
            "error_tok_pos": error_tok_pos,
            "prev_tok_pos": prev_tok_pos,
            "error_step": error_step,
            "prev_step": prev_step,
            "temporal": temporal,
            "prev_token": tokenizer.decode([gen_tokens[prev_tok_pos]]) if prev_tok_pos is not None and prev_tok_pos < len(gen_tokens) else None,
            "error_token": tokenizer.decode([gen_tokens[error_tok_pos]]) if error_tok_pos < len(gen_tokens) else None,
            "context": gen_text[error_context_start:error_context_end],
        }
        all_details.append(detail)

        print(f"\n  {row['instance_id']}: {temporal}")
        print(f"    Prev token: {repr(detail['prev_token'])} (step {prev_step})")
        print(f"    Error token: {repr(detail['error_token'])} (step {error_step})")
        print(f"    Context: ...{repr(detail['context'])}...")

    print(f"\n{'='*70}")
    print(f"RESULTS: Temporal Separation Analysis")
    print(f"{'='*70}")
    print(f"  Same step (factorization issue):     {same_step}")
    print(f"  Different steps (temporal separation): {diff_step}")
    print(f"  Unclear/missing data:                 {unclear}")
    total = same_step + diff_step
    if total > 0:
        print(f"\n  Same step:      {same_step}/{total} ({same_step/total:.1%})")
        print(f"  Different step: {diff_step}/{total} ({diff_step/total:.1%})")

    # save
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "same_step": same_step,
        "diff_step": diff_step,
        "unclear": unclear,
        "details": all_details,
    }
    out_path = os.path.join(args.output_dir, "temporal_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("history_path", nargs="?", default=None,
                        help="Path to history_data.json (for analysis-only mode)")
    parser.add_argument("--collect", action="store_true",
                        help="Re-collect histories with full position tracking")
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--steps", type=int, nargs="+", default=[64])
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/temporal")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed) if 'torch' in dir() else None
    import torch
    torch.manual_seed(args.seed)

    if args.collect:
        collect_and_analyze(args)
    elif args.history_path:
        analyze(args.history_path)
    else:
        print("Provide a history_data.json path or use --collect")


if __name__ == "__main__":
    main()