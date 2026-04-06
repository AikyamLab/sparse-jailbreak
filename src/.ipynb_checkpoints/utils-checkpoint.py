"""Shared utilities: serialization, data loading, generation helpers."""

import json
import os
import pickle
from typing import Any, Dict, List

import pandas as pd
import torch


def save_object(obj: Any, filename: str) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_object(filename: str) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_jsonl(entries: List[Dict], filepath: str, append: bool = False) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(filepath, mode) as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def read_jsonl(filepath: str) -> List[Dict]:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_harmbench(base_path: str):
    """
    Load HarmBench prompts and optimizer targets.

    Returns:
        base_dict: {BehaviorID: {"Behavior": str}}
        optimizer_targets: {BehaviorID: str}
    """
    prompts_csv = os.path.join(
        base_path, "HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    targets_json = os.path.join(
        base_path, "HarmBench/data/optimizer_targets/harmbench_targets_text.json"
    )
    base_dict = (
        pd.read_csv(prompts_csv)[["BehaviorID", "Behavior"]]
        .set_index("BehaviorID")
        .to_dict(orient="index")
    )
    with open(targets_json, "r", encoding="utf-8") as f:
        optimizer_targets = json.load(f)
    return base_dict, optimizer_targets


def load_completed_keys(completed_dir: str) -> set:
    """Scan a directory of .pkl result files and return all completed behavior keys."""
    keys = set()
    if not os.path.isdir(completed_dir):
        return keys
    for fname in os.listdir(completed_dir):
        if fname.endswith(".pkl") and ".ipynb" not in fname:
            try:
                d = load_object(os.path.join(completed_dir, fname))
                keys.update(d.keys())
            except Exception:
                pass
    return keys


def load_batch_keys(batch_path: str) -> set:
    """Load the set of behavior keys assigned to a specific batch."""
    return set(load_object(batch_path).keys())


def pipeline_generate(model, tokenizer, prompt: str, max_new_tokens: int = 1000,
                      do_sample: bool = False, temperature: float = 0.0,
                      use_chat_template: bool = True) -> str:
    """
    Generate a response with the (possibly SAE-augmented) model.

    Returns the assistant's response string.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    device = next(model.parameters()).device if hasattr(model, "parameters") else model.device
    input_ids = tokenizer(formatted, return_tensors="pt",
                          truncation=True, max_length=4096).to(device).input_ids

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample and temperature > 0:
        gen_kwargs["temperature"] = temperature

    if hasattr(model, "generate"):
        output_ids = model.generate(input_ids=input_ids, **gen_kwargs)
    else:
        output_ids = model.generate(input_ids, **gen_kwargs)

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Strip the prompt prefix
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if full_text.startswith(prompt_text):
        response = full_text[len(prompt_text):].strip()
    else:
        response = full_text

    return response
