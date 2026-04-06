#!/usr/bin/env python3
"""
run_eval.py — Unified suffix evaluation (generation) for all model families.

Evaluates pre-computed adversarial suffixes on a target model (base or SAE).
Produces JSONL output for downstream ASR computation.

Examples:
  # Evaluate LLaMA 8B SAE model on all available suffixes
  python scripts/run_eval.py --model_config llama_8b_sae \\
      --suffix_file suffixes/sampled_lightweight_suffixes.pkl \\
      --output_dir data/generation/

  # Evaluate Gemma 9B base model
  python scripts/run_eval.py --model_config gemma_9b_base \\
      --suffix_file suffixes/sampled_lightweight_suffixes.pkl \\
      --output_dir data/generation/
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_model
from src.utils import (
    load_object,
    load_harmbench,
    save_jsonl,
    pipeline_generate,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_completed_pairs(output_path: str) -> set:
    """Load (prompt_id, suffix_source) pairs already in the output file."""
    pairs = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    pairs.add((entry["prompt_id"], entry["suffix_source"]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Unified suffix evaluation.")

    parser.add_argument("--model_config", required=True,
                        help="Key from configs/models.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", default=None)

    parser.add_argument("--suffix_file", required=True,
                        help="Path to suffix .pkl (dict of {source: {behaviorID: {result: {best_string: ...}}}})")
    parser.add_argument("--suffix_sources", nargs="*", default=None,
                        help="Restrict to specific suffix sources (default: all)")

    parser.add_argument("--harmbench_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Apply chat template before generation")

    args = parser.parse_args()

    # ── Load configs ──
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_cfgs = load_yaml(os.path.join(project_root, "configs/models.yaml"))

    model_cfg = model_cfgs[args.model_config]
    model_cfg["device"] = args.device
    if args.cache_dir:
        model_cfg["cache_dir"] = args.cache_dir

    # ── Load model ──
    print(f"Loading model config: {args.model_config}")
    model, tokenizer = load_model(model_cfg)

    # ── Load data ──
    base_dict, _ = load_harmbench(args.harmbench_dir)
    suffixes = load_object(args.suffix_file)

    suffix_sources = args.suffix_sources or list(suffixes.keys())
    print(f"Suffix sources: {suffix_sources}")

    # ── Output path ──
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(
        args.output_dir,
        f"eval_{args.model_config}_{timestamp}.jsonl"
    )

    completed = load_completed_pairs(output_path)
    print(f"Already completed: {len(completed)} pairs")

    # ── Generate ──
    count = 0
    for suffix_source in suffix_sources:
        if suffix_source not in suffixes:
            print(f"Warning: suffix source '{suffix_source}' not found, skipping")
            continue

        source_suffixes = suffixes[suffix_source]
        print(f"\nProcessing suffix source: {suffix_source}")

        for key, val in tqdm(base_dict.items(), desc=suffix_source):
            if key not in source_suffixes:
                continue
            if (key, suffix_source) in completed:
                continue

            message = val["Behavior"]
            suffix = source_suffixes[key]["result"]["best_string"]
            prompt_text = message + suffix

            try:
                response = pipeline_generate(
                    model, tokenizer, prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    use_chat_template=args.use_chat_template,
                )

                entry = {
                    "prompt_id": key,
                    "suffix_source": suffix_source,
                    "eval_model": model_cfg["model_name"],
                    "model_config": args.model_config,
                    "sae_type": model_cfg.get("sae_type", "none"),
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {
                        "temperature": args.temperature,
                        "max_tokens": args.max_new_tokens,
                        "do_sample": args.do_sample,
                    },
                    "base_prompt": message,
                    "suffix": suffix,
                    "full_prompt": prompt_text,
                    "generation": response,
                }

                save_jsonl([entry], output_path, append=True)
                count += 1

            except Exception as e:
                print(f"Error on {key}/{suffix_source}: {e}")
                continue

    print(f"\nDone. {count} generations saved to {output_path}")


if __name__ == "__main__":
    main()
