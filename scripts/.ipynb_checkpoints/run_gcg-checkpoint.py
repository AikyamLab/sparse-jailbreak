#!/usr/bin/env python3
"""
run_gcg.py — Unified GCG suffix generation for all model families.

Supports:
  - All model/SAE pairs via --model_config (keys from configs/models.yaml)
  - Optional spectral gradient monitoring via --spectral_config
  - Batch parallelism via --batch_id / --batch_path
  - Checkpoint resumption via --output_dir

Examples:
  # LLaMA 8B SAE, 500 steps, no spectral
  python scripts/run_gcg.py --model_config llama_8b_sae --batch_id 1

  # Gemma 9B baseline with spectral monitoring
  python scripts/run_gcg.py --model_config gemma_9b_base --spectral_config spectral_full --batch_id 1

  # Gemma 9B SAE, 1500 steps (optimization budget ablation)
  python scripts/run_gcg.py --model_config gemma_9b_sae --attack_config extended --batch_id 1

  # Layer ablation
  python scripts/run_gcg.py --model_config llama_8b_sae_layer7 --batch_id 1
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogcg import GCGConfig
from src.models import load_model
from src.utils import (
    save_object,
    load_harmbench,
    load_completed_keys,
    load_batch_keys,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Unified GCG suffix generation.")

    # Model
    parser.add_argument("--model_config", required=True,
                        help="Key from configs/models.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", default=None)

    # Attack
    parser.add_argument("--attack_config", default="default",
                        help="Key from configs/attack.yaml (default | extended)")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override num_steps from attack config")

    # Spectral monitoring
    parser.add_argument("--spectral_config", default="spectral_off",
                        help="Key from configs/attack.yaml (spectral_full | spectral_light | spectral_off)")

    # Data
    parser.add_argument("--harmbench_dir", required=True,
                        help="Path containing HarmBench/ subdirectory")
    parser.add_argument("--batch_path", default=None,
                        help="Path to batch .pkl file")
    parser.add_argument("--batch_id", type=int, default=None,
                        help="Batch ID (used in output naming)")
    parser.add_argument("--behavior_subset", default=None,
                        help="Path to JSON list of BehaviorIDs to restrict to")

    # Output
    parser.add_argument("--output_dir", required=True,
                        help="Directory for results .pkl files")

    args = parser.parse_args()

    # ── Load configs ──
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_cfgs = load_yaml(os.path.join(project_root, "configs/models.yaml"))
    attack_cfgs = load_yaml(os.path.join(project_root, "configs/attack.yaml"))

    model_cfg = model_cfgs[args.model_config]
    model_cfg["device"] = args.device
    if args.cache_dir:
        model_cfg["cache_dir"] = args.cache_dir

    atk_cfg = attack_cfgs[args.attack_config]
    if args.num_steps is not None:
        atk_cfg["num_steps"] = args.num_steps

    spec_cfg = attack_cfgs.get(args.spectral_config, attack_cfgs["spectral_off"])

    # ── Load model ──
    print(f"Loading model config: {args.model_config}")
    model, tokenizer = load_model(model_cfg)

    # ── Load data ──
    base_dict, optimizer_targets = load_harmbench(args.harmbench_dir)

    # Filter by batch
    if args.batch_path:
        batch_keys = load_batch_keys(args.batch_path)
    else:
        batch_keys = set(base_dict.keys())

    # Filter by behavior subset
    if args.behavior_subset:
        with open(args.behavior_subset, "r") as f:
            subset = set(json.load(f))
        batch_keys = batch_keys & subset

    # Resumption
    os.makedirs(args.output_dir, exist_ok=True)
    completed_keys = load_completed_keys(args.output_dir)
    remaining = [k for k in base_dict if k in batch_keys and k not in completed_keys]

    print(f"Batch keys: {len(batch_keys)}  |  Completed: {len(completed_keys)}  "
          f"|  Remaining: {len(remaining)}")

    # ── GCG config ──
    gcg_config = GCGConfig(
        num_steps=atk_cfg["num_steps"],
        search_width=atk_cfg["search_width"],
        topk=atk_cfg["topk"],
        seed=atk_cfg["seed"],
        verbosity="WARNING",
        use_prefix_cache=atk_cfg["use_prefix_cache"],
    )

    # ── Decide whether to use spectral monitoring ──
    use_spectral = spec_cfg.get("compute_every_n_steps", 999999) < 999999

    if use_spectral:
        from src.gcg_spectral import SpectralMonitorConfig, run_with_spectral_monitoring
        spectral_config = SpectralMonitorConfig(
            compute_every_n_steps=spec_cfg["compute_every_n_steps"],
            store_gradients=spec_cfg["store_gradients"],
            store_gradients_every_n_steps=spec_cfg["store_gradients_every_n_steps"],
            store_projected=spec_cfg["store_projected"],
            compute_gradient_norms=spec_cfg["compute_gradient_norms"],
            compute_cosine_similarity=spec_cfg["compute_cosine_similarity"],
            compute_top_k_tokens=spec_cfg["compute_top_k_tokens"],
            top_k=spec_cfg["top_k"],
        )
        print(f"Spectral monitoring: ON (every {spectral_config.compute_every_n_steps} steps)")
    else:
        import nanogcg
        print("Spectral monitoring: OFF")

    # ── Run ──
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    batch_tag = f"_B{args.batch_id}" if args.batch_id is not None else ""
    out_file = os.path.join(
        args.output_dir,
        f"gcg_{args.model_config}_{timestamp}{batch_tag}.pkl"
    )

    gcg_results = {}

    for key in tqdm(remaining, desc=f"GCG {args.model_config}"):
        message = base_dict[key]["Behavior"]
        target = optimizer_targets[key]

        if use_spectral:
            result, spectral_history = run_with_spectral_monitoring(
                model, tokenizer, message, target,
                gcg_config=gcg_config, spectral_config=spectral_config,
            )
            gcg_results[key] = {
                "result": result,
                "spectral_history": spectral_history.to_dict(),
                "spectral_gradients": (spectral_history.get_gradient_tensors()
                                       if spec_cfg["store_gradients"] else []),
            }
        else:
            result = nanogcg.run(model, tokenizer, message, target, gcg_config)
            gcg_results[key] = {"result": result}

        # Common metadata
        gcg_results[key].update({
            "model_config": args.model_config,
            "model_name": model_cfg["model_name"],
            "sae_type": model_cfg.get("sae_type", "none"),
            "attack_config": args.attack_config,
            "num_steps": atk_cfg["num_steps"],
            "search_width": atk_cfg["search_width"],
            "topk": atk_cfg["topk"],
            "seed": atk_cfg["seed"],
        })

        print(f"  {key}: loss={result.best_loss:.4f}  suffix={result.best_string[:50]}")

        # Checkpoint after each prompt
        save_object(gcg_results, out_file)

    print(f"\nDone. {len(gcg_results)} results saved to {out_file}")


if __name__ == "__main__":
    main()