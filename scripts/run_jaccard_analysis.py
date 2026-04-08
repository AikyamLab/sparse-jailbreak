#!/usr/bin/env python3
"""
run_jaccard_analysis.py — Compute Jaccard similarity from pre-extracted SAE features.

Reads the per-source .pkl files produced by run_feature_extraction.py and computes:
  - Within-attack pairwise Jaccard similarity (adversarial vs adversarial)
  - Attack-vs-random Jaccard similarity (adversarial vs random baselines)
  - Per-family breakdown (within vs across model families)

Produces the data for Figure 3 and Tables 14-15.

Usage:
  python scripts/run_jaccard_analysis.py \\
      --feature_dir gemma_9b_sae_features/ \\
      --output_dir results/jaccard/ \\
      --top_k 100

  # Specify which sources are adversarial vs random
  python scripts/run_jaccard_analysis.py \\
      --feature_dir gemma_9b_sae_features/ \\
      --random_sources text_lowercase text_uppercase mixed_case numbers unicode \\
      --output_dir results/jaccard/
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_feature_file(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_top_k_indices(features: torch.Tensor, k: int) -> List[Set[int]]:
    """Get top-k activated feature indices for each sample."""
    top_k_sets = []
    for i in range(features.shape[0]):
        vals, indices = torch.topk(features[i], min(k, features.shape[1]))
        # Only include features that are actually active (> 0)
        active = indices[vals > 0]
        top_k_sets.append(set(active.cpu().tolist()))
    return top_k_sets


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pairwise_jaccard(sets: List[Set[int]]) -> List[float]:
    """Compute all pairwise Jaccard similarities."""
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            scores.append(jaccard(sets[i], sets[j]))
    return scores


def cross_jaccard(sets_a: List[Set[int]], sets_b: List[Set[int]]) -> List[float]:
    """Compute cross-group Jaccard similarities."""
    scores = []
    for a in sets_a:
        for b in sets_b:
            scores.append(jaccard(a, b))
    return scores


# Default random baseline source name patterns
DEFAULT_RANDOM_PATTERNS = [
    "alphanum", "lower", "mixed", "numbers", "unicode",
    "text_lowercase", "text_uppercase", "mixed_case",
    "random", "baseline",
]


def is_random_source(name: str, random_sources: List[str] = None) -> bool:
    """Determine if a source name is a random baseline."""
    name_lower = name.lower()
    if random_sources:
        return any(r.lower() in name_lower for r in random_sources)
    return any(p in name_lower for p in DEFAULT_RANDOM_PATTERNS)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Jaccard similarity from extracted SAE features.")
    parser.add_argument("--feature_dir", required=True,
                        help="Directory containing *_sae_features.pkl files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for output JSON")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top features to use for Jaccard")
    parser.add_argument("--random_sources", nargs="*", default=None,
                        help="Names of random baseline sources (default: auto-detect)")
    args = parser.parse_args()

    print("=" * 60)
    print("JACCARD SIMILARITY ANALYSIS")
    print("=" * 60)
    print(f"Feature dir: {args.feature_dir}")
    print(f"Top-k: {args.top_k}")

    # Load all feature files
    feature_files = sorted([
        f for f in os.listdir(args.feature_dir)
        if f.endswith("_sae_features.pkl")
    ])

    if not feature_files:
        print(f"No feature files found in {args.feature_dir}")
        return

    print(f"\nFound {len(feature_files)} feature files:")

    sources = {}  # source_name -> {'features': tensor, 'metadata': dict, 'top_k_sets': list}
    for fname in feature_files:
        data = load_feature_file(os.path.join(args.feature_dir, fname))
        source_name = data["source"]
        features = data["features"]
        top_k_sets = get_top_k_indices(features, args.top_k)

        sources[source_name] = {
            "features": features,
            "metadata": data.get("metadata", {}),
            "top_k_sets": top_k_sets,
            "n_samples": features.shape[0],
        }
        print(f"  {source_name}: {features.shape[0]} samples, "
              f"dim={features.shape[1]}")

    # Classify sources
    adv_sources = {}
    rnd_sources = {}
    for name, info in sources.items():
        if is_random_source(name, args.random_sources):
            rnd_sources[name] = info
        else:
            adv_sources[name] = info

    print(f"\nAdversarial sources ({len(adv_sources)}): {list(adv_sources.keys())}")
    print(f"Random sources ({len(rnd_sources)}): {list(rnd_sources.keys())}")

    results = {
        "top_k": args.top_k,
        "feature_dir": args.feature_dir,
        "num_adversarial_sources": len(adv_sources),
        "num_random_sources": len(rnd_sources),
    }

    # 1. Within-attack similarity (all adversarial pairs)
    print("\n" + "-" * 40)
    print("WITHIN-ATTACK SIMILARITY")
    print("-" * 40)

    all_adv_sets = []
    adv_source_labels = []
    for name, info in adv_sources.items():
        all_adv_sets.extend(info["top_k_sets"])
        adv_source_labels.extend([name] * len(info["top_k_sets"]))

    within_scores = pairwise_jaccard(all_adv_sets) if len(all_adv_sets) > 1 else []
    within_arr = np.array(within_scores) if within_scores else np.array([0.0])

    results["within_attack"] = {
        "mean": float(within_arr.mean()),
        "std": float(within_arr.std()),
        "n_pairs": len(within_scores),
        "n_samples": len(all_adv_sets),
    }
    print(f"  {results['within_attack']['mean']:.4f} ± "
          f"{results['within_attack']['std']:.4f} "
          f"({len(within_scores)} pairs)")

    # 2. Within-random similarity
    all_rnd_sets = []
    for info in rnd_sources.values():
        all_rnd_sets.extend(info["top_k_sets"])

    within_rnd_scores = pairwise_jaccard(all_rnd_sets) if len(all_rnd_sets) > 1 else []
    within_rnd_arr = np.array(within_rnd_scores) if within_rnd_scores else np.array([0.0])

    results["within_random"] = {
        "mean": float(within_rnd_arr.mean()),
        "std": float(within_rnd_arr.std()),
        "n_pairs": len(within_rnd_scores),
    }
    print(f"  Within-random: {results['within_random']['mean']:.4f} ± "
          f"{results['within_random']['std']:.4f}")

    # 3. Attack-vs-random
    print("\n" + "-" * 40)
    print("ATTACK VS RANDOM")
    print("-" * 40)

    cross_scores = cross_jaccard(all_adv_sets, all_rnd_sets) if (all_adv_sets and all_rnd_sets) else []
    cross_arr = np.array(cross_scores) if cross_scores else np.array([0.0])

    results["attack_vs_random"] = {
        "mean": float(cross_arr.mean()),
        "std": float(cross_arr.std()),
        "n_pairs": len(cross_scores),
    }
    print(f"  {results['attack_vs_random']['mean']:.4f} ± "
          f"{results['attack_vs_random']['std']:.4f} "
          f"({len(cross_scores)} pairs)")

    # 4. Ratio
    ratio = float(within_arr.mean() / max(cross_arr.mean(), 1e-9))
    results["ratio"] = ratio
    print(f"\n  Ratio (within/cross): {ratio:.1f}×")

    # 5. Per-source breakdown
    print("\n" + "-" * 40)
    print("PER-SOURCE BREAKDOWN")
    print("-" * 40)

    per_source = {}
    for name, info in adv_sources.items():
        within = pairwise_jaccard(info["top_k_sets"]) if len(info["top_k_sets"]) > 1 else []
        cross = cross_jaccard(info["top_k_sets"], all_rnd_sets) if all_rnd_sets else []

        w_arr = np.array(within) if within else np.array([0.0])
        c_arr = np.array(cross) if cross else np.array([0.0])

        per_source[name] = {
            "within_attack_mean": float(w_arr.mean()),
            "within_attack_std": float(w_arr.std()),
            "attack_vs_random_mean": float(c_arr.mean()),
            "attack_vs_random_std": float(c_arr.std()),
            "ratio": float(w_arr.mean() / max(c_arr.mean(), 1e-9)),
            "n_samples": len(info["top_k_sets"]),
        }
        print(f"  {name}: within={w_arr.mean():.4f}, "
              f"vs_random={c_arr.mean():.4f}, "
              f"ratio={per_source[name]['ratio']:.1f}×")

    results["per_source"] = per_source

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_path = os.path.join(args.output_dir, f"jaccard_analysis_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
