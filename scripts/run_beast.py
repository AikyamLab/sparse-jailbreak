#!/usr/bin/env python3
"""
run_beast.py — Unified BEAST attack for all model families.

BEAST is a beam-search-based jailbreak attack (no gradients). The SAE
intervention is applied during model forward passes used for scoring
candidates — gradients are not involved.

Supports:
  - All model/SAE pairs via --model_config (keys from configs/models.yaml)
  - Targeted (default) and untargeted attacks
  - Batch parallelism via --batch_id / --batch_path
  - Checkpoint resumption via --output_dir

Examples:
  # Gemma 2B base (no SAE)
  python scripts/run_beast.py --model_config gemma_2b_base \\
      --harmbench_dir /path/to/data/ --output_dir results/beast/gemma_2b_base/ --batch_id 1

  # Gemma 2B SAE
  python scripts/run_beast.py --model_config gemma_2b_sae \\
      --harmbench_dir /path/to/data/ --output_dir results/beast/gemma_2b_sae/ --batch_id 1

  # LLaMA 8B SAE
  python scripts/run_beast.py --model_config llama_8b_sae \\
      --harmbench_dir /path/to/data/ --output_dir results/beast/llama_8b_sae/ --batch_id 1
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_model
from src.utils import (
    save_object,
    load_harmbench,
    load_completed_keys,
    load_batch_keys,
)

# Suppress torch dynamo issues
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# BEAST sampling utilities
# ============================================================================

@torch.no_grad()
def sample_top_p(probs, p, return_tokens=0):
    """Top-p sampling with numerical stability."""
    probs = probs.float()
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    probs_sum = probs_sort.sum(dim=-1, keepdim=True)
    probs_sum = torch.where(probs_sum == 0, torch.ones_like(probs_sum), probs_sum)
    probs_sort = probs_sort / probs_sum
    probs_sort = torch.nan_to_num(probs_sort, nan=0.0, posinf=0.0, neginf=0.0)
    probs_sort = torch.clamp(probs_sort, min=0.0)

    row_sums = probs_sort.sum(dim=-1, keepdim=True)
    zero_rows = (row_sums == 0).squeeze(-1)
    if zero_rows.any():
        probs_sort[zero_rows, 0] = 1.0

    next_token = torch.multinomial(probs_sort, num_samples=max(1, return_tokens))
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# ============================================================================
# BEAST AutoRegressor
# ============================================================================

class BeastAttacker:
    """
    BEAST beam-search attacker that works with any model (base HF or SAE-wrapped).
    All operations are @torch.no_grad since BEAST doesn't use gradients.
    """

    def __init__(self, model, tokenizer, device, max_bs=8, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_bs = max_bs
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _to_device(self, tokens):
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens.to(self.device)

    @torch.no_grad()
    def _forward(self, input_ids, **kwargs):
        """Forward pass through the model (base or SAE-wrapped)."""
        return self.model(input_ids=input_ids, **kwargs)

    @torch.no_grad()
    def _generate(self, input_ids, **kwargs):
        """Generate from the model (base or SAE-wrapped)."""
        if hasattr(self.model, "generate"):
            return self.model.generate(input_ids=input_ids, **kwargs)
        else:
            return self.model.generate(input_ids, **kwargs)

    @torch.no_grad()
    def generate_n_tokens_batch(self, prompt_tokens, max_gen_len,
                                temperature=1.0, top_p=1.0, top_k=None):
        """Generate n tokens for a batch, returning logits and full sequences."""
        assert max(len(i) for i in prompt_tokens) == min(len(i) for i in prompt_tokens)
        prompt_tokens = self._to_device(prompt_tokens)

        if max_gen_len == 0:
            return None, prompt_tokens

        prompt_len = prompt_tokens.shape[1]

        # Build generation config
        gen_kwargs = dict(
            max_length=max_gen_len + prompt_len,
            min_length=max_gen_len + prompt_len,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        out = self._generate(prompt_tokens, **gen_kwargs)

        tokens = out.sequences
        logits = torch.stack(out.scores)
        logits = torch.permute(logits, (1, 0, 2))

        return logits, tokens

    @torch.no_grad()
    def perplexity(self, x1, x2):
        """Compute perplexity of x2 given prefix x1."""
        x2 = self._to_device(x2)
        output = self._forward(x2, use_cache=False)
        softmax = torch.nn.Softmax(dim=-1)
        logs = None

        start = len(x1[0]) if isinstance(x1, list) else x1.shape[1]

        for curr_pos in range(start, x2.shape[1]):
            log = -torch.log(
                softmax(output.logits)[
                    torch.arange(len(output.logits), device=self.device),
                    curr_pos - 1,
                    x2[:, curr_pos],
                ]
            )
            logs = log if logs is None else (logs + log)

        denom = x2.shape[1] - start
        if denom <= 0:
            return torch.ones(x2.shape[0], device=self.device)
        return torch.exp(logs / denom).float()

    @torch.no_grad()
    def attack_objective_targeted(self, tokens, target: str):
        """Targeted attack: minimize perplexity of target continuation."""
        tokens = self._to_device(tokens)
        target_ids = self.tokenizer.encode(target, return_tensors="pt",
                                           add_special_tokens=False).to(self.device)
        tokens_ext = torch.cat([tokens, target_ids.expand(len(tokens), -1)], dim=1)

        if tokens.shape == tokens_ext.shape:
            bos = self.tokenizer.encode(self.tokenizer.bos_token, return_tensors="pt",
                                        add_special_tokens=False).to(self.device)
            bos = bos.expand(len(tokens_ext), -1)
            tokens_ext = torch.cat([bos, tokens_ext], dim=1)
            scores = -self.perplexity(tokens_ext[:, :1], tokens_ext).cpu().numpy()
        else:
            scores = -self.perplexity(tokens, tokens_ext).cpu().numpy()
        return scores

    @torch.no_grad()
    def attack_objective_untargeted(self, toks, look_ahead_length):
        """Untargeted attack: maximize perplexity of generated continuation."""
        return self.perplexity(toks[:, :-look_ahead_length], toks).cpu().numpy()

    @torch.no_grad()
    def run_attack(self, prompt: str, target: str = None,
                   k1=15, k2=15, lookahead_length=10, new_gen_length=10,
                   n_trials=1, temperature=1.0, top_p=1.0, top_k=None,
                   verbose=1):
        """
        Run BEAST beam-search attack on a single prompt.

        Returns: (best_suffix, best_score, best_adv_prompt)
        """
        prompt_tokens = self.tokenizer.encode(prompt.strip(), add_special_tokens=True)
        max_bs = self.max_bs

        # Initial generation (1 token)
        logits, _ = self.generate_n_tokens_batch(
            [prompt_tokens], max_gen_len=1,
            temperature=temperature, top_p=top_p, top_k=top_k)

        curr_tokens = sample_top_p(
            torch.softmax(logits[:, 0], dim=-1), top_p, return_tokens=k1)[:, :k1]
        curr_tokens = [[prompt_tokens + [tok.cpu().item()] for tok in curr_tokens[0]]]

        best_scores = [[-float("inf")] * k1]
        best_prompts = [[None] * k1]

        start_time = time.time()

        # Main BEAST loop
        for step in range(new_gen_length - 1):
            if verbose:
                print(f"  Step {step + 2}/{new_gen_length}, "
                      f"Time: {(time.time() - start_time) / 60:.2f}m")

            curr_flat = curr_tokens[0]

            # Generate k2 next tokens per candidate
            next_tokens_all = []
            for b in range(0, len(curr_flat), max_bs):
                batch = curr_flat[b:b + max_bs]
                logits, _ = self.generate_n_tokens_batch(
                    batch, max_gen_len=1,
                    temperature=temperature, top_p=top_p, top_k=top_k)
                nexts = sample_top_p(
                    torch.softmax(logits[:, 0], dim=-1), top_p, return_tokens=k2)[:, :k2]
                next_tokens_all.extend(
                    [[tok.cpu().item() for tok in row] for row in nexts])

            # Expand candidates
            expanded = []
            for i, base in enumerate(curr_flat):
                for tok in next_tokens_all[i]:
                    expanded.append(base + [tok])

            # Score candidates
            scores = np.zeros(len(expanded))
            for b in range(0, len(expanded), max_bs):
                batch = expanded[b:b + max_bs]
                for _ in range(n_trials):
                    if target is not None:
                        scores[b:b + len(batch)] += self.attack_objective_targeted(
                            batch, target)
                    else:
                        toks = self.generate_n_tokens_batch(
                            batch, lookahead_length,
                            temperature=temperature, top_p=top_p, top_k=top_k)[1]
                        scores[b:b + len(batch)] += self.attack_objective_untargeted(
                            toks, lookahead_length)
                scores[b:b + len(batch)] /= n_trials

            # Beam prune: keep top-k1
            top_idx = np.argsort(scores)[-k1:]
            curr_tokens = [[expanded[j] for j in top_idx]]
            scores_list = [scores[top_idx].tolist()]

            # Track best
            best_scores[0] += scores_list[0]
            best_prompts[0] += curr_tokens[0]
            ind = np.argsort(best_scores[0])[-k1:]
            best_scores[0] = [best_scores[0][j] for j in ind]
            best_prompts[0] = [best_prompts[0][j] for j in ind]

            if verbose:
                print(f"    Best score: {max(scores_list[0]):.4f}")

        # Extract best result
        best_idx = int(np.argmax(best_scores[0]))
        best_adv_tokens = best_prompts[0][best_idx]
        best_adv_prompt = self.tokenizer.decode(best_adv_tokens, skip_special_tokens=True)
        best_score = float(best_scores[0][best_idx])

        suffix_tokens = best_adv_tokens[len(prompt_tokens):]
        suffix = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True)

        return suffix, best_score, best_adv_prompt


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified BEAST attack.")

    # Model
    parser.add_argument("--model_config", required=True,
                        help="Key from configs/models.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", default=None)

    # Data
    parser.add_argument("--harmbench_dir", required=True)
    parser.add_argument("--batch_path", default=None)
    parser.add_argument("--batch_id", type=int, default=None)

    # Output
    parser.add_argument("--output_dir", required=True)

    # BEAST hyperparameters
    parser.add_argument("--k1", type=int, default=15)
    parser.add_argument("--k2", type=int, default=15)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--new_gen_length", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_bs", type=int, default=8)

    # Attack type
    parser.add_argument("--untargeted", action="store_true",
                        help="Run untargeted attack (default: targeted)")

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

    # Determine device
    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device

    # ── Load data ──
    base_dict, optimizer_targets = load_harmbench(args.harmbench_dir)

    if args.batch_path:
        batch_keys = load_batch_keys(args.batch_path)
    else:
        batch_keys = set(base_dict.keys())

    # Resumption
    os.makedirs(args.output_dir, exist_ok=True)
    completed_keys = load_completed_keys(args.output_dir)
    remaining = [k for k in base_dict if k in batch_keys and k not in completed_keys]

    print(f"Batch keys: {len(batch_keys)}  |  Completed: {len(completed_keys)}  "
          f"|  Remaining: {len(remaining)}")

    # ── Build attacker ──
    attacker = BeastAttacker(
        model=model, tokenizer=tokenizer, device=device,
        max_bs=args.max_bs, seed=args.seed)

    # ── Run ──
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    batch_tag = f"_B{args.batch_id}" if args.batch_id is not None else ""
    attack_type = "untargeted" if args.untargeted else "targeted"
    out_file = os.path.join(
        args.output_dir,
        f"beast_{attack_type}_{args.model_config}_{timestamp}{batch_tag}.pkl"
    )

    beast_results = {}

    for key in tqdm(remaining, desc=f"BEAST {args.model_config}"):
        prompt = base_dict[key]["Behavior"]
        target = optimizer_targets.get(key, "Sure, here is")

        try:
            suffix, score, adv_prompt = attacker.run_attack(
                prompt=prompt,
                target=target if not args.untargeted else None,
                k1=args.k1, k2=args.k2,
                lookahead_length=args.lookahead,
                new_gen_length=args.new_gen_length,
                n_trials=args.n_trials,
                temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                verbose=1,
            )

            beast_results[key] = {
                "result": {
                    "best_string": suffix,
                    "best_score": score,
                    "best_adversarial_prompt": adv_prompt,
                    "original_prompt": prompt,
                    "target": target,
                    "attack_type": attack_type,
                },
                "model_config": args.model_config,
                "model_name": model_cfg["model_name"],
                "sae_type": model_cfg.get("sae_type", "none"),
                "k1": args.k1, "k2": args.k2,
                "lookahead_length": args.lookahead,
                "new_gen_length": args.new_gen_length,
                "n_trials": args.n_trials,
                "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k,
                "seed": args.seed,
            }

            print(f"  {key}: score={score:.4f}  suffix={suffix[:50]}")

            # Checkpoint
            save_object(beast_results, out_file)

        except Exception as e:
            import traceback
            print(f"Error on {key}: {e}")
            traceback.print_exc()
            beast_results[key] = {"error": repr(e), "prompt": prompt}
            save_object(beast_results, out_file)

    print(f"\nDone. {len(beast_results)} results saved to {out_file}")


if __name__ == "__main__":
    main()
