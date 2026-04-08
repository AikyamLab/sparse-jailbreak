#!/usr/bin/env python3
"""
run_feature_extraction.py — Unified SAE feature extraction for all model families.

Extracts SAE encoder features from adversarial/random suffixes, grouped by source.
Outputs per-source .pkl files containing feature vectors for downstream analysis
(Jaccard similarity, feature overlap, etc.).

Supports all three SAE formats via YAML config:
  - Gemma Scope (sae_lens): sae.type = "gemma_scope"
  - Goodfire (LLaMA): sae.type = "goodfire"
  - JoshEngels (Mistral): sae.type = "mistral_res"

Usage:
  python scripts/run_feature_extraction.py --config configs/feature_extraction/gemma_9b.yml
  python scripts/run_feature_extraction.py --config configs/feature_extraction/llama_8b.yml
  python scripts/run_feature_extraction.py --config configs/feature_extraction/mistral_7b.yml
  python scripts/run_feature_extraction.py --config configs/feature_extraction/gemma_9b.yml --device cuda:1
"""

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Config
# ============================================================================

@dataclass
class Config:
    """Configuration for SAE feature extraction."""

    # Input
    pickle_file: str

    # Model
    model_name: str
    model_cache_dir: Optional[str] = None

    # SAE — fields vary by type
    sae_type: str = "gemma_scope"  # "gemma_scope" | "goodfire" | "mistral_res"

    # gemma_scope fields
    sae_release: Optional[str] = None
    sae_id: Optional[str] = None

    # goodfire fields
    sae_name: Optional[str] = None
    expansion_factor: Optional[int] = None

    # mistral_res fields
    sae_repo: Optional[str] = None
    sae_size: Optional[int] = None

    # Common
    sae_layer: int = 19

    # Output
    output_dir: str = "sae_features"

    # Processing
    batch_size: int = 4
    max_length: int = 512
    device: str = "auto"

    # Auth
    hf_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        sae_data = data.get("sae", {})
        sae_type = sae_data.get("type", "gemma_scope")

        return cls(
            pickle_file=data["input"]["pickle_file"],
            model_name=data["model"]["name"],
            model_cache_dir=data["model"].get("cache_dir"),
            sae_type=sae_type,
            sae_release=sae_data.get("release"),
            sae_id=sae_data.get("id"),
            sae_name=sae_data.get("name"),
            expansion_factor=sae_data.get("expansion_factor"),
            sae_repo=sae_data.get("repo"),
            sae_size=sae_data.get("size"),
            sae_layer=sae_data.get("layer", 19),
            output_dir=data["output"]["dir"],
            batch_size=data["processing"]["batch_size"],
            max_length=data["processing"]["max_length"],
            device=data["processing"]["device"],
            hf_token=data.get("huggingface", {}).get("token"),
        )

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================================
# SAE loaders (encode-only, no gradient flow needed)
# ============================================================================

def load_sae_encoder(config: Config, device: str, dtype: torch.dtype):
    """
    Load the SAE encoder for feature extraction.

    Returns an object with:
      - .encode(x) -> features tensor
      - .d_sae -> int (feature dimension)
    """
    if config.sae_type == "gemma_scope":
        from sae_lens import SAE
        sae_result = SAE.from_pretrained(
            release=config.sae_release,
            sae_id=config.sae_id,
            device=device,
        )
        sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
        sae.eval()
        print(f"  Gemma Scope SAE loaded (d_sae={sae.cfg.d_sae})")
        return sae

    elif config.sae_type == "goodfire":
        from huggingface_hub import hf_hub_download
        from src.models import GoodfireSAE

        file_path = hf_hub_download(
            repo_id=f"Goodfire/{config.sae_name}",
            filename=f"{config.sae_name}.pth",
            repo_type="model",
        )

        # Need d_model from the model config
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(
            config.model_name, cache_dir=config.model_cache_dir)
        d_model = model_config.hidden_size

        sae = GoodfireSAE(
            d_model, d_model * config.expansion_factor,
            torch.device(device), dtype=dtype)
        sae.load_state_dict(
            torch.load(file_path, weights_only=True, map_location=device))
        sae.eval()
        # Add d_sae attribute for compatibility
        sae.d_sae = d_model * config.expansion_factor
        print(f"  Goodfire SAE loaded (d_sae={sae.d_sae})")
        return sae

    elif config.sae_type == "mistral_res":
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        sae_filename = f"mistral_7b_layer_{config.sae_layer}/sae_weights.safetensors"
        file_path = hf_hub_download(
            repo_id=config.sae_repo,
            filename=sae_filename,
            repo_type="model",
        )
        state = safetensors.torch.load_file(file_path)

        if "W_enc" in state:
            W_enc = state["W_enc"].to(dtype).to(device)
            b_enc = state["b_enc"].to(dtype).to(device)
        else:
            W_enc = state["encoder.weight"].to(dtype).to(device)
            b_enc = state["encoder.bias"].to(dtype).to(device)

        class MistralEncoder:
            """Lightweight encoder-only wrapper for Mistral SAE."""
            def __init__(self, W_enc, b_enc):
                self.W_enc = W_enc
                self.b_enc = b_enc
                self.d_sae = W_enc.shape[-1] if len(W_enc.shape) > 1 else b_enc.shape[0]

            def encode(self, x):
                norm_constant = 64.0
                original_norm = torch.norm(x, dim=-1, keepdim=True)
                x_normalized = x * (norm_constant / (original_norm + 1e-8))
                return torch.nn.functional.relu(
                    torch.matmul(x_normalized, self.W_enc) + self.b_enc)

        encoder = MistralEncoder(W_enc, b_enc)
        print(f"  Mistral SAE loaded (d_sae={encoder.d_sae})")
        return encoder

    else:
        raise ValueError(f"Unknown sae_type: {config.sae_type}")


# ============================================================================
# Data loading
# ============================================================================

def load_object(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def extract_prompts_from_pickle(data: Dict) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract prompts grouped by source from pickle data.

    Handles:
      - data[source][prompt_id]['result']['best_string'] (dict)
      - data[source][prompt_id]['result'].best_string (GCGResult object)
      - data[source][prompt_id] = plain string
    """
    grouped = {}

    for source_name, prompts in data.items():
        if not isinstance(prompts, dict):
            continue

        grouped[source_name] = []

        for prompt_id, prompt_data in prompts.items():
            best_string = None
            try:
                if isinstance(prompt_data, dict) and "result" in prompt_data:
                    result = prompt_data["result"]
                    if isinstance(result, dict):
                        best_string = result.get("best_string", result.get("suffix", ""))
                    elif hasattr(result, "best_string"):
                        best_string = result.best_string
                elif hasattr(prompt_data, "best_string"):
                    best_string = prompt_data.best_string
                elif isinstance(prompt_data, str):
                    best_string = prompt_data
            except (TypeError, KeyError, AttributeError):
                pass

            if best_string:
                grouped[source_name].append({
                    "prompt_id": prompt_id,
                    "best_string": best_string,
                })

        print(f"  {source_name}: {len(grouped[source_name])} prompts")

    return grouped


# ============================================================================
# Feature extraction
# ============================================================================

def extract_sae_features_batch(
    prompts: List[str],
    model,
    tokenizer,
    sae,
    layer: int,
    device: str,
    batch_size: int = 4,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """
    Extract SAE features for a list of prompts.
    Returns features from the last token position for each prompt.
    """
    all_features = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch = prompts[i: i + batch_size]

        try:
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            activations = {}

            def hook_fn(module, inp, output):
                activations["residual"] = (
                    output[0] if isinstance(output, tuple) else output
                )

            handle = model.model.layers[layer].register_forward_hook(hook_fn)

            with torch.no_grad():
                model(**inputs)

            handle.remove()

            residual = activations["residual"][:, -1, :].to(device)

            with torch.no_grad():
                batch_features = sae.encode(residual)

            all_features.extend([f.cpu() for f in batch_features])

            if (i // batch_size) % 10 == 0 and device.startswith("cuda"):
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError on batch {i // batch_size}: {e}")
            d_sae = sae.d_sae if hasattr(sae, "d_sae") else sae.cfg.d_sae
            for _ in batch:
                all_features.append(torch.zeros(d_sae))

    return all_features


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified SAE feature extraction for all model families.")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to YAML config file")
    parser.add_argument("--device", default=None,
                        help="Override device (e.g., cuda:0)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size

    device = config.resolve_device()

    print("=" * 60)
    print("SAE FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Input:    {config.pickle_file}")
    print(f"Model:    {config.model_name}")
    print(f"SAE type: {config.sae_type}")
    print(f"Layer:    {config.sae_layer}")
    print(f"Device:   {device}")
    print(f"Output:   {config.output_dir}")

    # Auth
    hf_token = config.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load data
    print("\nLoading data...")
    raw_data = load_object(config.pickle_file)
    grouped = extract_prompts_from_pickle(raw_data)
    total = sum(len(v) for v in grouped.values())
    print(f"Found {len(grouped)} sources, {total} total prompts")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, cache_dir=config.model_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    # Mistral uses float16
    if config.sae_type == "mistral_res":
        dtype = torch.float16
    # Gemma uses float16
    if config.sae_type == "gemma_scope" and "gemma" in config.model_name.lower():
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, cache_dir=config.model_cache_dir,
        torch_dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded to {device}")

    # Load SAE
    print("\nLoading SAE...")
    sae = load_sae_encoder(config, device, dtype)

    # Extract features
    print("\n" + "=" * 60)
    print("EXTRACTING FEATURES")
    print("=" * 60)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name, source_data in sorted(grouped.items()):
        if not source_data:
            continue

        print(f"\n--- {source_name} ({len(source_data)} samples) ---")

        prompts = [item["best_string"] for item in source_data]

        features = extract_sae_features_batch(
            prompts=prompts, model=model, tokenizer=tokenizer,
            sae=sae, layer=config.sae_layer, device=device,
            batch_size=config.batch_size, max_length=config.max_length,
        )

        for i, item in enumerate(source_data):
            item["sae_features"] = features[i]
            item["feature_shape"] = tuple(features[i].shape)

        stacked = torch.stack(features)

        # Stats
        active = (stacked > 0).sum(dim=1).float()
        print(f"  Sparsity: {(stacked == 0).float().mean().item():.2%}")
        print(f"  Avg active features: {active.mean().item():.1f}")

        # Save
        d_sae = sae.d_sae if hasattr(sae, "d_sae") else sae.cfg.d_sae
        out_file = output_dir / f"{source_name}_sae_features.pkl"
        save_data = {
            "source": source_name,
            "data": source_data,
            "features": stacked,
            "metadata": {
                "model": config.model_name,
                "sae_type": config.sae_type,
                "sae_layer": config.sae_layer,
                "num_samples": len(source_data),
                "feature_dim": d_sae,
            },
        }
        with open(out_file, "wb") as f:
            pickle.dump(save_data, f)
        print(f"  Saved: {out_file}")

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"DONE — features saved to {config.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
