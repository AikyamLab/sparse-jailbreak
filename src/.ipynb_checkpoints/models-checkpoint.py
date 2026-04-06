"""
Unified SAE intervention models for all model families.

All wrappers inherit from torch.nn.Module and hook the SAE into forward(),
ensuring gradients flow through the SAE during GCG optimization.

Supported SAE types:
  - "goodfire"     → Goodfire SAEs (LLaMA-3.1-8B, LLaMA-3.3-70B)
  - "mistral_res"  → JoshEngels Residual-Stream SAEs (Mistral-7B) with norm scaling
  - "gemma_scope"  → Gemma Scope via sae_lens (Gemma-2-2B, 9B, 27B)
  - "none"         → Bare HF model (baseline, no SAE)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download


# ============================================================================
# Goodfire SAE (LLaMA)
# ============================================================================

class GoodfireSAE(nn.Module):
    """Goodfire-format sparse autoencoder: linear encoder + ReLU, linear decoder."""

    def __init__(self, d_in: int, d_hidden: int, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.encoder_linear = nn.Linear(d_in, d_hidden)
        self.decoder_linear = nn.Linear(d_hidden, d_in)
        self.to(device, dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor):
        """Returns (reconstruction, features)."""
        f = self.encode(x)
        return self.decode(f), f


def load_goodfire_sae(repo_id: str, filename: str, d_model: int,
                      expansion_factor: int, device: str = "cuda",
                      cache_dir: str = None) -> GoodfireSAE:
    """Download and load a Goodfire SAE checkpoint."""
    path = hf_hub_download(repo_id=repo_id, filename=filename,
                           repo_type="model", cache_dir=cache_dir)
    sae = GoodfireSAE(d_model, d_model * expansion_factor, torch.device(device))
    sae.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return sae


# ============================================================================
# JoshEngels Residual-Stream SAE (Mistral)
# ============================================================================

class MistralSaeInterventionModel(nn.Module):
    """
    HF model wrapper that applies a JoshEngels Mistral residual-stream SAE
    during ALL forward passes (including nanogcg's inputs_embeds calls).

    Mirrors the original MistralSAEInterventionModel exactly:
    - SAE weights stored as plain tensors (moved to device on first use)
    - Norm-scaling: activations normalized to constant=64 before encoding,
      rescaled back after decoding
    - Hook registered in __call__/forward with no torch.no_grad,
      so gradients flow through for GCG

    Used for: Mistral-7B.
    """

    def __init__(self, model_name: str, sae_repo: str, layer_idx: int,
                 sae_size: int = 65536, device: str = "cuda",
                 dtype=torch.bfloat16, cache_dir: str = None):
        super().__init__()
        import safetensors.torch

        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map={"": device} if device != "auto" else "auto",
            cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # SAE config
        self.sae_repo = sae_repo
        self.sae_layer_idx = layer_idx
        self.sae_size = sae_size
        self.sae_layer = f"model.layers.{layer_idx}"

        # Download SAE weights
        sae_filename = f"mistral_7b_layer_{layer_idx}/sae_weights.safetensors"
        file_path = hf_hub_download(repo_id=sae_repo, filename=sae_filename,
                                    repo_type="model", cache_dir=cache_dir)

        # Load SAE weights as plain tensors (matching original implementation)
        sae_state_dict = safetensors.torch.load_file(file_path)
        if "W_enc" in sae_state_dict:
            self.W_enc = sae_state_dict["W_enc"].to(dtype)
            self.W_dec = sae_state_dict["W_dec"].to(dtype)
            self.b_enc = sae_state_dict["b_enc"].to(dtype)
            self.b_dec = sae_state_dict["b_dec"].to(dtype)
        else:
            self.W_enc = sae_state_dict["encoder.weight"].to(dtype)
            self.W_dec = sae_state_dict["decoder.weight"].to(dtype)
            self.b_enc = sae_state_dict["encoder.bias"].to(dtype)
            self.b_dec = sae_state_dict["decoder.bias"].to(dtype)

        # Tracking
        self.reconstruction_verified = False
        self.reconstruction_metrics = {}

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def apply_sae(self, x):
        """Apply SAE with normalization (exact same as original MistralSAEInterventionModel)."""
        norm_constant = 64.0
        original_norm = torch.norm(x, dim=-1, keepdim=True)
        x_normalized = x * (norm_constant / (original_norm + 1e-8))
        f = torch.nn.functional.relu(
            torch.matmul(x_normalized, self.W_enc) + self.b_enc
        )
        x_reconstructed_norm = torch.matmul(f, self.W_dec) + self.b_dec
        x_reconstructed = x_reconstructed_norm * (original_norm / norm_constant)
        return x_reconstructed

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        layer_module = self.model.model.layers[self.sae_layer_idx]

        def sae_intervention_hook(module, input, output):
            if isinstance(output, tuple):
                activations = output[0]
            else:
                activations = output

            batch_size, seq_len, hidden_size = activations.shape
            acts_reshaped = activations.view(-1, hidden_size)

            # Move SAE weights to correct device if needed
            if self.W_enc.device != acts_reshaped.device:
                self.W_enc = self.W_enc.to(acts_reshaped.device)
                self.W_dec = self.W_dec.to(acts_reshaped.device)
                self.b_enc = self.b_enc.to(acts_reshaped.device)
                self.b_dec = self.b_dec.to(acts_reshaped.device)

            # Apply SAE — no torch.no_grad, gradients flow through
            reconstructed = self.apply_sae(acts_reshaped)
            reconstructed = reconstructed.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (reconstructed,) + output[1:]
            return reconstructed

        hook_handle = layer_module.register_forward_hook(sae_intervention_hook)
        try:
            return self.model(input_ids=input_ids,
                              inputs_embeds=inputs_embeds, **kwargs)
        finally:
            hook_handle.remove()

    def generate(self, input_ids=None, **kwargs):
        layer_module = self.model.model.layers[self.sae_layer_idx]

        def sae_hook(module, input, output):
            if isinstance(output, tuple):
                activations = output[0]
            else:
                activations = output

            with torch.no_grad():
                batch_size, seq_len, hidden_size = activations.shape
                acts_reshaped = activations.view(-1, hidden_size)

                if self.W_enc.device != acts_reshaped.device:
                    self.W_enc = self.W_enc.to(acts_reshaped.device)
                    self.W_dec = self.W_dec.to(acts_reshaped.device)
                    self.b_enc = self.b_enc.to(acts_reshaped.device)
                    self.b_dec = self.b_dec.to(acts_reshaped.device)

                reconstructed = self.apply_sae(acts_reshaped)
                reconstructed = reconstructed.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (reconstructed,) + output[1:]
            return reconstructed

        hook_handle = layer_module.register_forward_hook(sae_hook)
        try:
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            hook_handle.remove()

    def test_sae_reconstruction(self, prompt="Hello, how are you?"):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        with torch.no_grad():
            output_without = self.model(input_ids=input_ids)

        activation_diffs = []

        def test_hook(module, input, output):
            acts = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                B, S, H = acts.shape
                flat = acts.view(-1, H)
                if self.W_enc.device != flat.device:
                    self.W_enc = self.W_enc.to(flat.device)
                    self.W_dec = self.W_dec.to(flat.device)
                    self.b_enc = self.b_enc.to(flat.device)
                    self.b_dec = self.b_dec.to(flat.device)
                recon = self.apply_sae(flat).view(B, S, H)
                activation_diffs.append(torch.norm(acts - recon).item())
            if isinstance(output, tuple):
                return (recon,) + output[1:]
            return recon

        module = self.model.model.layers[self.sae_layer_idx]
        handle = module.register_forward_hook(test_hook)
        try:
            with torch.no_grad():
                output_with = self.model(input_ids=input_ids)
        finally:
            handle.remove()

        logits_without = output_without.logits[:, -1, :]
        logits_with = output_with.logits[:, -1, :]
        probs_without = torch.softmax(logits_without, dim=-1)[0]
        probs_with = torch.softmax(logits_with, dim=-1)[0]
        similarity = torch.nn.functional.cosine_similarity(
            probs_with.unsqueeze(0), probs_without.unsqueeze(0)
        ).item()

        self.reconstruction_verified = (
            similarity < 0.9999 and activation_diffs and sum(activation_diffs) > 0
        )

        print(f"Mistral SAE Reconstruction Test:")
        print(f"  Layer: {self.sae_layer_idx}  |  "
              f"Cosine sim: {similarity:.6f}  |  "
              f"Verified: {self.reconstruction_verified}")
        return self.reconstruction_verified


# ============================================================================
# Goodfire Intervention Model (LLaMA)
# ============================================================================

class SaeInterventionModel(nn.Module):
    """
    HF model wrapper that applies a Goodfire SAE at a named transformer layer
    during ALL forward passes (including nanogcg's inputs_embeds calls).

    The SAE's forward(x) must return (reconstruction, features).

    Used for: LLaMA-3.1-8B, LLaMA-3.3-70B.
    """

    def __init__(self, model_name: str, sae: nn.Module, sae_layer: str,
                 device: str = "cuda", dtype=torch.bfloat16,
                 detach_reconstruction: bool = False,
                 record_stats: bool = True, cache_dir: str = None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device,
            cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sae = sae
        self.sae_layer = sae_layer
        self.detach_reconstruction = detach_reconstruction
        self.record_stats = record_stats

        # nanogcg expects these attributes
        self.dtype = self.model.dtype
        self.device = self.model.device

        # Runtime stats
        self.sae_hook_calls = 0
        self.last_recon_l2 = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _resolve_sae_module(self):
        """Walk the dotted layer name to find the target module."""
        module = self.model
        for name in self.sae_layer.split("."):
            module = getattr(module, name)
        return module

    def _make_hook(self):
        def hook_fn(module, inputs, output):
            acts = output[0] if isinstance(output, tuple) else output

            # SAE forward: (reconstruction, features)
            # No torch.no_grad — gradients flow through for GCG
            recon, _feats = self.sae(acts)

            if self.detach_reconstruction:
                recon = recon.detach()

            if self.record_stats:
                self.sae_hook_calls += 1
                self.last_recon_l2 = torch.norm(acts - recon).detach().float().item()

            if isinstance(output, tuple):
                return (recon,) + output[1:]
            return recon
        return hook_fn

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        target_module = self._resolve_sae_module()
        handle = target_module.register_forward_hook(self._make_hook())
        try:
            return self.model(input_ids=input_ids,
                              inputs_embeds=inputs_embeds, **kwargs)
        finally:
            handle.remove()

    def generate(self, *args, **kwargs):
        target_module = self._resolve_sae_module()
        handle = target_module.register_forward_hook(self._make_hook())
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            handle.remove()

    def test_sae_reconstruction(self, prompt="Hello, how are you?"):
        self.sae_hook_calls = 0
        self.last_recon_l2 = None
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.forward(**inputs)
        ok = (self.sae_hook_calls > 0 and self.last_recon_l2 is not None
              and self.last_recon_l2 > 0)
        print(f"SAE hook calls: {self.sae_hook_calls}  |  "
              f"Recon L2: {self.last_recon_l2}  |  Verified: {ok}")
        return ok


# ============================================================================
# sae_lens Wrapper (adapts sae_lens SAE → (recon, feats) interface)
# ============================================================================

class SaeLensWrapper(nn.Module):
    """
    Wraps a sae_lens SAE so that forward(acts) returns (recon, feats),
    matching the interface that SaeInterventionModel._make_hook expects.
    """

    def __init__(self, sae_lens_sae):
        super().__init__()
        self.sae = sae_lens_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(x)

    def forward(self, x: torch.Tensor) -> tuple:
        feats = self.sae.encode(x)
        recon = self.sae.decode(feats)
        return recon, feats


# ============================================================================
# andyrdt SAE — Layer Ablation (LLaMA-3.1-8B)
# ============================================================================
# Uses sae_lens to load SAEs, wraps with SaeLensWrapper, feeds into
# SaeInterventionModel. Gradients flow through the SAE during GCG.

ANDYRDT_LLAMA_RELEASE = "llama-3.1-8b-instruct-andyrdt"
ANDYRDT_LLAMA_LAYERS = {
    3:  "resid_post_layer_3_trainer_1",
    7:  "resid_post_layer_7_trainer_1",
    11: "resid_post_layer_11_trainer_1",
    15: "resid_post_layer_15_trainer_1",
    19: "resid_post_layer_19_trainer_1",
    23: "resid_post_layer_23_trainer_1",
    27: "resid_post_layer_27_trainer_1",
}


def load_andyrdt_layer_model(cfg: dict, device: str, dtype: torch.dtype,
                             cache_dir: str = None):
    """
    Load an andyrdt layer-ablation SAE for LLaMA-3.1-8B via sae_lens,
    wrap it, and return a SaeInterventionModel.

    Config keys:
      model_name, sae_layer_idx, sae_release (optional), sae_id (optional)
    """
    from sae_lens import SAE

    layer_idx = cfg["sae_layer_idx"]
    sae_release = cfg.get("sae_release", ANDYRDT_LLAMA_RELEASE)
    sae_id = cfg.get("sae_id", ANDYRDT_LLAMA_LAYERS[layer_idx])
    sae_layer = f"model.layers.{layer_idx}"

    sae_result = SAE.from_pretrained(release=sae_release, sae_id=sae_id,
                                     device=device)
    raw_sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    raw_sae = raw_sae.to(dtype)
    sae = SaeLensWrapper(raw_sae)

    model = SaeInterventionModel(
        model_name=cfg["model_name"], sae=sae, sae_layer=sae_layer,
        device=device, dtype=dtype, cache_dir=cache_dir)
    return model


# ============================================================================
# andyrdt SAE — Sparsity Ablation (LLaMA-3.1-8B, Layer 19)
# ============================================================================
# Only trainer_1 is registered in sae_lens. For trainers 0, 2, 3 we load
# the trainer_1 shell and hot-swap weights + threshold from ae.pt.

ANDYRDT_HF_REPO = "andyrdt/saes-llama-3.1-8b-instruct"
ANDYRDT_D_SAE = 131072
ANDYRDT_TRAINER_TO_K = {0: 32, 1: 64, 2: 128, 3: 256}


def load_andyrdt_sparsity_model(cfg: dict, device: str, dtype: torch.dtype,
                                cache_dir: str = None):
    """
    Load an andyrdt sparsity-ablation SAE for LLaMA-3.1-8B layer 19.

    For trainer != 1, loads the trainer_1 sae_lens shell and hot-swaps
    weights and threshold from the raw ae.pt checkpoint.

    Config keys:
      model_name, trainer (int 0-3), sae_layer_idx (default 19)
    """
    from sae_lens import SAE

    trainer = cfg["trainer"]
    layer_idx = cfg.get("sae_layer_idx", 19)
    sae_layer = f"model.layers.{layer_idx}"
    sae_release = cfg.get("sae_release", ANDYRDT_LLAMA_RELEASE)
    hf_repo = cfg.get("sae_hf_repo", ANDYRDT_HF_REPO)

    # Always load the trainer_1 shell from sae_lens
    sae_result = SAE.from_pretrained(
        release=sae_release,
        sae_id=f"resid_post_layer_{layer_idx}_trainer_1",
        device=device,
    )
    raw_sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result

    # Hot-swap weights for trainers 0, 2, 3
    if trainer != 1:
        ae_path = hf_hub_download(
            hf_repo,
            f"resid_post_layer_{layer_idx}/trainer_{trainer}/ae.pt",
            cache_dir=cache_dir,
        )
        sd = torch.load(ae_path, map_location=device)

        raw_sae.W_enc.data = sd["encoder.weight"].T
        raw_sae.W_dec.data = sd["decoder.weight"].T
        raw_sae.b_enc.data = sd["encoder.bias"]
        raw_sae.b_dec.data = sd["b_dec"]
        raw_sae.threshold.data = sd["threshold"].expand(ANDYRDT_D_SAE)
        del sd

    raw_sae = raw_sae.to(dtype)
    sae = SaeLensWrapper(raw_sae)

    model = SaeInterventionModel(
        model_name=cfg["model_name"], sae=sae, sae_layer=sae_layer,
        device=device, dtype=dtype, cache_dir=cache_dir)
    return model


# ============================================================================
# Gemma Scope SAE (Gemma-2 family, Qwen via sae_lens)
# ============================================================================

class GemmaScopeInterventionModel(nn.Module):
    """
    HF model wrapper that applies a Gemma Scope (sae_lens) SAE at a specified
    layer during ALL forward passes.

    Used for: Gemma-2-2B, Gemma-2-9B, Gemma-2-27B.
    """

    def __init__(self, model_name: str, sae_release: str, sae_id: str,
                 sae_layer_idx: int = None,
                 device: str = "cuda", dtype=torch.float32,
                 detach_reconstruction: bool = False,
                 record_stats: bool = True, cache_dir: str = None):
        super().__init__()
        from sae_lens import SAE

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device,
            cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        sae_result = SAE.from_pretrained(release=sae_release,
                                         sae_id=sae_id, device=device)
        if isinstance(sae_result, tuple):
            self.sae, self.cfg_dict, self.sparsity = sae_result
        else:
            self.sae = sae_result
            self.cfg_dict = None
            self.sparsity = None

        # Layer index: explicit parameter, or parse from Gemma-style sae_id
        if sae_layer_idx is not None:
            self.sae_layer_idx = sae_layer_idx
        else:
            # Gemma format: "layer_19/width_16k/canonical"
            self.sae_layer_idx = int(sae_id.split("/")[0].split("_")[1])

        self.dtype = dtype
        self.device = torch.device(device)
        self.detach_reconstruction = detach_reconstruction
        self.record_stats = record_stats

        self.sae_hook_calls = 0
        self.last_recon_norm = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _get_sae_module(self):
        return self.model.model.layers[self.sae_layer_idx]

    def _apply_sae_reconstruction(self, activations):
        """
        Apply SAE encode→decode. Handles multiple sae_lens API versions.
        No torch.no_grad — gradients flow through for GCG.
        """
        B, S, H = activations.shape
        flat = activations.reshape(-1, H)

        if hasattr(self.sae, "encode") and hasattr(self.sae, "decode"):
            recon_flat = self.sae.decode(self.sae.encode(flat))
        else:
            out = self.sae(flat)
            if hasattr(out, "sae_out"):
                recon_flat = out.sae_out
            elif isinstance(out, torch.Tensor):
                recon_flat = out
            elif isinstance(out, tuple):
                recon_flat = out[0]
            else:
                raise TypeError(f"Unexpected SAE output: {type(out)}")

        return recon_flat.reshape(B, S, H)

    def _make_sae_hook(self):
        def hook(module, inp, output):
            acts = output[0] if isinstance(output, tuple) else output
            recon = self._apply_sae_reconstruction(acts)

            if self.detach_reconstruction:
                recon = recon.detach()

            if self.record_stats:
                self.sae_hook_calls += 1
                self.last_recon_norm = torch.norm(acts - recon).detach().item()

            if isinstance(output, tuple):
                return (recon,) + output[1:]
            return recon
        return hook

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        module = self._get_sae_module()
        handle = module.register_forward_hook(self._make_sae_hook())
        try:
            return self.model(input_ids=input_ids,
                              inputs_embeds=inputs_embeds, **kwargs)
        finally:
            handle.remove()

    def generate(self, input_ids=None, **kwargs):
        module = self._get_sae_module()
        handle = module.register_forward_hook(self._make_sae_hook())
        try:
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            handle.remove()

    def test_sae_reconstruction(self, prompt="Hello, how are you?"):
        self.sae_hook_calls = 0
        self.last_recon_norm = None
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.forward(**inputs)
        ok = (self.sae_hook_calls > 0 and self.last_recon_norm is not None
              and self.last_recon_norm > 0)
        print(f"SAE hook calls: {self.sae_hook_calls}  |  "
              f"Recon norm: {self.last_recon_norm}  |  Verified: {ok}")
        return ok


# ============================================================================
# Factory
# ============================================================================

def load_model(cfg: dict):
    """
    Build the appropriate model from a config dict.

    Returns (model, tokenizer).

    Config keys by sae_type:
      "none"             → bare HF model
      "goodfire"         → GoodfireSAE + SaeInterventionModel
      "mistral_res"      → MistralSaeInterventionModel
      "andyrdt_layer"    → andyrdt layer ablation SAE + SaeInterventionModel
      "andyrdt_sparsity" → andyrdt sparsity ablation SAE (hot-swap) + SaeInterventionModel
      "gemma_scope"      → GemmaScopeInterventionModel (Gemma, Qwen)
    """
    sae_type = cfg.get("sae_type", "none")
    device = cfg.get("device", "cuda")
    dtype_str = cfg.get("dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    cache_dir = cfg.get("cache_dir")

    # ── Baseline (no SAE) ──
    if sae_type in (None, "none"):
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=dtype, device_map=device,
            cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name"], cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    # ── Goodfire SAE (LLaMA) ──
    elif sae_type == "goodfire":
        sae = load_goodfire_sae(
            repo_id=cfg["sae_repo_id"],
            filename=cfg["sae_filename"],
            d_model=cfg["d_model"],
            expansion_factor=cfg["expansion_factor"],
            device=device, cache_dir=cache_dir)
        model = SaeInterventionModel(
            model_name=cfg["model_name"], sae=sae,
            sae_layer=cfg["sae_layer"], device=device, dtype=dtype,
            cache_dir=cache_dir)
        model.test_sae_reconstruction()
        return model, model.tokenizer

    # ── JoshEngels Residual-Stream SAE (Mistral) ──
    elif sae_type == "mistral_res":
        model = MistralSaeInterventionModel(
            model_name=cfg["model_name"],
            sae_repo=cfg["sae_repo_id"],
            layer_idx=cfg["sae_layer_idx"],
            sae_size=cfg.get("sae_size", 65536),
            device=device, dtype=dtype,
            cache_dir=cache_dir)
        model.test_sae_reconstruction()
        return model, model.tokenizer

    # ── andyrdt layer ablation SAE (LLaMA-3.1-8B) ──
    elif sae_type == "andyrdt_layer":
        model = load_andyrdt_layer_model(cfg, device=device, dtype=dtype,
                                         cache_dir=cache_dir)
        model.test_sae_reconstruction()
        return model, model.tokenizer

    # ── andyrdt sparsity ablation SAE (LLaMA-3.1-8B, layer 19) ──
    elif sae_type == "andyrdt_sparsity":
        model = load_andyrdt_sparsity_model(cfg, device=device, dtype=dtype,
                                            cache_dir=cache_dir)
        model.test_sae_reconstruction()
        return model, model.tokenizer

    # ── Gemma Scope / sae_lens SAE (Gemma-2, andyrdt LLaMA) ──
    elif sae_type == "gemma_scope":
        model = GemmaScopeInterventionModel(
            model_name=cfg["model_name"],
            sae_release=cfg["sae_release"],
            sae_id=cfg["sae_id"],
            sae_layer_idx=cfg.get("sae_layer_idx"),
            device=device, dtype=dtype, cache_dir=cache_dir)
        model.test_sae_reconstruction()
        return model, model.tokenizer

    else:
        raise ValueError(f"Unknown sae_type: {sae_type}")