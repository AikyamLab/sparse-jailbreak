import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Standard library imports
import json
import os
import pickle
from datetime import datetime
from pathlib import Path  # Using Path instead of pathlib
from typing import Any, Callable, Dict, List, Optional

# Third-party imports
import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Project-specific imports
import goodfire
import nanogcg
import nnsight
from nanogcg import GCGConfig

# Authentication
login('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')




class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)

    return sae




class SaeInterventionModel(torch.nn.Module):
    """
    HF model wrapper that applies SAE reconstruction at a specified layer
    during *all* forward passes (including nanogcg's inputs_embeds passes).
    """

    def __init__(
        self,
        model_name: str,
        sae: torch.nn.Module,
        sae_layer: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
        detach_reconstruction: bool = False,   # False = grads flow through SAE
        record_stats: bool = True,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.sae = sae
        self.sae_layer = sae_layer
        self.detach_reconstruction = detach_reconstruction
        self.record_stats = record_stats

        # nanogcg expects these
        self.dtype = self.model.dtype
        self.device = self.model.device

        # runtime verification
        self.sae_hook_calls = 0
        self.last_recon_l2 = None

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _resolve_sae_module(self):
        module = self.model
        for name in self.sae_layer.split("."):
            module = getattr(module, name)
        return module

    def _make_hook(self):
        def hook_fn(module, inputs, output):
            # HF blocks typically output tuple(hidden_states, ...) or just tensor
            acts = output[0] if isinstance(output, tuple) else output

            recon, feats = self.sae(acts)  # your SAE.forward returns (decode(f), f)

            if self.detach_reconstruction:
                recon = recon.detach()

            if self.record_stats:
                self.sae_hook_calls += 1
                # L2 recon error (scalar)
                self.last_recon_l2 = torch.norm(acts - recon).detach().float().item()

            if isinstance(output, tuple):
                return (recon,) + output[1:]
            return recon

        return hook_fn

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        # Apply SAE for all forwards that run through this wrapper (including GCG)
        target_module = self._resolve_sae_module()
        handle = target_module.register_forward_hook(self._make_hook())
        try:
            return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        finally:
            handle.remove()

    # keep __call__ default from nn.Module (don’t override)
    # generate wrapper (optional) — generation will also go through forward hooks
    def generate(self, *args, **kwargs):
        # Note: HF generate internally calls model.forward repeatedly, so SAE applies.
        return self.model.generate(*args, **kwargs)

    def test_sae_reconstruction(self, prompt="Hello, how are you?"):
        self.sae_hook_calls = 0
        self.last_recon_l2 = None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.forward(**inputs)

        ok = (self.sae_hook_calls > 0) and (self.last_recon_l2 is not None) and (self.last_recon_l2 > 0)
        print(f"SAE hook calls: {self.sae_hook_calls}")
        print(f"Last recon L2:  {self.last_recon_l2}")
        print(f"Verified active: {ok}")
        return ok




# Save an object
def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

# Load an object
def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def read_jsonl_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Read {len(data)} items from {filename}")
    return data
