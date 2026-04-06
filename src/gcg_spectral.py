# gcg_spectral.py
"""
GCG with Spectral Monitoring - Subclass of nanogcg.GCG that monitors
gradient spectral properties during optimization.
"""

import copy
import gc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
from transformers import set_seed

import nanogcg
from nanogcg import GCGConfig
from nanogcg.gcg import (
    GCG,
    GCGResult,
    AttackBuffer,
    sample_ids_from_grad,
    filter_ids,
    find_executable_batch_size,
    logger,
)


@dataclass
class SpectralMonitorConfig:
    """Configuration for spectral monitoring during GCG optimization."""
    
    # How often to compute SVD metrics (1 = every step)
    compute_every_n_steps: int = 1
    
    # Whether to store gradient tensors
    store_gradients: bool = False
    
    # If storing gradients, how often (1 = every step)
    store_gradients_every_n_steps: int = 10
    
    # Store projected gradients [suffix_len, d_embed] vs full [suffix_len, vocab_size]
    store_projected: bool = True
    
    # Additional metrics to compute
    compute_gradient_norms: bool = True
    compute_cosine_similarity: bool = True  # Between successive gradients
    compute_top_k_tokens: bool = True  # Track which tokens have highest gradient
    top_k: int = 10


@dataclass 
class SpectralMetrics:
    """Spectral metrics computed at a single optimization step."""
    step: int
    loss: float
    max_singular_value: float
    effective_rank: float
    spectral_gap: float
    spectral_ratio: float
    condition_number: float
    gradient_norm: float
    num_singular_values: int
    top_5_singular_values: List[float]
    variance_explained_top1: float
    variance_explained_top5: float
    timestamp: str
    
    # Optional metrics
    cosine_sim_with_previous: Optional[float] = None
    top_k_tokens: Optional[List[List[int]]] = None  # Per position
    gradient_sparsity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SpectralHistory:
    """Container for all spectral monitoring results."""
    config: SpectralMonitorConfig
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    gradients: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summary statistics (computed at end)
    summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "metrics": self.metrics,
            "gradients": [
                {"step": g["step"], "shape": list(g["gradient"].shape)}
                for g in self.gradients
            ],  # Don't serialize full tensors to dict
            "summary": self.summary,
        }
    
    def get_gradient_tensors(self) -> List[Dict[str, Any]]:
        """Return gradients with actual tensors (for separate storage)."""
        return self.gradients


class GCGWithSpectralMonitoring(GCG):
    """
    GCG optimizer with spectral monitoring of gradients.
    
    Subclasses nanogcg.GCG to intercept gradients and compute SVD-based
    metrics at each optimization step.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        gcg_config: GCGConfig,
        spectral_config: Optional[SpectralMonitorConfig] = None,
    ):
        super().__init__(model, tokenizer, gcg_config)
        
        self.spectral_config = spectral_config or SpectralMonitorConfig()
        self.spectral_history = SpectralHistory(config=self.spectral_config)
        
        # Cache for cosine similarity computation
        self._previous_gradient: Optional[Tensor] = None
        
        # Current step counter (set during run)
        self._current_step: int = 0
        self._current_loss: float = 0.0
        
    def _compute_effective_rank(self, singular_values: Tensor) -> float:
        """
        Compute effective rank: r_eff = (Σσᵢ)² / Σσᵢ²
        """
        s = singular_values[singular_values > 1e-8]
        if len(s) == 0:
            return 0.0
        sum_s = s.sum()
        sum_s_sq = (s ** 2).sum()
        if sum_s_sq == 0:
            return 0.0
        return ((sum_s ** 2) / sum_s_sq).item()
    
    def _compute_spectral_metrics(
        self,
        grad: Tensor,
        step: int,
        loss: float,
    ) -> SpectralMetrics:
        """
        Compute spectral metrics from gradient tensor.
        
        Args:
            grad: Gradient tensor [suffix_len, vocab_size] or [suffix_len, d_embed]
            step: Current optimization step
            loss: Current loss value
        """
        # Project to embedding space if needed
        if grad.shape[1] == self.embedding_layer.num_embeddings:
            # Full gradient [suffix_len, vocab_size] -> project to [suffix_len, d_embed]
            grad_for_svd = grad @ self.embedding_layer.weight  # [suffix_len, d_embed]
        else:
            grad_for_svd = grad
        
        # Compute SVD
        try:
            U, s, Vt = torch.linalg.svd(grad_for_svd.float().cpu(), full_matrices=False)
        except RuntimeError as e:
            logger.warning(f"SVD failed at step {step}: {e}")
            return None
        
        # Filter near-zero singular values
        s_filtered = s[s > 1e-8]
        
        if len(s_filtered) < 2:
            logger.warning(f"Insufficient singular values at step {step}")
            return None
        
        # Core metrics
        sigma_1 = s_filtered[0].item()
        sigma_2 = s_filtered[1].item()
        sigma_r = s_filtered[-1].item()
        
        sum_s = s_filtered.sum().item()
        sum_s_sq = (s_filtered ** 2).sum().item()
        
        effective_rank = (sum_s ** 2 / sum_s_sq) if sum_s_sq > 0 else 0.0
        spectral_ratio = sigma_2 / sigma_1 if sigma_1 > 0 else 0.0
        spectral_gap = sigma_1 / sigma_2 if sigma_2 > 0 else float("inf")
        condition_number = sigma_1 / sigma_r if sigma_r > 0 else float("inf")
        
        variance_explained_top1 = (sigma_1 ** 2) / sum_s_sq if sum_s_sq > 0 else 0.0
        variance_explained_top5 = (s_filtered[:5] ** 2).sum().item() / sum_s_sq if sum_s_sq > 0 else 0.0
        
        # Gradient norm
        gradient_norm = grad_for_svd.norm().item() if self.spectral_config.compute_gradient_norms else 0.0
        
        # Cosine similarity with previous gradient
        cosine_sim = None
        if self.spectral_config.compute_cosine_similarity and self._previous_gradient is not None:
            try:
                flat_current = grad_for_svd.flatten()
                flat_previous = self._previous_gradient.flatten()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    flat_current.unsqueeze(0),
                    flat_previous.unsqueeze(0)
                ).item()
            except Exception:
                cosine_sim = None
        
        # Update previous gradient cache
        if self.spectral_config.compute_cosine_similarity:
            self._previous_gradient = grad_for_svd.detach().clone()
        
        # Top-k tokens per position
        top_k_tokens = None
        if self.spectral_config.compute_top_k_tokens:
            # Use original full gradient for this
            if grad.shape[1] == self.embedding_layer.num_embeddings:
                top_k_tokens = (-grad).topk(self.spectral_config.top_k, dim=1).indices.tolist()
        
        # Gradient sparsity (fraction of near-zero elements)
        gradient_sparsity = (grad_for_svd.abs() < 1e-6).float().mean().item()
        
        return SpectralMetrics(
            step=step,
            loss=loss,
            max_singular_value=sigma_1,
            effective_rank=effective_rank,
            spectral_gap=spectral_gap,
            spectral_ratio=spectral_ratio,
            condition_number=condition_number,
            gradient_norm=gradient_norm,
            num_singular_values=len(s_filtered),
            top_5_singular_values=s_filtered[:5].tolist(),
            variance_explained_top1=variance_explained_top1,
            variance_explained_top5=variance_explained_top5,
            timestamp=datetime.now().isoformat(),
            cosine_sim_with_previous=cosine_sim,
            top_k_tokens=top_k_tokens,
            gradient_sparsity=gradient_sparsity,
        )
    
    def _should_compute_metrics(self, step: int) -> bool:
        """Check if we should compute metrics at this step."""
        return step % self.spectral_config.compute_every_n_steps == 0
    
    def _should_store_gradient(self, step: int) -> bool:
        """Check if we should store gradient at this step."""
        return (
            self.spectral_config.store_gradients and 
            step % self.spectral_config.store_gradients_every_n_steps == 0
        )
    
    def _store_gradient(self, grad: Tensor, step: int):
        """Store gradient tensor."""
        if self.spectral_config.store_projected:
            # Project to embedding space
            if grad.shape[1] == self.embedding_layer.num_embeddings:
                grad_to_store = (grad @ self.embedding_layer.weight).detach().cpu()
            else:
                grad_to_store = grad.detach().cpu()
        else:
            # Store full gradient
            grad_to_store = grad.detach().cpu()
        
        self.spectral_history.gradients.append({
            "step": step,
            "gradient": grad_to_store,
        })
    
    def compute_token_gradient(self, optim_ids: Tensor) -> Tensor:
        """
        Override parent method to intercept gradients for spectral analysis.
        """
        # Call parent implementation
        optim_ids_onehot_grad = super().compute_token_gradient(optim_ids)
        
        # Extract gradient for analysis
        grad = optim_ids_onehot_grad.squeeze(0)  # [suffix_len, vocab_size]
        
        # Compute spectral metrics if needed
        if self._should_compute_metrics(self._current_step):
            metrics = self._compute_spectral_metrics(
                grad, 
                self._current_step, 
                self._current_loss
            )
            if metrics is not None:
                self.spectral_history.metrics.append(metrics.to_dict())
        
        # Store gradient if needed
        if self._should_store_gradient(self._current_step):
            self._store_gradient(grad, self._current_step)
        
        return optim_ids_onehot_grad
    
    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        """
        Run GCG optimization with spectral monitoring.
        
        This overrides the parent run() to track step count and loss.
        """
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        # Reset spectral monitoring state
        self._previous_gradient = None
        self._current_step = 0
        self._current_loss = buffer.get_lowest_loss()

        for step in tqdm(range(config.num_steps), desc="GCG + Spectral"):
            self._current_step = step
            
            # Compute the token gradient (spectral analysis happens here)
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds)
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update current loss for next iteration's spectral analysis
                self._current_loss = current_loss

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        # Compute summary statistics
        self._compute_summary()

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result
    
    def _compute_summary(self):
        """Compute summary statistics over the full optimization trajectory."""
        if not self.spectral_history.metrics:
            return
        
        metrics = self.spectral_history.metrics
        
        # Extract trajectories
        steps = [m["step"] for m in metrics]
        sigma_vals = [m["max_singular_value"] for m in metrics]
        rank_vals = [m["effective_rank"] for m in metrics]
        gap_vals = [m["spectral_gap"] for m in metrics]
        loss_vals = [m["loss"] for m in metrics]
        
        # Compute correlations
        try:
            loss_sigma_corr = np.corrcoef(loss_vals, sigma_vals)[0, 1]
            loss_rank_corr = np.corrcoef(loss_vals, rank_vals)[0, 1]
        except Exception:
            loss_sigma_corr = None
            loss_rank_corr = None
        
        self.spectral_history.summary = {
            "num_steps_monitored": len(metrics),
            "max_singular_value": {
                "initial": sigma_vals[0],
                "final": sigma_vals[-1],
                "min": min(sigma_vals),
                "max": max(sigma_vals),
                "mean": np.mean(sigma_vals),
                "std": np.std(sigma_vals),
            },
            "effective_rank": {
                "initial": rank_vals[0],
                "final": rank_vals[-1],
                "min": min(rank_vals),
                "max": max(rank_vals),
                "mean": np.mean(rank_vals),
                "std": np.std(rank_vals),
            },
            "spectral_gap": {
                "initial": gap_vals[0],
                "final": gap_vals[-1],
                "min": min(gap_vals),
                "max": max(gap_vals),
                "mean": np.mean(gap_vals),
                "std": np.std(gap_vals),
            },
            "loss": {
                "initial": loss_vals[0],
                "final": loss_vals[-1],
                "min": min(loss_vals),
            },
            "correlations": {
                "loss_vs_max_singular_value": loss_sigma_corr,
                "loss_vs_effective_rank": loss_rank_corr,
            },
        }
    
    def get_spectral_history(self) -> SpectralHistory:
        """Return the spectral history object."""
        return self.spectral_history
    
    def save_spectral_history(self, filepath: str):
        """Save spectral history to file."""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                "config": self.spectral_config.__dict__,
                "metrics": self.spectral_history.metrics,
                "gradients": self.spectral_history.gradients,
                "summary": self.spectral_history.summary,
            }, f)
        
        logger.info(f"Saved spectral history to {filepath}")


# Convenience function matching nanogcg.run() API
def run_with_spectral_monitoring(
    model,
    tokenizer,
    messages: Union[str, List[dict]],
    target: str,
    gcg_config: Optional[GCGConfig] = None,
    spectral_config: Optional[SpectralMonitorConfig] = None,
) -> tuple[GCGResult, SpectralHistory]:
    """
    Run GCG with spectral monitoring.
    
    Args:
        model: The model to optimize against
        tokenizer: The model's tokenizer
        messages: The conversation/prompt
        target: The target string to optimize for
        gcg_config: GCG configuration
        spectral_config: Spectral monitoring configuration
        
    Returns:
        Tuple of (GCGResult, SpectralHistory)
    """
    if gcg_config is None:
        gcg_config = GCGConfig()
    
    if spectral_config is None:
        spectral_config = SpectralMonitorConfig()
    
    logger.setLevel(getattr(__import__('logging'), gcg_config.verbosity))
    
    gcg = GCGWithSpectralMonitoring(model, tokenizer, gcg_config, spectral_config)
    result = gcg.run(messages, target)
    
    return result, gcg.get_spectral_history()