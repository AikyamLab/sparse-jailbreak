# Towards Understanding the Robustness of Sparse Autoencoders

**Ahson Saiyed, Sabrina Sadiekh, Chirag Agarwal**

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><b>arXiv</b></a>
</p>

---

## Overview

Large Language Models (LLMs) remain vulnerable to optimization-based jailbreak attacks that exploit internal gradient structure. While Sparse Autoencoders (SAEs) are widely used for interpretability, their robustness implications remain underexplored. We present a study of integrating pretrained SAEs into transformer residual streams at inference time, without modifying model weights or blocking gradients.

Across four model families (**Gemma**, **LLaMA**, **Mistral**, **Qwen**) and two strong white-box attacks (**GCG**, **BEAST**) plus three black-box benchmarks, SAE-augmented models achieve **up to a 5× reduction in jailbreak success rate** relative to the undefended baseline and **reduce cross-model attack transferability**.

Parametric ablations reveal:
1. A **monotonic dose–response relationship** between L0 sparsity and attack success rate
2. A **layer-dependent defense–utility tradeoff**, where intermediate layers balance robustness and clean performance

These findings are consistent with a **representational bottleneck hypothesis**: sparse projection reshapes the optimization geometry exploited by jailbreak attacks.


---

## Dataset

All experiments use the [HarmBench](https://github.com/centerforaisafety/HarmBench) evaluation framework:

- **HarmBench prompts**: 218 harmful behavior prompts (`harmbench_behaviors_text_all.csv`)
- **Optimizer targets**: per-prompt target strings (`harmbench_targets_text.json`)
- **Black-box benchmarks**: Salad-Data, Prompt Injections Benchmark, SafeEval (1,500 jailbreak prompts)
- **Random baselines**: 5 categories of random text suffixes (alphanumeric, mixed case, lowercase, numbers, unicode)

Pre-computed adversarial suffixes and SAE features will be released on HuggingFace.

---

## Repository Structure

```
sparse-jailbreak/
├── configs/
│   ├── models.yaml                          # 41 model+SAE configurations
│   ├── attack.yaml                          # GCG + spectral monitoring configs
│   └── feature_extraction/                  # Per-model SAE feature extraction configs
│       ├── gemma_9b.yml
│       ├── llama_8b.yml
│       └── mistral_7b.yml
├── scripts/
│   ├── run_gcg.py                           # GCG suffix generation (RQ1, RQ2, RQ3)
│   ├── run_beast.py                         # BEAST suffix generation (RQ1)
│   ├── run_eval.py                          # Suffix evaluation / generation
│   ├── run_feature_extraction.py            # SAE feature extraction (RQ4)
│   ├── run_jaccard_analysis.py              # Jaccard similarity analysis (RQ4)
│   └── run_all_feature_extraction.sh        # Batch runner for all extraction jobs
├── src/
│   ├── __init__.py
│   ├── models.py                            # All SAE intervention model wrappers
│   ├── utils.py                             # Shared utilities
│   └── gcg_spectral.py                      # Spectral gradient monitoring (RQ4)
├── suffixes/                                # Pre-computed adversarial suffixes
├── random_suffixes/                         # Random baseline suffixes
├── envs/                                    # Conda environment files
├── requirements.txt
└── README.md
```

---

## SAE Model Wrappers

All SAE wrappers inherit from `torch.nn.Module` and hook the SAE into `forward()`, ensuring gradients flow through the SAE encode–decode during GCG optimization. Six SAE types are supported:

| `sae_type` | Source | Models | Notes |
|:---|:---|:---|:---|
| `none` | — | All | Bare HF model (baseline) |
| `goodfire` | [Goodfire SAEs](https://huggingface.co/Goodfire) | LLaMA-3.1-8B, LLaMA-3.3-70B | Linear enc + ReLU + linear dec |
| `mistral_res` | [JoshEngels](https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs) | Mistral-7B | Norm-scaling (constant=64) |
| `gemma_scope` | [Gemma Scope](https://huggingface.co/google/gemma-scope) via sae_lens | Gemma-2-2B/9B/27B, Qwen2.5-7B | `SAE.from_pretrained()` |
| `andyrdt_layer` | [andyrdt](https://huggingface.co/andyrdt/saes-llama-3.1-8b-instruct) via sae_lens | LLaMA-3.1-8B (layer ablation) | trainer_1 at each layer |
| `andyrdt_sparsity` | andyrdt via sae_lens + hot-swap | LLaMA-3.1-8B (sparsity ablation) | Hot-swaps `ae.pt` for k=32/128/256 |

---

## Model Configurations

### Primary Models (Table 7)

| Config | Model | SAE | Layer | Width |
|:---|:---|:---|:---|:---|
| `gemma_2b_sae` | Gemma-2-2B | Gemma Scope | 12 | 16K |
| `gemma_9b_sae` | Gemma-2-9B | Gemma Scope | 19 | 16K |
| `gemma_27b_sae` | Gemma-2-27B | Gemma Scope | 34 | 131K |
| `llama_8b_sae` | LLaMA-3.1-8B-Instruct | Goodfire | 19 | 65K |
| `llama_70b_sae` | LLaMA-3.3-70B-Instruct | Goodfire | 50 | 65K |
| `mistral_7b_sae` | Mistral-7B | JoshEngels | 16 | 65K |
| `qwen_7b_sae` | Qwen2.5-7B-Instruct | andyrdt | 19 | 131K |

### Ablation Configs

**Layer placement** (Tables 5–6, 13):
```
gemma_9b_sae_layer{5,10,20,30,35}
llama_8b_sae_layer{3,7,11,15,19,23,27}
qwen_7b_sae_layer{3,7,11,15,19,23,27}
```

**Sparsity** (Tables 3–4):
```
gemma_9b_sae_l20_w{16k,65k,131k}          # Gemma width sweep
llama_8b_sae_l19_k{32,64,128,256}          # LLaMA top-k sweep
```

---

## Usage

### 1. GCG Suffix Generation (RQ1, RQ2)

```bash
# SAE-augmented model
python scripts/run_gcg.py \
    --model_config llama_8b_sae \
    --harmbench_dir /path/to/data/ \
    --output_dir results/suffixes/llama_8b_sae/ \
    --batch_path batches/batch_1_of_12.pkl --batch_id 1

# Baseline
python scripts/run_gcg.py \
    --model_config llama_8b_base \
    --harmbench_dir /path/to/data/ \
    --output_dir results/suffixes/llama_8b_base/ --batch_id 1
```

### 2. BEAST Suffix Generation (RQ1)

```bash
python scripts/run_beast.py \
    --model_config gemma_2b_sae \
    --harmbench_dir /path/to/data/ \
    --output_dir results/beast/gemma_2b_sae/ --batch_id 1
```

### 3. Layer Placement Ablation (RQ3)

```bash
for layer in 3 7 11 15 19 23 27; do
    python scripts/run_gcg.py \
        --model_config llama_8b_sae_layer${layer} \
        --harmbench_dir /path/to/data/ \
        --output_dir results/layer_ablation/llama_8b_layer${layer}/
done
```

### 4. Sparsity Ablation (RQ3)

```bash
for k in 32 64 128 256; do
    python scripts/run_gcg.py \
        --model_config llama_8b_sae_l19_k${k} \
        --harmbench_dir /path/to/data/ \
        --output_dir results/sparsity/llama_8b_k${k}/
done
```

### 5. Spectral Gradient Analysis (RQ4)

```bash
python scripts/run_gcg.py \
    --model_config llama_8b_sae \
    --spectral_config spectral_full \
    --harmbench_dir /path/to/data/ \
    --output_dir results/spectral/llama_8b_sae/ --batch_id 1
```

### 6. Suffix Evaluation (Tables 1, 9, 10)

```bash
python scripts/run_eval.py \
    --model_config llama_8b_sae \
    --suffix_file suffixes/sampled_lightweight_suffixes.pkl \
    --harmbench_dir /path/to/data/ \
    --output_dir results/generation/
```

### 7. SAE Feature Extraction & Jaccard Analysis (RQ4, Figure 3)

```bash
# Extract features (adversarial)
python scripts/run_feature_extraction.py \
    --config configs/feature_extraction/gemma_9b.yml

# Extract features (random baselines)
python scripts/run_feature_extraction.py \
    --config configs/feature_extraction/gemma_9b.yml \
    --pickle_file random_suffixes/text_lowercase_suffixes.pkl \
    --output_dir results/features/gemma_9b_text_lowercase

# Or run all 18 jobs at once
bash scripts/run_all_feature_extraction.sh --parallel

# Compute Jaccard similarity
python scripts/run_jaccard_analysis.py \
    --feature_dir results/features/gemma_9b/ \
    --output_dir results/jaccard/ --top_k 100
```

---

## Environments

Conda environment files are provided in `envs/`:

```bash
# Primary environment (GCG, evaluation, feature extraction)
conda env create -f envs/sae_lens.yml

# SAE steering / intervention experiments
conda env create -f envs/saeSteer.yml
```

Or install from requirements:

```bash
conda create -n sae_robustness python=3.10 -y
conda activate sae_robustness
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|:---|:---|:---|
| `torch` | ≥ 2.1.0 | Core framework |
| `transformers` | ≥ 4.40.0 | Model loading |
| `nanogcg` | ≥ 0.2.0 | GCG attack |
| `sae-lens` | ≥ 4.0.0 | Gemma Scope / andyrdt SAEs |
| `safetensors` | ≥ 0.4.0 | Mistral SAE weights |

---

## Hardware Requirements
 
| Model | VRAM (approx) |
|:---|:---|
| Gemma-2-2B + SAE | ~12 GB |
| Gemma-2-9B + SAE | ~40 GB (fp32) |
| LLaMA-3.1-8B + SAE | ~20 GB (bf16) |
| Mistral-7B + SAE | ~20 GB (bf16) |
| Qwen2.5-7B + SAE | ~35 GB (fp32) |
| Gemma-2-27B + SAE | ~60 GB (bf16) |
| LLaMA-3.3-70B + SAE | ~150 GB (bf16) |
 
---
---

## Citation (placeholder)

```bibtex
@article{saiyed2026sparseshield,
  title   = {Towards Understanding the Robustness of Sparse Autoencoders},
  author  = {Saiyed, Ahson and Sadiekh, Sabrina and Agarwal, Chirag},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}
```

---

## License

[MIT License](LICENSE)