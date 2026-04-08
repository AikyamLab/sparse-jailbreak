#!/bin/bash
# ============================================================================
# run_all_feature_extraction.sh
# 
# Runs SAE feature extraction for all model families × all suffix types.
# 3 models × 6 suffix sources = 18 jobs.
#
# Usage:
#   bash scripts/run_all_feature_extraction.sh
#
# Parallel (one model per GPU):
#   bash scripts/run_all_feature_extraction.sh --parallel
# ============================================================================

set -e

SCRIPT="scripts/run_feature_extraction.py"

# Suffix types: adversarial + 5 random baselines
declare -A SUFFIXES
SUFFIXES[adversarial]="suffixes/lightweight_suffixes_with_beast_1_3.pkl"
SUFFIXES[alphanumeric]="random_suffixes/full_alphanumeric_symbols_suffixes.pkl"
SUFFIXES[mixed_case]="random_suffixes/mixed_case_suffixes.pkl"
SUFFIXES[text_lowercase]="random_suffixes/text_lowercase_suffixes.pkl"
SUFFIXES[text_numbers]="random_suffixes/text_numbers_suffixes.pkl"
SUFFIXES[unicode_text]="random_suffixes/unicode_text_suffixes.pkl"

# Model configs
CONFIGS=(
    "configs/feature_extraction/gemma_9b.yml"
    "configs/feature_extraction/llama_8b.yml"
    "configs/feature_extraction/mistral_7b.yml"
)

# Short names for output dirs
declare -A MODEL_NAMES
MODEL_NAMES["configs/feature_extraction/gemma_9b.yml"]="gemma_9b"
MODEL_NAMES["configs/feature_extraction/llama_8b.yml"]="llama_8b"
MODEL_NAMES["configs/feature_extraction/mistral_7b.yml"]="mistral_7b"


run_model() {
    local config=$1
    local gpu=$2
    local model_name=${MODEL_NAMES[$config]}

    for suffix_name in adversarial alphanumeric mixed_case text_lowercase text_numbers unicode_text; do
        local pickle=${SUFFIXES[$suffix_name]}
        local outdir="results/features/${model_name}_${suffix_name}"

        echo "========================================"
        echo "[${model_name}] ${suffix_name}"
        echo "  Config:  ${config}"
        echo "  Input:   ${pickle}"
        echo "  Output:  ${outdir}"
        echo "  Device:  cuda:${gpu}"
        echo "========================================"

        python $SCRIPT \
            --config "$config" \
            --pickle_file "$pickle" \
            --output_dir "$outdir" \
            --device "cuda:${gpu}"

        echo "✓ [${model_name}] ${suffix_name} done"
        echo ""
    done

    echo "✓ [${model_name}] ALL DONE"
}


if [ "$1" == "--parallel" ]; then
    echo "Running 3 models in parallel (one per GPU)..."
    run_model "${CONFIGS[0]}" 0 &
    PID0=$!
    run_model "${CONFIGS[1]}" 1 &
    PID1=$!
    run_model "${CONFIGS[2]}" 2 &
    PID2=$!

    wait $PID0; S0=$?
    wait $PID1; S1=$?
    wait $PID2; S2=$?

    echo ""
    echo "========================================"
    echo "SUMMARY"
    echo "========================================"
    [ $S0 -eq 0 ] && echo "✓ Gemma: SUCCESS" || echo "✗ Gemma: FAILED"
    [ $S1 -eq 0 ] && echo "✓ LLaMA: SUCCESS" || echo "✗ LLaMA: FAILED"
    [ $S2 -eq 0 ] && echo "✓ Mistral: SUCCESS" || echo "✗ Mistral: FAILED"
    echo "========================================"
else
    echo "Running sequentially (use --parallel for GPU parallelism)..."
    for config in "${CONFIGS[@]}"; do
        run_model "$config" 0
    done
    echo "ALL DONE"
fi
