#!/bin/bash

# ===========================================
# Run all SAE feature extraction jobs
# Uses 3 GPUs in parallel
# ===========================================

# Gemma configs (GPU 0)
gemma_configs=(
    "C00_gemma_config_R1.yml"
    "C00_gemma_config_R2.yml"
    "C00_gemma_config_R3.yml"
    "C00_gemma_config_R4.yml"
    "C00_gemma_config_R5.yml"
)

# Llama configs (GPU 1)
llama_configs=(
    "C01_llama_config_R1.yml"
    "C01_llama_config_R2.yml"
    "C01_llama_config_R3.yml"
    "C01_llama_config_R4.yml"
    "C01_llama_config_R5.yml"
)

# Mistral configs (GPU 2)
mistral_configs=(
    "C02_mistral_config_R1.yml"
    "C02_mistral_config_R2.yml"
    "C02_mistral_config_R3.yml"
    "C02_mistral_config_R4.yml"
    "C02_mistral_config_R5.yml"
)

# Function to run configs sequentially on a specific GPU
run_on_gpu() {
    local gpu=$1
    local script=$2
    shift 2
    local configs=("$@")
    
    for config in "${configs[@]}"; do
        echo "========================================"
        echo "[GPU $gpu] Running: $config"
        echo "========================================"
        CUDA_VISIBLE_DEVICES=$gpu python "$script" --config "$config"
        
        if [ $? -ne 0 ]; then
            echo "❌ [GPU $gpu] Failed on $config"
            return 1
        fi
        
        echo "✓ [GPU $gpu] Completed $config"
        echo ""
    done
    
    echo "✓ [GPU $gpu] All jobs completed"
}

# Run all three in parallel
run_on_gpu 0 "00_GEMMA_SAE_FEATURE_EXT.py" "${gemma_configs[@]}" &
PID_GEMMA=$!

run_on_gpu 1 "01_LLAMA_SAE_FEATURE_EXT.py" "${llama_configs[@]}" &
PID_LLAMA=$!

run_on_gpu 2 "02_MISTRAL_SAE_FEATURE_EXT.py" "${mistral_configs[@]}" &
PID_MISTRAL=$!

# Wait for all to complete
echo "========================================"
echo "Running 3 jobs in parallel..."
echo "  GPU 0: Gemma (PID $PID_GEMMA)"
echo "  GPU 1: Llama (PID $PID_LLAMA)"
echo "  GPU 2: Mistral (PID $PID_MISTRAL)"
echo "========================================"

wait $PID_GEMMA
STATUS_GEMMA=$?

wait $PID_LLAMA
STATUS_LLAMA=$?

wait $PID_MISTRAL
STATUS_MISTRAL=$?

# Summary
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
[ $STATUS_GEMMA -eq 0 ] && echo "✓ Gemma: SUCCESS" || echo "❌ Gemma: FAILED"
[ $STATUS_LLAMA -eq 0 ] && echo "✓ Llama: SUCCESS" || echo "❌ Llama: FAILED"
[ $STATUS_MISTRAL -eq 0 ] && echo "✓ Mistral: SUCCESS" || echo "❌ Mistral: FAILED"
echo "========================================"