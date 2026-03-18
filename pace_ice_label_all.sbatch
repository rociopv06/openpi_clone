#!/bin/bash
#SBATCH -J sae_label                # Job name
#SBATCH --array=0-5                 # 6 checkpoints
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --gres=gpu:H100:1           # DINOv2 (Phase 1) + Llama Vision (Phase 2)
#SBATCH --mem-per-gpu=80G
#SBATCH -t 4:00:00
#SBATCH -o logs/label_%A_%a.out
#SBATCH -e logs/label_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

###############################################################################
# Phase 1: compute MS scores + generate label_prompts.jsonl  (DINOv2, GPU)
# Phase 2: label features with Llama-3.2-11B-Vision           (local, no API)
#
# Submit:
#   sbatch pace_ice_label_all.sbatch
#
#   0 -> lang_layer_3
#   1 -> lang_layer_6
#   2 -> lang_layer_9
#   3 -> lang_layer_12
#   4 -> lang_layer_17
#   5 -> expert_layer_0
###############################################################################

echo "Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  on $(hostname)  $(date)"

export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$TMPDIR" "$OPENPI_DIR/logs"

module load cuda/12.1
PYTHON="$PROJECT_DIR/conda_envs/py311_openpi/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

SAE_NAMES=(
    "pi05_libero_lang_layer_3_k32_exp4"    # 0
    "pi05_libero_lang_layer_6_k32_exp4"    # 1
    "pi05_libero_lang_layer_9_k32_exp4"    # 2
    "pi05_libero_lang_layer_12_k32_exp4"   # 3
    "pi05_libero_lang_layer_17_k32_exp4"   # 4
    "pi05_libero_expert_layer_0_k32_exp4"  # 5
)

SAE_NAME="${SAE_NAMES[$SLURM_ARRAY_TASK_ID]}"
echo "SAE: $SAE_NAME"

cd "$OPENPI_DIR"

# ── Phase 1: MS scores + label prompts ───────────────────────────────────────
echo "=== Phase 1: MS scores + prompt generation ==="
$PYTHON compute_ms_and_label.py \
    --sae_name "$SAE_NAME" \
    --ms_threshold 0.45 \
    --top_n 10 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 1 failed for $SAE_NAME"
    exit 1
fi

# ── Phase 2: local Llama Vision labeling ─────────────────────────────────────
echo "=== Phase 2: LLM labeling with Llama-3.2-11B-Vision ==="
$PYTHON run_local_labels.py \
    --sae_name "$SAE_NAME" \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct

EXIT_CODE=$?
echo "Done (exit=$EXIT_CODE) at $(date)"
exit $EXIT_CODE
