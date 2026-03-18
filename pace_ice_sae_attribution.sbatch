#!/bin/bash
#SBATCH -J sae_attribution             # Job name
#SBATCH -N1 --ntasks-per-node=8        # 1 node, 8 CPU cores
#SBATCH --gres=gpu:H100:1              # Request 1 H100 GPU
#SBATCH --mem-per-gpu=80G              # Memory per GPU (full H100 — gradients need more memory)
#SBATCH -t 3:00:00                     # 3 hours
#SBATCH -o sae_attribution_%j.out      # Output file
#SBATCH -e sae_attribution_%j.err      # Error file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx
###############################################################################
# SAE Feature Attribution on PACE ICE
#
# Computes gradient × activation (GxA) attribution for each SAE feature at
# lang_layer_9, showing which features causally influence the training loss.
#
# Method:
#   For each data example:
#     1. Forward pass (with gradient tracking at lang_layer_9)
#     2. Compute grad = ∂(sum_loss)/∂(lang_layer_9 activations)
#     3. Project grad onto each SAE decoder direction d_i
#     4. Attribution_i = (grad · d_i) × f_i(activation)
#   Average |attribution_i| across dataset → feature importance scores
#
# USAGE:
#   sbatch pace_ice_sae_attribution.sbatch
#
# OUTPUTS (in $ATTR_OUTPUT_DIR):
#   feature_attribution_scores.pt      [dict_size] mean |GxA| per feature
#   feature_attribution_signed.pt      [dict_size] mean signed GxA
#   feature_grad_projection.pt         [dict_size] mean |grad · d_i|
#   feature_activation_magnitude.pt    [dict_size] mean |f_i|
#   feature_attribution_summary.json   top-feature rankings (human-readable)
###############################################################################

echo "============================================================"
echo "SAE Feature Attribution Job on PACE ICE"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================================"

# ============================================================================
# STORAGE SETUP
# ============================================================================
export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"

export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export TMPDIR="$PROJECT_DIR/tmp"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

echo "Storage:"
echo "  Project dir: $PROJECT_DIR"
echo "  OpenPI dir:  $OPENPI_DIR"
echo "============================================================"

# ============================================================================
# MODULE AND ENVIRONMENT SETUP
# ============================================================================
module load cuda/12.1

export PYTHON="$OPENPI_DIR/.venv/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "============================================================"

# ============================================================================
# ATTRIBUTION CONFIGURATION
# ============================================================================
CONFIG="pi05_libero"
HOOK_POINT="lang_layer_9"
K=32
EXPANSION_FACTOR=4

EXP_NAME="${CONFIG}_${HOOK_POINT}_k${K}_exp${EXPANSION_FACTOR}"

SAE_CHECKPOINT_DIR="$PROJECT_DIR/sae_checkpoints/$EXP_NAME"
SAE_CHECKPOINT="$SAE_CHECKPOINT_DIR/trainer_0/ae.pt"

if [ ! -f "$SAE_CHECKPOINT" ]; then
    echo "Final checkpoint not found at $SAE_CHECKPOINT"
    LATEST=$(ls -t "$SAE_CHECKPOINT_DIR/trainer_0/checkpoints/" 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        SAE_CHECKPOINT="$SAE_CHECKPOINT_DIR/trainer_0/checkpoints/$LATEST"
        echo "Using step checkpoint: $SAE_CHECKPOINT"
    else
        echo "ERROR: No SAE checkpoint found in $SAE_CHECKPOINT_DIR"
        exit 1
    fi
fi

PI05_CHECKPOINT="$PROJECT_DIR/pi05_pytorch/model.safetensors"

# Output into the same directory as the existing interpretations
ATTR_OUTPUT_DIR="$PROJECT_DIR/sae_interpretations/$EXP_NAME"
mkdir -p "$ATTR_OUTPUT_DIR"

# Number of batches for attribution (100 batches × ~16 samples = ~1600 examples)
N_BATCHES=100

echo "Configuration:"
echo "  Experiment:      $EXP_NAME"
echo "  Hook point:      $HOOK_POINT"
echo "  SAE checkpoint:  $SAE_CHECKPOINT"
echo "  PI05 checkpoint: $PI05_CHECKPOINT"
echo "  Output dir:      $ATTR_OUTPUT_DIR"
echo "  N batches:       $N_BATCHES"
echo "============================================================"

# ============================================================================
# RUN ATTRIBUTION
# ============================================================================
cd "$OPENPI_DIR"

CMD="$PYTHON feature_attribution_layer9.py \
    --config $CONFIG \
    --checkpoint_path $PI05_CHECKPOINT \
    --hook_point $HOOK_POINT \
    --sae_checkpoint $SAE_CHECKPOINT \
    --output_dir $ATTR_OUTPUT_DIR \
    --n_batches $N_BATCHES \
    --dtype bfloat16"

echo "Running:"
echo "$CMD"
echo "============================================================"

eval $CMD

# ============================================================================
# COMPLETION
# ============================================================================
EXIT_CODE=$?
echo "============================================================"
echo "Attribution job completed with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo ""
echo "Output files:"
ls -lh "$ATTR_OUTPUT_DIR/feature_attribution"* 2>/dev/null || echo "  (none found)"
echo "============================================================"

exit $EXIT_CODE
