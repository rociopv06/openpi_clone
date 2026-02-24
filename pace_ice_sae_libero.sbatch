#!/bin/bash
#SBATCH -J sae_pi05_libero           # Job name
#SBATCH -N1 --ntasks-per-node=8      # 1 node, 8 CPU cores
#SBATCH --gres=gpu:H100:1            # Request 1 H100 GPU
#SBATCH --mem-per-gpu=80G            # Memory per GPU
#SBATCH -t 5:00:00                   # 3 hours for testing (increase for full training)
#SBATCH -o sae_train_%j.out          # Output file
#SBATCH -e sae_train_%j.err          # Error file
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx
###############################################################################
# SAE Training on PACE ICE - Updated with Auto-Checkpoint Download
#
# This script trains Sparse Autoencoders on PI0.5 LIBERO activations.
# Checkpoints are automatically downloaded from Google Cloud Storage.
#
# USAGE:
#   Single job:
#     sbatch pace_ice_sae_libero_updated.sbatch
#
#   Array job for multiple layers (recommended):
#     sbatch --array=0-5 pace_ice_sae_libero_updated.sbatch
#
###############################################################################

echo "============================================================"
echo "SAE Training Job on PACE ICE"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================================"

# ============================================================================
# STORAGE SETUP - CRITICAL!
# ============================================================================

# Use project storage for everything (HOME is full)
export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"

# Cache directories in project storage (HOME is full — redirect everything)
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export TMPDIR="$PROJECT_DIR/tmp"

# Create directories
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

# Output directory for SAE checkpoints
export SAE_OUTPUT_DIR="$PROJECT_DIR/sae_checkpoints"
mkdir -p "$SAE_OUTPUT_DIR"

echo "Storage locations:"
echo "  OpenPI repo: $OPENPI_DIR"
echo "  Checkpoint cache: $OPENPI_DATA_HOME"
echo "  SAE outputs: $SAE_OUTPUT_DIR"
echo "============================================================"

# ============================================================================
# MODULE AND ENVIRONMENT SETUP
# ============================================================================

# Load required modules
module load cuda/12.1

# Use uv-managed venv (NOT conda — broken deps)
export PYTHON="$OPENPI_DIR/.venv/bin/python"

# Add src to python path for models imports
export PYTHONPATH="/storage/project/r-agarg35-0/rvaldes6/openpi_clone:/storage/project/r-agarg35-0/rvaldes6/openpi_clone/src:$PYTHONPATH"

# Verify GPU is available
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "============================================================"

# ============================================================================
# SAE TRAINING CONFIGURATION
# ============================================================================

# Model configuration - uses pi05_libero which auto-downloads the checkpoint!
CONFIG="pi05_libero"

# Hook points to train on (for array jobs, each task uses one)
HOOK_POINTS=(
    "expert_layer_0"
    "expert_layer_3"
    "expert_layer_6"
    "expert_layer_12"
    "expert_layer_18"
    "expert_layer_24"
)

# SAE Hyperparameters
EXPANSION_FACTOR=4    # Dictionary size = EXPANSION_FACTOR * d_model
K=32                  # Number of active features (sparsity)
LR=1e-4               # Learning rate
STEPS=10000           # Training steps
BATCH_SIZE=4096       # SAE training batch size
DATA_BATCH_SIZE=32    # Batch size for model forward passes (larger = fewer passes per buffer refresh)

# Data parameters
N_CTXS=2000           # Activation buffer size (buffer = n_ctxs * 50 = 100K activations)

# Wandb settings (optional)
USE_WANDB="--use_wandb"
WANDB_PROJECT="pi05-sae-libero-pace"
WANDB_ENTITY=""  # Your wandb username (leave empty if not using)

# ============================================================================
# DETERMINE HOOK POINT (for array jobs)
# ============================================================================

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    HOOK_POINT="${HOOK_POINTS[$SLURM_ARRAY_TASK_ID]}"
    echo "Array job: Using hook point index $SLURM_ARRAY_TASK_ID"
else
    HOOK_POINT="lang_layer_9"  # middle layer of PaliGemma LM (multimodal vision+language representations)
    echo "Single job: Using hook point $HOOK_POINT"
fi

# Experiment name
EXP_NAME="${CONFIG}_${HOOK_POINT}_k${K}_exp${EXPANSION_FACTOR}"

echo "Configuration:"
echo "  Model config: $CONFIG"
echo "  Hook point: $HOOK_POINT"
echo "  Experiment name: $EXP_NAME"
echo "  Checkpoints will auto-download to: $OPENPI_DATA_HOME"
echo "============================================================"

# ============================================================================
# RUN SAE TRAINING
# ============================================================================

cd "$OPENPI_DIR"

# Build the training command
PI05_CHECKPOINT="$PROJECT_DIR/pi05_pytorch/model.safetensors"

CMD="$PYTHON -m openpi.interpretability.train_sae \
    --config $CONFIG \
    --checkpoint_path $PI05_CHECKPOINT \
    --hook_point $HOOK_POINT \
    --expansion_factor $EXPANSION_FACTOR \
    --k $K \
    --lr $LR \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --data_batch_size $DATA_BATCH_SIZE \
    --n_ctxs $N_CTXS \
    --save_dir $SAE_OUTPUT_DIR \
    --exp_name $EXP_NAME"

# Add wandb flags if enabled
if [ -n "$USE_WANDB" ]; then
    CMD="$CMD $USE_WANDB --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        CMD="$CMD --wandb_entity $WANDB_ENTITY"
    fi
fi

echo "Running command:"
echo "$CMD"
echo "============================================================"

# Run the training
eval $CMD

# ============================================================================
# COMPLETION
# ============================================================================

EXIT_CODE=$?
echo "============================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "SAE checkpoint saved to: $SAE_OUTPUT_DIR/$EXP_NAME"
echo "============================================================"

exit $EXIT_CODE
