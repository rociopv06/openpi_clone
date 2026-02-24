#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --partition=gpu           # Adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1         # Single GPU per job (SAE training is not distributed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sae_%j.out
#SBATCH --error=logs/sae_%j.err

###############################################################################
# SAE Training SLURM Script for PI0 Interpretability
#
# This script trains Sparse Autoencoders on PI0 model activations.
#
# Usage:
#   # Single task, single layer
#   sbatch slurm_sae_train.sh
#
#   # Run multiple experiments (different layers) with job array:
#   sbatch --array=0-5 slurm_sae_train.sh
#
# Configuration:
#   Modify the variables below to customize your training run.
###############################################################################

# ============================================================================
# CONFIGURATION - Modify these for your setup
# ============================================================================

# Model config (from openpi training configs)
CONFIG="pi0_aloha_sim"

# Path to model checkpoint (required)
CHECKPOINT_PATH="/path/to/your/checkpoint/model.safetensors"

# Hook points to train on (for job arrays)
HOOK_POINTS=(
    "expert_layer_0"
    "expert_layer_3"
    "expert_layer_6"
    "lang_layer_6"
    "lang_layer_12"
    "lang_layer_18"
)

# SAE hyperparameters
EXPANSION_FACTOR=4   # Dictionary size = EXPANSION_FACTOR * d_model
K=32                 # Number of active features (sparsity)
LR=1e-4              # Learning rate
STEPS=30000          # Training steps
BATCH_SIZE=4096      # SAE training batch size

# Data parameters
DATA_BATCH_SIZE=32   # Batch size for model forward passes
N_CTXS=30000         # Number of contexts in activation buffer

# Logging
USE_WANDB="--use_wandb"  # Remove to disable wandb
WANDB_PROJECT="pi0-sae"
WANDB_ENTITY=""           # Your wandb username or team

# Output
SAVE_DIR="/path/to/save/sae_checkpoints"
LOG_DIR="./logs"

# ============================================================================
# SETUP
# ============================================================================

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

# Get hook point for this job (supports job arrays)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    HOOK_POINT="${HOOK_POINTS[$SLURM_ARRAY_TASK_ID]}"
else
    HOOK_POINT="${HOOK_POINTS[0]}"
fi

# Experiment name
EXP_NAME="${CONFIG}_${HOOK_POINT}_k${K}_exp${EXPANSION_FACTOR}"

echo "============================================================"
echo "SAE Training Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Hook Point: $HOOK_POINT"
echo "Experiment: $EXP_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "============================================================"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules (adjust for your cluster)
# module load cuda/12.1
# module load anaconda/2023.09

# Activate conda environment
# source activate openpi

# Or use uv
cd /path/to/openpi_clone
source .venv/bin/activate

# ============================================================================
# RUN TRAINING
# ============================================================================

python -m openpi.interpretability.train_sae \
    --config "$CONFIG" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --hook_point "$HOOK_POINT" \
    --expansion_factor $EXPANSION_FACTOR \
    --k $K \
    --lr $LR \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --data_batch_size $DATA_BATCH_SIZE \
    --n_ctxs $N_CTXS \
    --exp_name "$EXP_NAME" \
    --save_dir "$SAVE_DIR" \
    $USE_WANDB \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --device cuda

echo "============================================================"
echo "Training Complete!"
echo "Checkpoints saved to: $SAVE_DIR/$EXP_NAME"
echo "============================================================"
