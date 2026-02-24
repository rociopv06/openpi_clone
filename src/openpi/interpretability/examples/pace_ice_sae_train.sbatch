#!/bin/bash
#SBATCH -J sae_pi0_liber                 # Job name
#SBATCH -N1 --ntasks-per-node=8          # 1 node, 8 CPU cores
#SBATCH --gres=gpu:H100:1                # Request 1 H100 GPU (or use A100:1, V100:1)
#SBATCH --mem-per-gpu=80G                # Memory per GPU (H100 has 80GB)
#SBATCH -t 3:00:00                       # Set 3 hours for GPU 
#SBATCH -o sae_train_%j.out              # Output file
#SBATCH -e sae_train_%j.err              # Error file
#SBATCH --mail-type=BEGIN,END,FAIL       # Email notifications
#SBATCH --mail-user=rvaldes6@gatech.edu  # Mail
#SBATCH --account=gts-agarg35		 # Account
###############################################################################
# SAE Training on PACE ICE Cluster (Georgia Tech)
#
# This script trains Sparse Autoencoders on PI0 model activations using
# the dictionary_learning framework.
#
# BEFORE RUNNING:
# 1. Change YOUR_EMAIL@gatech.edu to your actual email
# 2. Set CHECKPOINT_PATH to your model checkpoint
# 3. Adjust HOOK_POINT for the layer you want to analyze
# 4. Make sure your conda environment is set up (see setup instructions below)
#
# USAGE:
#   Single job:
#     sbatch pace_ice_sae_train.sbatch
#
#   Array job for multiple layers:
#     sbatch --array=0-5 pace_ice_sae_train.sbatch
#
###############################################################################

# ============================================================================
# CONFIGURATION - MODIFY THESE
# ============================================================================

# Your openpi installation directory
OPENPI_DIR="$HOME/openpi_clone"

# Model configuration (available: pi0_aloha_sim, pi0_aloha, pi0_aloha_pen_uncap, etc.)
CONFIG="pi0_aloha_sim"

# Path to model checkpoint (.safetensors file)
# For pi0_aloha_sim, you can use the pretrained weights or your finetuned checkpoint
CHECKPOINT_PATH="$HOME/checkpoints/pi0_aloha_sim/model.safetensors"

# Hook points to train on (for array jobs, each array index uses one)
HOOK_POINTS=(
    "expert_layer_0"
    "expert_layer_3"
    "expert_layer_6"
    "lang_layer_6"
    "lang_layer_12"
    "action_out_proj"
)

# SAE Hyperparameters
EXPANSION_FACTOR=4    # Dictionary size = EXPANSION_FACTOR * d_model (try 4, 8, 16)
K=32                  # Sparsity: number of active features (try 16, 32, 64)
LR=1e-4               # Learning rate
STEPS=30000           # Training steps
BATCH_SIZE=4096       # SAE training batch size

# Data parameters
DATA_BATCH_SIZE=16    # Batch size for model forward passes (reduce if OOM)
N_CTXS=20000          # Number of contexts in activation buffer

# Output directory (use scratch for large files)
SAVE_DIR="$HOME/sae_checkpoints"

# Wandb settings (optional - comment out USE_WANDB to disable)
USE_WANDB="--use_wandb"
WANDB_PROJECT="pi0-sae-pace"
WANDB_ENTITY=""  # Your wandb username

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "============================================================"
echo "SAE Training Job on PACE ICE"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================================"

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Load required modules
module load anaconda3
module load cuda/12.1

# Activate conda environment (create this first - see setup instructions)
# To create: conda create -n openpi python=3.11 && conda activate openpi
# Then: pip install -e /path/to/openpi_clone
source activate openpi

# Verify GPU is available
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "============================================================"

# ============================================================================
# DETERMINE HOOK POINT
# ============================================================================

# For array jobs, use SLURM_ARRAY_TASK_ID to select hook point
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    HOOK_POINT="${HOOK_POINTS[$SLURM_ARRAY_TASK_ID]}"
    echo "Array job: Using hook point index $SLURM_ARRAY_TASK_ID"
else
    HOOK_POINT="${HOOK_POINTS[0]}"
    echo "Single job: Using first hook point"
fi

# Experiment name includes config, hook point, and key hyperparameters
EXP_NAME="${CONFIG}_${HOOK_POINT}_k${K}_exp${EXPANSION_FACTOR}"

echo "Configuration: $CONFIG"
echo "Hook Point: $HOOK_POINT"
echo "Experiment: $EXP_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "============================================================"

# Create output directory
mkdir -p "$SAVE_DIR"

# ============================================================================
# RUN SAE TRAINING
# ============================================================================

cd "$OPENPI_DIR"

# Build the command
CMD="python -m openpi.interpretability.train_sae \
    --config $CONFIG \
