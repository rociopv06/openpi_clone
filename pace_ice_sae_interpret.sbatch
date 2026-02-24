#!/bin/bash
#SBATCH -J sae_interpret                # Job name
#SBATCH -N1 --ntasks-per-node=8         # 1 node, 8 CPU cores
#SBATCH --gres=gpu:H100:1               # Request 1 H100 GPU
#SBATCH --mem-per-gpu=80G               # Memory per GPU
#SBATCH -t 2:00:00                      # 2 hours (interpretation is faster than training)
#SBATCH -o sae_interpret_%j.out         # Output file
#SBATCH -e sae_interpret_%j.err         # Error file
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx
###############################################################################
# SAE Feature Interpretation on PACE ICE
#
# Runs interpretation analysis on a trained SAE checkpoint.
# Produces JSON reports + tensor files for further visualization.
#
# USAGE:
#   sbatch pace_ice_sae_interpret.sbatch
#
# OUTPUTS (in $INTERP_OUTPUT_DIR):
#   health_metrics.json              - Reconstruction quality, feature utilization
#   feature_statistics.json          - Per-feature mean, std, frequency, max
#   decoder_analysis.json            - Decoder weight structure, pairwise cosines
#   top_activating_examples.json     - Top-k data points per feature with metadata
#   feature_action_correlations.json - Which features correlate with action dims
#   decoder_weights.pt               - Raw decoder weight matrix
#   decoder_cosine_sim.pt            - Feature pairwise cosine similarity matrix
#   feature_action_correlation.pt    - Full correlation matrix [dict_size, action_dim]
#   feature_fire_counts.pt           - Per-feature activation counts
###############################################################################

echo "============================================================"
echo "SAE Interpretation Job on PACE ICE"
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
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"  # Auto-downloads pi0 checkpoints here
export HF_HOME="$PROJECT_DIR/.cache/huggingface"     # HuggingFace datasets + hub cache
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export TMPDIR="$PROJECT_DIR/tmp"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

# Create directories
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

echo "Storage locations:"
echo "  OpenPI repo: $OPENPI_DIR"
echo "  Checkpoint cache: $OPENPI_DATA_HOME"
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
# INTERPRETATION CONFIGURATION
# ============================================================================

# Model configuration
CONFIG="pi05_libero"

# ---- EDIT THESE TO MATCH YOUR TRAINED SAE ----
HOOK_POINT="lang_layer_9"
EXPANSION_FACTOR=4
K=32

# Experiment name (must match the training experiment name)
EXP_NAME="${CONFIG}_${HOOK_POINT}_k${K}_exp${EXPANSION_FACTOR}"

# Path to the trained SAE checkpoint
SAE_CHECKPOINT_DIR="$PROJECT_DIR/sae_checkpoints/$EXP_NAME"
SAE_CHECKPOINT="$SAE_CHECKPOINT_DIR/trainer_0/ae.pt"

# If the final ae.pt doesn't exist, try a step checkpoint
if [ ! -f "$SAE_CHECKPOINT" ]; then
    echo "Final checkpoint not found at $SAE_CHECKPOINT"
    # Try to find the latest step checkpoint
    LATEST=$(ls -t "$SAE_CHECKPOINT_DIR/trainer_0/checkpoints/" 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        SAE_CHECKPOINT="$SAE_CHECKPOINT_DIR/trainer_0/checkpoints/$LATEST"
        echo "Using step checkpoint: $SAE_CHECKPOINT"
    else
        echo "ERROR: No SAE checkpoint found in $SAE_CHECKPOINT_DIR"
        exit 1
    fi
fi

# Output directory for interpretation results
INTERP_OUTPUT_DIR="$PROJECT_DIR/sae_interpretations/$EXP_NAME"
mkdir -p "$INTERP_OUTPUT_DIR"

# Analysis parameters
N_BATCHES=200         # Batches for top-activating example search
DATA_BATCH_SIZE=16    # Model forward pass batch size
TOP_K=20              # Top-k examples per feature
EVAL_BATCHES=50       # Batches for health metric evaluation

echo "Configuration:"
echo "  Model config: $CONFIG"
echo "  Hook point: $HOOK_POINT"
echo "  Experiment name: $EXP_NAME"
echo "  SAE checkpoint: $SAE_CHECKPOINT"
echo "  Output dir: $INTERP_OUTPUT_DIR"
echo "  N batches: $N_BATCHES"
echo "  Top-K: $TOP_K"
echo "============================================================"

# ============================================================================
# RUN INTERPRETATION
# ============================================================================

cd "$OPENPI_DIR"

# Path to converted pi05 PyTorch checkpoint
PI05_CHECKPOINT="$PROJECT_DIR/pi05_pytorch/model.safetensors"

# Build the interpretation command
CMD="$PYTHON -m openpi.interpretability.interpret_sae \
    --config $CONFIG \
    --checkpoint_path $PI05_CHECKPOINT \
    --hook_point $HOOK_POINT \
    --sae_checkpoint $SAE_CHECKPOINT \
    --output_dir $INTERP_OUTPUT_DIR \
    --n_batches $N_BATCHES \
    --data_batch_size $DATA_BATCH_SIZE \
    --top_k $TOP_K \
    --eval_batches $EVAL_BATCHES"

echo "Running command:"
echo "$CMD"
echo "============================================================"

# Run the interpretation
eval $CMD

# ============================================================================
# COMPLETION
# ============================================================================

EXIT_CODE=$?
echo "============================================================"
echo "Interpretation completed with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "Results saved to: $INTERP_OUTPUT_DIR"
echo ""
echo "Output files:"
ls -lh "$INTERP_OUTPUT_DIR/"
echo "============================================================"

exit $EXIT_CODE
