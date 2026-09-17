#!/bin/bash
#SBATCH -J sae_new_layers          # Job name
#SBATCH -N1 --ntasks-per-node=8   # 1 node, 8 CPU cores
#SBATCH --gres=gpu:H100:1          # Request 1 H100 GPU
#SBATCH --mem-per-gpu=80G          # Memory per GPU
#SBATCH -t 2:00:00                 # 2 hours per layer
#SBATCH -o sae_train_%A_%a.out     # Output file (array job ID + task ID)
#SBATCH -e sae_train_%A_%a.err     # Error file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

###############################################################################
# Array job: train SAEs on 5 new hook points
#
# Submit with:
#   sbatch --array=0-4 pace_ice_sae_new_layers.sbatch
#
# Array index -> hook point:
#   0 -> lang_layer_3    (early VLM: vision tokens being read)
#   1 -> lang_layer_6    (mid-early VLM: language grounding)
#   2 -> lang_layer_12   (mid-late VLM: task abstraction)
#   3 -> lang_layer_17   (final VLM output -> action expert input)
#   4 -> expert_layer_6  (mid action denoiser)
###############################################################################

echo "============================================================"
echo "SAE Training — New Layers Array Job"
echo "Job: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)  Date: $(date)"
echo "============================================================"

export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

export SAE_OUTPUT_DIR="$PROJECT_DIR/sae_checkpoints"
mkdir -p "$SAE_OUTPUT_DIR"

module load cuda/12.1
export PYTHON="$OPENPI_DIR/.venv/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "============================================================"

# ── Hook point selection ──────────────────────────────────────────────────────
HOOK_POINTS=(
    "lang_layer_3"    # 0: early VLM
    "lang_layer_6"    # 1: mid-early VLM
    "lang_layer_12"   # 2: mid-late VLM
    "lang_layer_17"   # 3: final VLM layer
    "expert_layer_6"  # 4: mid action expert
)

HOOK_POINT="${HOOK_POINTS[$SLURM_ARRAY_TASK_ID]}"
echo "Hook point: $HOOK_POINT"

# ── SAE hyperparameters (same as existing experiments) ───────────────────────
CONFIG="pi05_libero"
EXPANSION_FACTOR=4
K=32
LR=1e-4
STEPS=10000
BATCH_SIZE=4096
DATA_BATCH_SIZE=32
N_CTXS=2000

EXP_NAME="pi05_libero_${HOOK_POINT}_k${K}_exp${EXPANSION_FACTOR}"
PI05_CHECKPOINT="$PROJECT_DIR/pi05_pytorch/model.safetensors"

echo "Experiment: $EXP_NAME"
echo "============================================================"

cd "$OPENPI_DIR"

$PYTHON -m openpi.interpretability.train_sae \
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
    --exp_name $EXP_NAME \
    --use_wandb \
    --wandb_project pi05-sae-libero-pace

EXIT_CODE=$?
echo "============================================================"
echo "Done. Exit: $EXIT_CODE  Date: $(date)"
echo "Checkpoint: $SAE_OUTPUT_DIR/$EXP_NAME"
echo "============================================================"
exit $EXIT_CODE
