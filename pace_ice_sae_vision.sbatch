#!/bin/bash
#SBATCH -J sae_vision               # Job name
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=80G
#SBATCH -t 3:00:00
#SBATCH -o logs/sae_vision_%j.out
#SBATCH -e logs/sae_vision_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

###############################################################################
# Train vision-only SAE on the multi-modal projector output (vision_proj).
#
# Hook point: vision_proj
#   = output of paligemma.multi_modal_projector (SigLIP -> language model bridge)
#   = [batch, 256, 2048]  per camera (2048 = Gemma language model width)
#
# Each forward pass fires the hook once per camera (2 cameras for LIBERO).
# The hook captures the LAST camera's projector output (256 tokens × 2048-dim).
# SAE training sees pure visual token activations — no language contamination.
#
# This lets us understand:
#   - What visual features does PI0 extract from robot workspace images?
#   - Are these features generic (like COCO-type objects) or robot-specific?
#   - How do visual features at vision_proj compare to lang_layer_N features?
#
# Exp name: pi05_libero_vision_proj_k32_exp4
###############################################################################

echo "Job $SLURM_JOB_ID  on $(hostname)  $(date)"

export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$OPENPI_DIR/logs"

export SAE_OUTPUT_DIR="$PROJECT_DIR/sae_checkpoints"
mkdir -p "$SAE_OUTPUT_DIR"

module load cuda/12.1
PYTHON="$OPENPI_DIR/.venv/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

HOOK_POINT="vision_proj"
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

echo "Hook:  $HOOK_POINT"
echo "Exp:   $EXP_NAME"
echo "Steps: $STEPS  k=$K  dict_size=$(( 2048 * EXPANSION_FACTOR ))"

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
echo "Done (exit=$EXIT_CODE) at $(date)"
exit $EXIT_CODE
