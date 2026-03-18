#!/bin/bash
#SBATCH -J interp_vision
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=200G
#SBATCH -t 2:00:00
#SBATCH -o logs/interp_vision_%j.out
#SBATCH -e logs/interp_vision_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

echo "Job $SLURM_JOB_ID  on $(hostname)  $(date)"

export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$OPENPI_DIR/logs"

module load cuda/12.1
PYTHON="$OPENPI_DIR/.venv/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

EXP_NAME="pi05_libero_vision_proj_k32_exp4"
HOOK_POINT="vision_proj"
SAE_CKPT="$PROJECT_DIR/sae_checkpoints/$EXP_NAME/trainer_0/ae.pt"
OUT_DIR="$PROJECT_DIR/sae_interpretations/$EXP_NAME"
PI05_CKPT="$PROJECT_DIR/pi05_pytorch/model.safetensors"

mkdir -p "$OUT_DIR"
cd "$OPENPI_DIR"

$PYTHON interpret_sae.py \
    --config pi05_libero \
    --checkpoint_path "$PI05_CKPT" \
    --hook_point "$HOOK_POINT" \
    --sae_checkpoint "$SAE_CKPT" \
    --output_dir "$OUT_DIR" \
    --n_batches 300 \
    --data_batch_size 16 \
    --top_k 50 \
    --eval_batches 50

EXIT_CODE=$?
echo "Done (exit=$EXIT_CODE) at $(date)"
exit $EXIT_CODE
