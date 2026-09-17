#!/bin/bash
#SBATCH -J verify_sae
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH -t 0:30:00
#SBATCH -o logs/verify_%j.out
#SBATCH -e logs/verify_%j.err
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

cd "$OPENPI_DIR"

$PYTHON verify_top_examples.py \
    --config pi05_libero \
    --checkpoint_path "$PROJECT_DIR/pi05_pytorch/model.safetensors" \
    --sae_checkpoint "$PROJECT_DIR/sae_checkpoints/pi05_libero_lang_layer_9_k32_exp4/trainer_0/ae.pt" \
    --interp_dir "$PROJECT_DIR/sae_interpretations/pi05_libero_lang_layer_9_k32_exp4" \
    --hook_point lang_layer_9 \
    --n_features 20 \
    --top_k 5

EXIT_CODE=$?
echo "Done (exit=$EXIT_CODE) at $(date)"
exit $EXIT_CODE
