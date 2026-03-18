#!/bin/bash
#SBATCH -J interpret              # Job name
#SBATCH --array=0-7               # 8 checkpoints to interpret
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=80G
#SBATCH -t 4:00:00
#SBATCH -o logs/interpret_%A_%a.out
#SBATCH -e logs/interpret_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

###############################################################################
# Interpret all remaining SAE checkpoints
# Checkpoints (index: name -> hook_point):
#   0: expert_layer_14
#   1: expert_layer_18
#   2: action_in_proj
#   3: lang_layer_3
#   4: lang_layer_6
#   5: lang_layer_12
#   6: lang_layer_17
#   7: expert_layer_6
###############################################################################

echo "Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  on $(hostname)  $(date)"

export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TMPDIR="$PROJECT_DIR/tmp"
export OPENPI_DATA_HOME="$PROJECT_DIR/openpi_cache"
mkdir -p "$TMPDIR" "$PROJECT_DIR/sae_interpretations" "$OPENPI_DIR/logs"

module load cuda/12.1

PYTHON="$OPENPI_DIR/.venv/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"

# Array of (exp_name, hook_point) pairs
EXP_NAMES=(
    "pi05_libero_expert_layer_14_k32_exp4"   # 0
    "pi05_libero_expert_layer_18_k32_exp4"   # 1
    "pi05_libero_action_in_proj_k32_exp4"    # 2
    "pi05_libero_lang_layer_3_k32_exp4"      # 3
    "pi05_libero_lang_layer_6_k32_exp4"      # 4
    "pi05_libero_lang_layer_12_k32_exp4"     # 5
    "pi05_libero_lang_layer_17_k32_exp4"     # 6
    "pi05_libero_expert_layer_6_k32_exp4"    # 7
)

HOOK_POINTS=(
    "expert_layer_14"   # 0
    "expert_layer_18"   # 1
    "action_in_proj"    # 2
    "lang_layer_3"      # 3
    "lang_layer_6"      # 4
    "lang_layer_12"     # 5
    "lang_layer_17"     # 6
    "expert_layer_6"    # 7
)

EXP_NAME="${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}"
HOOK_POINT="${HOOK_POINTS[$SLURM_ARRAY_TASK_ID]}"

SAE_CKPT="$PROJECT_DIR/sae_checkpoints/$EXP_NAME/trainer_0/ae.pt"
OUT_DIR="$PROJECT_DIR/sae_interpretations/$EXP_NAME"
PI05_CKPT="$PROJECT_DIR/pi05_pytorch/model.safetensors"

echo "Exp:       $EXP_NAME"
echo "Hook:      $HOOK_POINT"
echo "SAE ckpt:  $SAE_CKPT"
echo "Output:    $OUT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

mkdir -p "$OUT_DIR"

cd "$OPENPI_DIR"

$PYTHON interpret_sae.py \
    --config pi05_libero \
    --checkpoint_path "$PI05_CKPT" \
    --hook_point "$HOOK_POINT" \
    --sae_checkpoint "$SAE_CKPT" \
    --output_dir "$OUT_DIR" \
    --n_batches 200 \
    --data_batch_size 16 \
    --top_k 20 \
    --eval_batches 50

EXIT_CODE=$?
echo "Done (exit=$EXIT_CODE) at $(date)"
exit $EXIT_CODE
