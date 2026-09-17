#!/bin/bash
#SBATCH -J ms_label_sae            # Job name
#SBATCH -N1 --ntasks-per-node=8   # 1 node, 8 CPU cores
#SBATCH --gres=gpu:H100:1          # GPU for DINOv2
#SBATCH --mem-per-gpu=40G          # 40GB is plenty
#SBATCH -t 3:00:00                 # ~2h for 1331 features
#SBATCH -o ms_label_%j.out
#SBATCH -e ms_label_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

###############################################################################
# MS score computation + LLM label prompt generation for lang_layer_9 SAE
#
# To also run Claude labeling:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   Then add: --export=ALL,LABEL_WITH_CLAUDE=1
###############################################################################

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$TMPDIR"

module load cuda/12.1

PYTHON="$PROJECT_DIR/conda_envs/py311_openpi/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"

echo "GPU:"
nvidia-smi --query-gpu=name,memory.free --format=csv
echo "Python: $PYTHON"

cd "$OPENPI_DIR"

$PYTHON compute_ms_and_label.py \
    --sae_name pi05_libero_lang_layer_9_k32_exp4 \
    --ms_threshold 0.45 \
    --top_n 10 \
    --device cuda

echo "Done at $(date)"
