#!/bin/bash
#SBATCH -J llm_label               # Job name
#SBATCH -N1 --ntasks-per-node=2    # CPU only, just API calls
#SBATCH --mem=8G
#SBATCH -t 2:00:00                 # ~1h for ~500 features at 0.5s/call
#SBATCH -o llm_label_%j.out
#SBATCH -e llm_label_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

###############################################################################
# LLM feature labeling — reads label_prompts.jsonl, calls Anthropic API
#
# REQUIRED: set your API key before submitting:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   sbatch --export=ALL pace_ice_llm_label.sbatch
#
# Supports resume: re-run after interruption, already-labeled features skipped.
###############################################################################

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Submit with:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-... && sbatch --export=ALL pace_ice_llm_label.sbatch"
    exit 1
fi

export PROJECT_DIR="/storage/project/r-agarg35-0/rvaldes6"
export OPENPI_DIR="$PROJECT_DIR/openpi_clone"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$TMPDIR"

PYTHON="$PROJECT_DIR/conda_envs/py311_openpi/bin/python"
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"

echo "Python: $PYTHON"

cd "$OPENPI_DIR"

$PYTHON run_llm_labels.py \
    --sae_name pi05_libero_lang_layer_9_k32_exp4 \
    --model claude-opus-4-6 \
    --delay 0.5

echo "Done at $(date)"
