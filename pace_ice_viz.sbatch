#!/bin/bash
#SBATCH -J sae_viz
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH -t 1:30:00
#SBATCH -o /storage/project/r-agarg35-0/rvaldes6/openpi_clone/logs/viz_%j.out
#SBATCH -e /storage/project/r-agarg35-0/rvaldes6/openpi_clone/logs/viz_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvaldes6@gatech.edu
#SBATCH --account=gts-agarg35-ideasci23_dgx

# CPU-only viz — uses conda env (pandas, pyarrow, PIL)
PYTHON=/storage/project/r-agarg35-0/rvaldes6/conda_envs/py311_openpi/bin/python
OPENPI_DIR=/storage/project/r-agarg35-0/rvaldes6/openpi_clone
export PYTHONPATH="$OPENPI_DIR:$OPENPI_DIR/src:$PYTHONPATH"
export HF_HOME=/storage/project/r-agarg35-0/rvaldes6/.cache/huggingface

cd $OPENPI_DIR
$PYTHON visualize_top_examples.py --top_n_features 30 --top_k_examples 5

EXIT_CODE=$?
echo "Done (exit=$EXIT_CODE) at $(date)"
exit $EXIT_CODE
