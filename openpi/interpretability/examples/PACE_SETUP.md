# SAE Training on PACE ICE (Georgia Tech)

This guide walks you through setting up and running SAE training on GT PACE ICE cluster.

## 1. Initial Setup (One-Time)

### Connect to PACE ICE

```bash
ssh your_gtusername@login-ice.pace.gatech.edu
```

### Clone the Repository

```bash
cd $HOME
git clone --recursive https://github.com/YOUR_FORK/openpi.git openpi_clone
cd openpi_clone
```

### Create Conda Environment

```bash
# Load anaconda
module load anaconda3

# Create environment
conda create -n openpi python=3.11 -y
conda activate openpi

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install openpi
pip install -e .

# Install additional dependencies for SAE training
pip install wandb tqdm safetensors
```

### Download Model Checkpoint

For ALOHA sim, you can either:

**Option A: Use pretrained weights**
```bash
# Download from Hugging Face (if available)
mkdir -p $HOME/checkpoints/pi0_aloha_sim
# huggingface-cli download physical-intelligence/pi0 --local-dir $HOME/checkpoints/
```

**Option B: Train your own model first**
```bash
# See openpi training documentation
python scripts/train_pytorch.py pi0_aloha_sim --exp_name my_aloha_model
```

## 2. Configure the SLURM Script

Edit the SLURM script:

```bash
cd openpi_clone/openpi/interpretability/examples
nano pace_ice_sae_train.sbatch
```

**Required changes:**

1. **Email**: Change `YOUR_EMAIL@gatech.edu` to your GT email
2. **Checkpoint path**: Set `CHECKPOINT_PATH` to your model location
3. **OpenPI directory**: Verify `OPENPI_DIR` is correct

**Optional adjustments:**

- `CONFIG`: Training config (`pi0_aloha_sim`, `pi0_aloha`, etc.)
- `HOOK_POINTS`: Layers to analyze
- `EXPANSION_FACTOR`: SAE dictionary expansion (4, 8, or 16)
- `K`: Sparsity level (16, 32, 64)
- `STEPS`: Training steps (10000 for testing, 30000+ for real runs)

## 3. Submit Jobs

### Single Layer Training

```bash
cd $HOME/openpi_clone/openpi/interpretability/examples
sbatch pace_ice_sae_train.sbatch
```

### Multiple Layers (Array Job)

Train on multiple layers in parallel:

```bash
# Train on all 6 hook points defined in the script
sbatch --array=0-5 pace_ice_sae_train.sbatch
```

### Check Job Status

```bash
# Your jobs
squeue -u $USER

# Detailed job info
sacct -j <jobid>

# Cancel a job
scancel <jobid>
```

## 4. Monitor Training

### Check Output Logs

```bash
# Real-time output
tail -f sae_train_<jobid>.out

# Check errors
cat sae_train_<jobid>.err
```

### Wandb Dashboard

If you enabled wandb, visit https://wandb.ai to monitor:
- Training loss
- L0 sparsity
- Fraction of variance explained

## 5. GPU Options on PACE ICE

The script defaults to H100. You can modify for other GPUs:

| GPU | SBATCH Option | Memory | Notes |
|-----|--------------|--------|-------|
| H100 | `--gres=gpu:H100:1` | 80GB | Fastest, best for large models |
| A100-80GB | `--gres=gpu:1 -C A100-80GB` | 80GB | Good alternative |
| A100-40GB | `--gres=gpu:1 -C A100-40GB` | 40GB | Reduce batch size |
| V100-32GB | `--gres=gpu:1 -C V100-32GB` | 32GB | Reduce batch size |

**To change GPU**, edit the SBATCH header:
```bash
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-gpu=80G
```

## 6. Recommended Progressive Training Strategy

### Phase 1: Validate Pipeline (Quick Test)

```bash
# Modify script for quick test:
# STEPS=1000
# N_CTXS=5000
# Single hook point

sbatch pace_ice_sae_train.sbatch
```

### Phase 2: Single Task Full Training

```bash
# Full training on ALOHA sim
# STEPS=30000
# N_CTXS=20000
# Multiple layers

sbatch --array=0-5 pace_ice_sae_train.sbatch
```

### Phase 3: Scale to More Data

For Franka/OpenX data, you'll need to:
1. Prepare the dataset (see OpenPI data docs)
2. Update `CONFIG` to appropriate config
3. Increase `STEPS` and `N_CTXS`

## 7. Troubleshooting

### Out of Memory (OOM)

Reduce these parameters:
```bash
DATA_BATCH_SIZE=8   # Reduce from 16
N_CTXS=10000        # Reduce from 20000
BATCH_SIZE=2048     # Reduce from 4096
```

### Job Pending Too Long

Try different GPU:
```bash
#SBATCH --gres=gpu:A100:1  # Instead of H100
```

Or request less memory:
```bash
#SBATCH --mem-per-gpu=40G
```

### Module Not Found

Make sure conda environment is activated:
```bash
module load anaconda3
source activate openpi
```

### Checkpoint Not Found

Verify the path exists:
```bash
ls -la $HOME/checkpoints/pi0_aloha_sim/
```

## 8. Output Files

After training completes:

```
$HOME/sae_checkpoints/
└── pi0_aloha_sim_expert_layer_6_k32_exp4/
    ├── args.json              # Training arguments
    └── trainer_0/
        ├── config.json        # SAE configuration
        ├── ae.pt              # Final SAE weights
        └── checkpoints/
            ├── ae_5000.pt     # Intermediate checkpoints
            ├── ae_10000.pt
            └── ...
```

## 9. Loading a Trained SAE

```python
import torch
import json

# Load config
with open("sae_checkpoints/.../trainer_0/config.json") as f:
    config = json.load(f)

# Load SAE weights
from openpi.interpretability.dictionary_learning.dictionary_learning.dictionary import AutoEncoder

sae = AutoEncoder(
    activation_dim=config["trainer"]["activation_dim"],
    dict_size=config["trainer"]["dict_size"],
)
state_dict = torch.load("sae_checkpoints/.../trainer_0/ae.pt")
sae.load_state_dict(state_dict)
sae.eval()

# Use SAE
activations = ...  # [batch, d_model]
features = sae.encode(activations)  # Sparse features
reconstructed = sae.decode(features)
```

## 10. Resource Limits on ICE

- **Max walltime**: 16 hours (GPU), 18 hours (CPU)
- **Max GPU hours per job**: 16 hours
- **Max CPU hours per job**: 512 hours

For longer training, implement checkpointing and resubmit.
