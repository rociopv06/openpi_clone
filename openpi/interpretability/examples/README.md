# PI0 SAE Training Examples

This directory contains examples for training Sparse Autoencoders (SAEs) on PI0 model activations.

## Quick Start

### 1. List Available Hook Points

```bash
python -m openpi.interpretability.train_sae --list_hooks
```

Available hook points include:
- `vision_out`: SigLIP vision encoder output
- `vision_proj`: Multi-modal projector output
- `lang_layer_0` to `lang_layer_26`: Language model transformer layers
- `lang_mlp_0` to `lang_mlp_26`: Language model MLP activations
- `expert_layer_0` to `expert_layer_26`: Action expert transformer layers
- `expert_mlp_0` to `expert_mlp_26`: Action expert MLP activations
- `action_in_proj`, `action_out_proj`: Action projection layers
- `state_proj`: State projection (pi0 only)

### 2. Train SAE on Single Hook Point

```bash
python -m openpi.interpretability.train_sae \
    --config pi0_aloha_sim \
    --checkpoint_path /path/to/model.safetensors \
    --hook_point expert_layer_6 \
    --expansion_factor 4 \
    --k 32 \
    --steps 30000 \
    --exp_name my_sae_experiment \
    --use_wandb
```

### 3. Collect Activations Only (for later training)

```bash
python openpi/interpretability/examples/collect_activations.py \
    --config pi0_aloha_sim \
    --checkpoint_path /path/to/model.safetensors \
    --hook_points expert_layer_6 lang_layer_12 \
    --output_dir ./activations \
    --num_batches 1000
```

## Cluster Training with SLURM

### Single Job

```bash
sbatch openpi/interpretability/examples/slurm_sae_train.sh
```

### Array Job (Multiple Layers)

Train SAEs on multiple layers in parallel:

```bash
sbatch --array=0-5 openpi/interpretability/examples/slurm_sae_train.sh
```

This will train on layers defined in the `HOOK_POINTS` array in the script.

## Progressive Training Strategy

As mentioned in the goal, the recommended approach is:

### Phase 1: Single Task, Small Dataset
Start with a specific task (e.g., ALOHA simulation) to validate the pipeline:

```bash
python -m openpi.interpretability.train_sae \
    --config pi0_aloha_sim \
    --checkpoint_path /path/to/aloha_checkpoint.safetensors \
    --hook_point expert_layer_6 \
    --expansion_factor 4 \
    --k 32 \
    --steps 10000 \
    --n_ctxs 10000 \
    --exp_name phase1_aloha_expert_l6
```

### Phase 2: All Franka Tasks in OpenX
After validation, scale to all Franka data:

```bash
# Submit array job for multiple layers
sbatch --array=0-11 slurm_sae_train_franka.sh
```

## Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `expansion_factor` | Ratio of SAE dict size to input dim | 4, 8, 16 |
| `k` | Number of active features (sparsity) | 16, 32, 64, 128 |
| `lr` | Learning rate | 1e-4, 3e-4 |
| `steps` | Training steps | 20k-100k |
| `batch_size` | SAE training batch size | 2048, 4096, 8192 |
| `n_ctxs` | Contexts in activation buffer | 20k-50k |

## SAE Trainers Available

The dictionary_learning submodule provides several SAE training algorithms:

- **BatchTopKTrainer** (recommended): Uses top-k activation selection per batch
- **StandardTrainer**: Classic L1-regularized SAE
- **TopKTrainer**: Per-example top-k selection
- **JumpReLUTrainer**: JumpReLU activation function
- **GatedAnnealTrainer**: Gated SAE with annealing

To use a different trainer, modify `train_sae.py` or create a custom training script.

## Output Structure

After training, you'll find:

```
sae_checkpoints/
└── my_experiment/
    ├── args.json           # Training arguments
    ├── trainer_0/
    │   ├── config.json     # SAE config
    │   ├── ae.pt           # Final SAE weights
    │   └── checkpoints/
    │       ├── ae_5000.pt
    │       ├── ae_10000.pt
    │       └── ...
```

## Loading Trained SAE

```python
from openpi.interpretability.dictionary_learning.dictionary_learning.dictionary import AutoEncoder
import torch

# Load SAE
state_dict = torch.load("sae_checkpoints/my_experiment/trainer_0/ae.pt")
config = json.load(open("sae_checkpoints/my_experiment/trainer_0/config.json"))

sae = AutoEncoder(
    activation_dim=config["trainer"]["activation_dim"],
    dict_size=config["trainer"]["dict_size"],
)
sae.load_state_dict(state_dict)
sae.eval()

# Use SAE
activations = ...  # [batch, d_model]
features = sae.encode(activations)  # [batch, dict_size]
reconstructed = sae.decode(features)  # [batch, d_model]
```
