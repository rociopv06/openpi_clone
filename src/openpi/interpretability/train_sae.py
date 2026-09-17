"""
SAE (Sparse Autoencoder) training script for PI0 model activations.

This script trains SAEs on activations from PI0 models using the dictionary_learning
framework. It supports:
- Multiple hook points (vision, language model layers, action expert layers)
- Single/multi-GPU training
- Wandb logging
- Checkpoint saving and resuming

Usage:
    # Train SAE on action expert layer 6 with default settings
    python -m openpi.interpretability.train_sae \
        --config pi0_aloha_sim \
        --hook_point expert_layer_6 \
        --exp_name sae_expert_l6

    # Train on language model layer 12 with custom settings
    python -m openpi.interpretability.train_sae \
        --config pi0_aloha_sim \
        --hook_point lang_layer_12 \
        --expansion_factor 8 \
        --k 64 \
        --steps 50000 \
        --exp_name sae_lang_l12

    # Multi-GPU on cluster with SLURM
    # See examples/slurm_sae_train.sh for SLURM script template
"""

import argparse
import json
import logging
import os
from pathlib import Path

import safetensors.torch
import torch

# Import dictionary learning components
from openpi.interpretability.dictionary_learning.dictionary_learning.trainers.batch_top_k import (
    BatchTopKTrainer,
)
from openpi.interpretability.dictionary_learning.dictionary_learning.training import trainSAE

# Import openpi components
import openpi.models.pi0_config as _pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.interpretability.activation_hooks import (
    PI0ActivationBuffer,
    PI0_HOOK_POINTS,
    list_available_hooks,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAE on PI0 activations")

    # Model and data config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config name (e.g., pi0_aloha_sim, pi0_fast_droid)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to converted PyTorch model checkpoint (model.safetensors). Required — never uses random init.",
    )
    parser.add_argument(
        "--data_subset",
        type=str,
        default=None,
        help="Subset of data to use (for testing with smaller datasets)",
    )

    # Hook configuration
    parser.add_argument(
        "--hook_point",
        type=str,
        required=True,
        help=f"Hook point name. Available: {list(PI0_HOOK_POINTS.keys())[:10]}... (use --list_hooks for full list)",
    )
    parser.add_argument(
        "--list_hooks",
        action="store_true",
        help="List all available hook points and exit",
    )

    # SAE architecture
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=4,
        help="SAE expansion factor (dict_size = expansion_factor * d_model)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of active features (for BatchTopK trainer)",
    )

    # Training parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="SAE training batch size (out_batch_size from buffer)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--n_ctxs",
        type=int,
        default=30000,
        help="Number of contexts to store in activation buffer",
    )
    parser.add_argument(
        "--data_batch_size",
        type=int,
        default=32,
        help="Batch size for loading data from PI0 model",
    )

    # Logging and saving
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name for logging and checkpoints",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./sae_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_steps",
        type=str,
        default="5000,10000,20000,30000",
        help="Comma-separated list of steps at which to save checkpoints",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
        help="Log every N steps",
    )

    # Wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="pi0-sae",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Wandb entity (username or team)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for training",
    )

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_str]


def load_pi0_model(config: _config.TrainConfig, checkpoint_path: str | None, device: str):
    """Load PI0 model from checkpoint."""
    from openpi.models_pytorch import pi0_pytorch

    logger.info(f"Loading PI0 model with config: {config.model}")

    model = pi0_pytorch.PI0Pytorch(config=config.model)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    safetensors.torch.load_model(model, checkpoint_path)

    model = model.to(device)
    model.eval()

    # Disable torch.compile for hook compatibility
    model.sample_actions = model.sample_actions.__wrapped__

    return model


def create_data_loader(config: _config.TrainConfig, subset: str | None = None):         
      """Create data loader for activation collection."""                                 
      # Optionally filter to subset                                                       
      if subset:                                                                          
          logger.info(f"Filtering data to subset: {subset}")                              
                                                                                          
      # Create the data loader using openpi's data loading utilities                      
      train_loader = _data.create_data_loader(                                            
          config,                                                                         
          shuffle=True,                                                                   
          framework="pytorch",                                                            
      )                                                                                   
                                                                                          
      return train_loader


def main():
    args = parse_args()

    if args.list_hooks:
        list_available_hooks()
        return

    # Validate hook point
    if args.hook_point not in PI0_HOOK_POINTS:
        logger.error(f"Unknown hook point: {args.hook_point}")
        list_available_hooks()
        return

    logger.info(f"Training SAE for hook point: {args.hook_point}")
    logger.info(f"Experiment: {args.exp_name}")

    # Parse save steps
    save_steps = [int(s) for s in args.save_steps.split(",")]

    # Create save directory
    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load training config
    logger.info(f"Loading config: {args.config}")
    train_config = _config.get_config(args.config)

    # Load model
    model = load_pi0_model(train_config, args.checkpoint_path, args.device)

    # Create data loader
    logger.info("Creating data loader...")
    data_loader = create_data_loader(
        train_config,
        subset=args.data_subset
    )

    # Create activation buffer
    logger.info(f"Creating activation buffer for {args.hook_point}...")
    dtype = get_dtype(args.dtype)

    buffer = PI0ActivationBuffer(
        data_loader=data_loader,
        model=model,
        hook_point=args.hook_point,
        n_ctxs=args.n_ctxs,
        out_batch_size=args.batch_size,
        device=args.device,
        dtype=dtype,
    )

    d_model = buffer.d_submodule
    dict_size = args.expansion_factor * d_model

    logger.info(f"Activation dimension: {d_model}")
    logger.info(f"Dictionary size: {dict_size} ({args.expansion_factor}x expansion)")

    # Create SAE trainer config
    trainer_config = {                                                                                                                                                                                             
         "trainer": BatchTopKTrainer,                                                                                                                                                                               
         "activation_dim": d_model,                                                                                                                                                                                 
         "dict_size": dict_size,                                                                                                                                                                                    
         "k": args.k,                                                                                                                                                                                               
         "lr": args.lr,                                                                                                                                                                                             
         "device": args.device,                                                                                                                                                                                     
         "wandb_name": f"{args.exp_name}_{args.hook_point}",                                                                                                                                                        
         "steps": args.steps,                                                                                                                                                                                       
         "layer": args.hook_point,                                                                                                                                                                                  
         "lm_name": args.config,                                                                                                                                                                                    
     }

    run_cfg = {
        "hook_point": args.hook_point,
        "expansion_factor": args.expansion_factor,
        "config": args.config,
        **buffer.config,
    }

    # Train SAE
    logger.info("Starting SAE training...")
    logger.info(f"  Steps: {args.steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  K (sparsity): {args.k}")
    logger.info(f"  Learning rate: {args.lr}")

    trainSAE(
        data=buffer,
        trainer_configs=[trainer_config],
        steps=args.steps,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        save_steps=save_steps,
        save_dir=str(save_dir),
        log_steps=args.log_steps,
        run_cfg=run_cfg,
        device=args.device,
        autocast_dtype=dtype,
    )

    # Cleanup
    buffer.close()
    logger.info(f"Training complete! Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
