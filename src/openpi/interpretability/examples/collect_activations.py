"""
Example: Collecting activations from PI0 model for SAE training.

This script demonstrates how to:
1. Load a PI0 model
2. Register activation hooks
3. Collect activations during inference
4. Save activations to disk for later SAE training

Usage:
    python collect_activations.py \
        --config pi0_aloha_sim \
        --checkpoint_path /path/to/checkpoint.safetensors \
        --hook_points expert_layer_6 lang_layer_12 \
        --output_dir ./activations \
        --num_batches 100
"""

import argparse
import logging
from pathlib import Path

import safetensors.torch
import torch
from tqdm import tqdm

import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.models_pytorch import pi0_pytorch
from openpi.interpretability.activation_hooks import (
    PI0ActivationCollector,
    list_available_hooks,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--hook_points", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./activations")
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--list_hooks", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_hooks:
        list_available_hooks()
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    logger.info(f"Loading config: {args.config}")
    train_config = _config.get_config(args.config)

    logger.info(f"Loading model from: {args.checkpoint_path}")
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    safetensors.torch.load_model(model, args.checkpoint_path)
    model = model.to(args.device)
    model.eval()

    # Create collector
    collector = PI0ActivationCollector(model)
    collector.register_hooks(args.hook_points)
    logger.info(f"Registered hooks: {args.hook_points}")

    # Create data loader
    logger.info("Creating data loader...")
    data_loader = _data.create_data_loader(
        train_config.data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Collect activations
    all_activations = {hp: [] for hp in args.hook_points}

    logger.info(f"Collecting activations from {args.num_batches} batches...")
    data_iter = iter(data_loader)

    for batch_idx in tqdm(range(args.num_batches)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        observation = batch["observation"]
        actions = batch["actions"]

        # Move to device
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(args.device)
            elif isinstance(x, dict):
                return {k: to_device(v) for k, v in x.items()}
            return x

        observation = to_device(observation)
        actions = to_device(actions)

        # Forward pass
        with torch.no_grad():
            _ = model(observation, actions)

        # Collect activations
        acts = collector.get_activations(flatten=True)
        for hp in args.hook_points:
            all_activations[hp].append(acts[hp].cpu())

        collector.clear_activations()

    # Concatenate and save
    for hp in args.hook_points:
        acts = torch.cat(all_activations[hp], dim=0)
        output_path = output_dir / f"{hp}_activations.pt"
        torch.save(acts, output_path)
        logger.info(f"Saved {hp}: {acts.shape} -> {output_path}")

    collector.clear_hooks()
    logger.info("Done!")


if __name__ == "__main__":
    main()
