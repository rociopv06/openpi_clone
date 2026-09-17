"""
OpenPI Interpretability Module

This module provides tools for interpreting PI0/PI05 vision-language-action models
using Sparse Autoencoders (SAEs) and dictionary learning.

Main components:
- activation_hooks: Hook utilities for collecting model activations
- train_sae: SAE training script
- dictionary_learning: SAE training framework (submodule)

Quick start:
    from openpi.interpretability.activation_hooks import (
        PI0ActivationCollector,
        PI0ActivationBuffer,
        list_available_hooks,
    )

    # List available hook points
    list_available_hooks()

    # Collect activations
    collector = PI0ActivationCollector(model)
    collector.register_hooks(["expert_layer_6", "lang_layer_12"])
"""

from openpi.interpretability.activation_hooks import (
    PI0ActivationCollector,
    PI0ActivationBuffer,
    PI0_HOOK_POINTS,
    HookPoint,
    list_available_hooks,
)

__all__ = [
    "PI0ActivationCollector",
    "PI0ActivationBuffer",
    "PI0_HOOK_POINTS",
    "HookPoint",
    "list_available_hooks",
]
