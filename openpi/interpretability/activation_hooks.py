"""
Activation hooks for PI0/PI05 models to support SAE (Sparse Autoencoder) training.

This module provides utilities to:
1. Register forward hooks on specific layers of the PI0 model
2. Collect activations during forward passes
3. Create activation buffers compatible with the dictionary_learning framework

Usage:
    from openpi.interpretability.activation_hooks import PI0ActivationCollector

    collector = PI0ActivationCollector(model)
    collector.register_hooks(hook_points=["vision_out", "lang_layer_12", "expert_layer_6"])

    # Run forward pass
    with torch.no_grad():
        _ = model(observation, actions)

    # Get collected activations
    activations = collector.get_activations()
"""

import gc
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
from torch import nn
from tqdm import tqdm


@dataclass
class HookPoint:
    """Describes a hook point in the model."""
    name: str
    module_path: str  # dot-separated path to the module
    io: Literal["in", "out"] = "out"
    description: str = ""


# Pre-defined hook points for PI0 model
PI0_HOOK_POINTS = {
    # Vision encoder
    "vision_out": HookPoint(
        name="vision_out",
        module_path="paligemma_with_expert.paligemma.vision_tower",
        io="out",
        description="Output of SigLIP vision encoder"
    ),
    "vision_proj": HookPoint(
        name="vision_proj",
        module_path="paligemma_with_expert.paligemma.multi_modal_projector",
        io="out",
        description="Multi-modal projector output"
    ),

    # Language model layers (PaliGemma uses Gemma with 18 layers by default)
    **{
        f"lang_layer_{i}": HookPoint(
            name=f"lang_layer_{i}",
            module_path=f"paligemma_with_expert.paligemma.language_model.layers.{i}",
            io="out",
            description=f"Language model transformer layer {i} output"
        )
        for i in range(27)  # Support up to 27 layers
    },

    # Language model MLP activations
    **{
        f"lang_mlp_{i}": HookPoint(
            name=f"lang_mlp_{i}",
            module_path=f"paligemma_with_expert.paligemma.language_model.layers.{i}.mlp",
            io="out",
            description=f"Language model layer {i} MLP output"
        )
        for i in range(27)
    },

    # Action expert layers
    **{
        f"expert_layer_{i}": HookPoint(
            name=f"expert_layer_{i}",
            module_path=f"paligemma_with_expert.gemma_expert.model.layers.{i}",
            io="out",
            description=f"Action expert transformer layer {i} output"
        )
        for i in range(27)
    },

    # Action expert MLP activations
    **{
        f"expert_mlp_{i}": HookPoint(
            name=f"expert_mlp_{i}",
            module_path=f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp",
            io="out",
            description=f"Action expert layer {i} MLP output"
        )
        for i in range(27)
    },

    # Action projections
    "action_in_proj": HookPoint(
        name="action_in_proj",
        module_path="action_in_proj",
        io="out",
        description="Action input projection"
    ),
    "action_out_proj": HookPoint(
        name="action_out_proj",
        module_path="action_out_proj",
        io="out",
        description="Action output projection"
    ),

    # State projection (for pi0, not pi05)
    "state_proj": HookPoint(
        name="state_proj",
        module_path="state_proj",
        io="out",
        description="State projection (pi0 only)"
    ),
}


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Get a submodule by dot-separated path."""
    parts = path.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


class PI0ActivationCollector:
    """
    Collects activations from PI0/PI05 model during forward passes.

    Example:
        collector = PI0ActivationCollector(model)
        collector.register_hooks(["lang_layer_12", "expert_layer_6"])

        with torch.no_grad():
            _ = model(observation, actions)

        acts = collector.get_activations()
        # acts = {"lang_layer_12": Tensor[B, S, D], "expert_layer_6": Tensor[B, S, D]}
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.activations: dict[str, torch.Tensor] = {}
        self.hook_points: dict[str, HookPoint] = {}

    def _make_hook(self, name: str, io: Literal["in", "out"]) -> Callable:
        """Create a hook function that stores activations."""
        def hook(module, inputs, outputs):
            if io == "in":
                # inputs is a tuple
                act = inputs[0] if isinstance(inputs, tuple) else inputs
            else:
                # outputs can be a tuple or tensor
                act = outputs[0] if isinstance(outputs, tuple) else outputs

            # Detach and clone to avoid memory issues
            if isinstance(act, torch.Tensor):
                self.activations[name] = act.detach().clone()

        return hook

    def register_hooks(
        self,
        hook_points: list[str] | None = None,
        custom_hooks: list[HookPoint] | None = None
    ) -> None:
        """
        Register forward hooks on specified layers.

        Args:
            hook_points: List of pre-defined hook point names (see PI0_HOOK_POINTS)
            custom_hooks: List of custom HookPoint objects for non-standard hooks
        """
        self.clear_hooks()

        if hook_points is None:
            hook_points = []

        # Register pre-defined hooks
        for name in hook_points:
            if name not in PI0_HOOK_POINTS:
                raise ValueError(f"Unknown hook point: {name}. Available: {list(PI0_HOOK_POINTS.keys())}")

            hp = PI0_HOOK_POINTS[name]
            try:
                module = get_module_by_path(self.model, hp.module_path)
                hook_fn = self._make_hook(hp.name, hp.io)
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                self.hook_points[name] = hp
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not register hook '{name}': {e}")

        # Register custom hooks
        if custom_hooks:
            for hp in custom_hooks:
                try:
                    module = get_module_by_path(self.model, hp.module_path)
                    hook_fn = self._make_hook(hp.name, hp.io)
                    handle = module.register_forward_hook(hook_fn)
                    self.hooks.append(handle)
                    self.hook_points[hp.name] = hp
                except (AttributeError, IndexError) as e:
                    print(f"Warning: Could not register custom hook '{hp.name}': {e}")

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.hook_points = {}
        self.activations = {}

    def clear_activations(self) -> None:
        """Clear stored activations (keeps hooks registered)."""
        self.activations = {}

    def get_activations(self, flatten: bool = False) -> dict[str, torch.Tensor]:
        """
        Get collected activations.

        Args:
            flatten: If True, flatten spatial/sequence dimensions to [N, D]

        Returns:
            Dictionary mapping hook names to activation tensors
        """
        if not flatten:
            return self.activations.copy()

        flattened = {}
        for name, act in self.activations.items():
            if act.ndim >= 2:
                # Flatten all but last dimension: [B, S, D] -> [B*S, D]
                flattened[name] = act.reshape(-1, act.shape[-1])
            else:
                flattened[name] = act
        return flattened

    def __del__(self):
        self.clear_hooks()


class PI0ActivationBuffer:
    """
    Activation buffer for SAE training, compatible with dictionary_learning.

    Yields batches of activations collected from PI0 model during inference
    on a dataset.

    Example:
        buffer = PI0ActivationBuffer(
            data_loader=train_loader,
            model=pi0_model,
            hook_point="expert_layer_6",
            device="cuda",
            out_batch_size=4096,
        )

        for batch in buffer:
            # batch: Tensor[out_batch_size, d_model]
            trainer.update(step, batch)
    """

    def __init__(
        self,
        data_loader,  # DataLoader yielding (observation, actions) batches
        model: nn.Module,
        hook_point: str,
        d_submodule: int | None = None,
        n_ctxs: int = 30000,
        out_batch_size: int = 8192,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            data_loader: PyTorch DataLoader yielding batches
            model: PI0Pytorch model
            hook_point: Name of hook point (from PI0_HOOK_POINTS)
            d_submodule: Activation dimension (inferred if None)
            n_ctxs: Approximate number of contexts to store
            out_batch_size: Batch size for yielded activations
            device: Device to store activations
            dtype: Data type for activations
        """
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.model = model
        self.hook_point = hook_point
        self.device = device
        self.dtype = dtype
        self.out_batch_size = out_batch_size
        self.n_ctxs = n_ctxs

        # Setup collector
        self.collector = PI0ActivationCollector(model)
        self.collector.register_hooks([hook_point])

        # Infer d_submodule if not provided
        if d_submodule is None:
            d_submodule = self._infer_dimension()
        self.d_submodule = d_submodule

        # Initialize buffer
        self.activation_buffer_size = int(n_ctxs * 50)  # Assume ~50 tokens per context
        self.activations = torch.empty(0, d_submodule, device=device, dtype=dtype)
        self.read = torch.zeros(0, dtype=torch.bool, device=device)

    def _infer_dimension(self) -> int:
        """Infer activation dimension by running one forward pass."""
        self.model.eval()
        try:
            batch = next(iter(self.data_loader))
            observation, actions = batch["observation"], batch["actions"]

            # Move to device
            def to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(self.model.action_in_proj.weight.device)
                elif isinstance(x, dict):
                    return {k: to_device(v) for k, v in x.items()}
                return x

            observation = to_device(observation)
            actions = to_device(actions)

            with torch.no_grad():
                _ = self.model(observation, actions)

            acts = self.collector.get_activations(flatten=True)
            d = acts[self.hook_point].shape[-1]
            self.collector.clear_activations()
            return d
        except Exception as e:
            raise RuntimeError(f"Could not infer activation dimension: {e}")

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """Return a batch of activations."""
        with torch.no_grad():
            # Refresh if buffer is less than half full
            unread_count = (~self.read).sum().item()
            if unread_count < self.activation_buffer_size // 2:
                self.refresh()

            # Return a batch
            unreads = (~self.read).nonzero().squeeze(-1)
            if unreads.numel() == 0:
                raise StopIteration("Buffer exhausted")

            perm = torch.randperm(len(unreads), device=unreads.device)
            idxs = unreads[perm[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def refresh(self) -> None:
        """Refresh the activation buffer by running more forward passes."""
        gc.collect()
        torch.cuda.empty_cache()

        # Keep unread activations
        self.activations = self.activations[~self.read]
        current_idx = len(self.activations)

        # Allocate new buffer
        new_activations = torch.empty(
            self.activation_buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.dtype
        )
        new_activations[:current_idx] = self.activations
        self.activations = new_activations

        self.model.eval()
        pbar = tqdm(
            total=self.activation_buffer_size,
            initial=current_idx,
            desc=f"Collecting {self.hook_point} activations"
        )

        while current_idx < self.activation_buffer_size:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                # Reset data iterator
                self.data_iter = iter(self.data_loader)
                batch = next(self.data_iter)

            observation = batch["observation"]
            actions = batch["actions"]

            # Move to model device
            model_device = self.model.action_in_proj.weight.device

            def to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(model_device)
                elif isinstance(x, dict):
                    return {k: to_device(v) for k, v in x.items()}
                return x

            observation = to_device(observation)
            actions = to_device(actions)

            with torch.no_grad():
                _ = self.model(observation, actions)

            # Get flattened activations
            acts = self.collector.get_activations(flatten=True)[self.hook_point]
            acts = acts.to(device=self.device, dtype=self.dtype)

            # Add to buffer
            remaining = self.activation_buffer_size - current_idx
            n_to_add = min(len(acts), remaining)
            self.activations[current_idx:current_idx + n_to_add] = acts[:n_to_add]
            current_idx += n_to_add
            pbar.update(n_to_add)

            self.collector.clear_activations()

        pbar.close()
        self.read = torch.zeros(len(self.activations), dtype=torch.bool, device=self.device)

    @property
    def config(self) -> dict:
        """Return buffer configuration for logging."""
        return {
            "hook_point": self.hook_point,
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
        }

    def close(self) -> None:
        """Clean up resources."""
        self.collector.clear_hooks()


def list_available_hooks() -> None:
    """Print all available hook points."""
    print("Available PI0 Hook Points:")
    print("-" * 60)
    for name, hp in sorted(PI0_HOOK_POINTS.items()):
        print(f"  {name:25s} - {hp.description}")
