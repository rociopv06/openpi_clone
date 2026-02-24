import gc
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def to_device(x, device, dtype=None):
    """Recursively move data to device while preserving semantic dtypes."""
    if isinstance(x, torch.Tensor):
        # Only cast floating tensors
        if dtype is not None and torch.is_floating_point(x):
            return x.to(device=device, dtype=dtype)
        return x.to(device=device)

    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if dtype is not None and torch.is_floating_point(t):
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)

    elif isinstance(x, dict):
        return {k: to_device(v, device, dtype) for k, v in x.items()}

    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device, dtype) for v in x)

    elif hasattr(x, "images") and hasattr(x, "state"):
        # Observation flax struct — be VERY careful with masks
        return x.replace(
            images=to_device(x.images, device, dtype),
            image_masks=to_device(x.image_masks, device, dtype=None),  # MUST stay bool
            state=to_device(x.state, device, dtype),
            tokenized_prompt=to_device(x.tokenized_prompt, device, dtype)
            if x.tokenized_prompt is not None else None,
            tokenized_prompt_mask=to_device(
                x.tokenized_prompt_mask, device, dtype=None
            ) if x.tokenized_prompt_mask is not None else None,
            token_ar_mask=to_device(x.token_ar_mask, device, dtype=None)
            if x.token_ar_mask is not None else None,
            token_loss_mask=to_device(x.token_loss_mask, device, dtype=None)
            if x.token_loss_mask is not None else None,
        )

    return x



@dataclass
class HookPoint:
    name: str
    module_path: str
    io: Literal["in", "out"] = "out"
    description: str = ""


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    parts = path.split(".")
    module = model
    for part in parts:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


# ===========================
# Hook point definitions
# ===========================

PI0_HOOK_POINTS = {
    "vision_out": HookPoint(
        "vision_out",
        "paligemma_with_expert.paligemma.vision_tower",
        "out",
        "Output of SigLIP vision encoder",
    ),
    "vision_proj": HookPoint(
        "vision_proj",
        "paligemma_with_expert.paligemma.multi_modal_projector",
        "out",
        "Multi-modal projector output",
    ),
    **{
        f"lang_layer_{i}": HookPoint(
            f"lang_layer_{i}",
            f"paligemma_with_expert.paligemma.language_model.layers.{i}",
            "out",
            f"Language model layer {i} output",
        )
        for i in range(18)  # gemma_300m has 18 layers (0–17)
    },
    **{
        f"lang_mlp_{i}": HookPoint(
            f"lang_mlp_{i}",
            f"paligemma_with_expert.paligemma.language_model.layers.{i}.mlp",
            "out",
            f"Language model MLP {i} output",
        )
        for i in range(18)  # gemma_300m has 18 layers (0–17)
    },
    **{
        f"expert_layer_{i}": HookPoint(
            f"expert_layer_{i}",
            f"paligemma_with_expert.gemma_expert.model.layers.{i}",
            "out",
            f"Action expert layer {i} output",
        )
        for i in range(18)  # gemma_300m has 18 layers (0–17)
    },
    **{
        f"expert_mlp_{i}": HookPoint(
            f"expert_mlp_{i}",
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp",
            "out",
            f"Action expert MLP {i} output",
        )
        for i in range(18)  # gemma_300m has 18 layers (0–17)
    },
    "action_in_proj": HookPoint(
        "action_in_proj", "action_in_proj", "out", "Action input projection"
    ),
    "action_out_proj": HookPoint(
        "action_out_proj", "action_out_proj", "out", "Action output projection"
    ),
    "state_proj": HookPoint(
        "state_proj", "state_proj", "out", "State projection (pi0 only)"
    ),
}


# ===========================
# Activation collector
# ===========================

class PI0ActivationCollector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.hook_points = {}
        self._cached_hook_names = set()  # Hook points that use caching instead of forward hooks

    def _make_hook(self, name: str, io: Literal["in", "out"]):
        def hook(module, inputs, outputs):
            act = inputs[0] if io == "in" else (outputs[0] if isinstance(outputs, tuple) else outputs)
            if isinstance(act, torch.Tensor):
                self.activations[name] = act.detach().clone()
        return hook

    def register_hooks(self, hook_points=None, custom_hooks=None):
        self.clear_hooks()
        hook_points = hook_points or []

        # Separate cached vs forward-hook based hook points
        expert_layers = []
        lang_layers = []
        forward_hook_points = []

        for name in hook_points:
            hp = PI0_HOOK_POINTS[name]
            self.hook_points[name] = hp

            # Check if this is an expert layer or lang layer (needs caching)
            if name.startswith("expert_layer_"):
                layer_idx = int(name.split("_")[-1])
                expert_layers.append(layer_idx)
                self._cached_hook_names.add(name)
            elif name.startswith("lang_layer_"):
                layer_idx = int(name.split("_")[-1])
                lang_layers.append(layer_idx)
                self._cached_hook_names.add(name)
            else:
                forward_hook_points.append((name, hp))

        # Enable caching on the model for expert/lang layers
        paligemma_model = self._get_paligemma_model()
        if paligemma_model is not None and (expert_layers or lang_layers):
            paligemma_model.enable_activation_cache(
                expert_layers=expert_layers,
                lang_layers=lang_layers
            )

        # Register forward hooks for non-cached hook points
        for name, hp in forward_hook_points:
            try:
                module = get_module_by_path(self.model, hp.module_path)
                handle = module.register_forward_hook(self._make_hook(hp.name, hp.io))
                self.hooks.append(handle)
            except (AttributeError, KeyError) as e:
                print(f"Warning: Could not register hook for {name}: {e}")

        if custom_hooks:
            for hp in custom_hooks:
                module = get_module_by_path(self.model, hp.module_path)
                handle = module.register_forward_hook(self._make_hook(hp.name, hp.io))
                self.hooks.append(handle)
                self.hook_points[hp.name] = hp

    def _get_paligemma_model(self):
        """Get the PaliGemmaWithExpertModel from nested model structure."""
        model = self.model
        # Handle wrapped models (e.g., Pi0Policy wraps PaliGemmaWithExpertModel)
        if hasattr(model, "paligemma_with_expert"):
            return model.paligemma_with_expert
        elif hasattr(model, "model") and hasattr(model.model, "paligemma_with_expert"):
            return model.model.paligemma_with_expert
        elif hasattr(model, "_cached_activations"):
            # Model itself is PaliGemmaWithExpertModel
            return model
        return None

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activations.clear()
        self.hook_points.clear()
        self._cached_hook_names.clear()

        # Disable caching on the model
        paligemma_model = self._get_paligemma_model()
        if paligemma_model is not None and hasattr(paligemma_model, "disable_activation_cache"):
            paligemma_model.disable_activation_cache()

    def clear_activations(self):
        self.activations.clear()
        # Also clear model's cached activations
        paligemma_model = self._get_paligemma_model()
        if paligemma_model is not None and hasattr(paligemma_model, "clear_cached_activations"):
            paligemma_model.clear_cached_activations()

    def get_activations(self, flatten=False):
        # Merge forward hook activations with cached activations from model
        all_activations = self.activations.copy()

        # Get cached activations from model (for expert/lang layers)
        paligemma_model = self._get_paligemma_model()
        if paligemma_model is not None and hasattr(paligemma_model, "get_cached_activations"):
            cached = paligemma_model.get_cached_activations()
            for name in self._cached_hook_names:
                if name in cached:
                    all_activations[name] = cached[name]

        if not flatten:
            return all_activations
        out = {}
        for k, v in all_activations.items():
            out[k] = v.reshape(-1, v.shape[-1]) if v.ndim >= 2 else v
        return out

    def __del__(self):
        self.clear_hooks()


# ===========================
# Activation buffer (SAE)
# ===========================

class PI0ActivationBuffer:
    def __init__(
        self,
        data_loader,
        model: nn.Module,
        hook_point: str,
        d_submodule=None,
        n_ctxs=30000,
        out_batch_size=8192,
        device="cuda",
        dtype=torch.float32,
    ):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.model = model
        self.hook_point = hook_point
        self.device = device
        self.dtype = dtype
        self.out_batch_size = out_batch_size
        self.n_ctxs = n_ctxs

        self.collector = PI0ActivationCollector(model)
        self.collector.register_hooks([hook_point])

        if d_submodule is None:
            d_submodule = self._infer_dimension()
        self.d_submodule = d_submodule

        self.activation_buffer_size = int(n_ctxs * 50)
        self.activations = torch.empty(0, d_submodule, device=device, dtype=dtype)
        self.read = torch.zeros(0, dtype=torch.bool, device=device)

    def _infer_dimension(self):
        self.model.eval()
        try:
            batch = next(iter(self.data_loader))
            observation, actions = batch if isinstance(batch, tuple) else (
                batch["observation"], batch["actions"]
            )

            param = self.model.action_in_proj.weight
            device, dtype = param.device, param.dtype

            observation = to_device(observation, device, dtype)
            actions = to_device(actions, device, dtype)

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

    def __next__(self):
        with torch.no_grad():
            unread = (~self.read).sum().item()
            if unread < self.activation_buffer_size // 2:
                self.refresh()

            unreads = (~self.read).nonzero().squeeze(-1)
            if unreads.numel() == 0:
                raise StopIteration

            idxs = unreads[torch.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def refresh(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.activations = self.activations[~self.read]
        current = len(self.activations)

        new_buf = torch.empty(
            self.activation_buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.dtype,
        )
        new_buf[:current] = self.activations
        self.activations = new_buf

        param = self.model.action_in_proj.weight
        device, dtype = param.device, param.dtype

        pbar = tqdm(total=self.activation_buffer_size, initial=current,
                    desc=f"Collecting {self.hook_point} activations")

        while current < self.activation_buffer_size:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.data_loader)
                batch = next(self.data_iter)

            observation, actions = batch if isinstance(batch, tuple) else (
                batch["observation"], batch["actions"]
            )

            observation = to_device(observation, device, dtype)
            actions = to_device(actions, device, dtype)

            with torch.no_grad():
                _ = self.model(observation, actions)

            acts = self.collector.get_activations(flatten=True)[self.hook_point]
            acts = acts.to(self.device, self.dtype)

            n = min(len(acts), self.activation_buffer_size - current)
            self.activations[current:current + n] = acts[:n]
            current += n
            pbar.update(n)

            self.collector.clear_activations()

        pbar.close()
        self.read = torch.zeros(len(self.activations), dtype=torch.bool, device=self.device)

    def close(self):
        self.collector.clear_hooks()
    @property
    def config(self):
        return {
            "hook_point": self.hook_point,
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "out_batch_size": self.out_batch_size,
            "device": self.device,                                                                                                                                                                                     
        } 

def list_available_hooks():
    print("Available PI0 Hook Points")
    print("-" * 60)
    for name, hp in sorted(PI0_HOOK_POINTS.items()):
        print(f"{name:25s} - {hp.description}")
