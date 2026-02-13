"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model's learned output structure.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List

import torch

from .gpu_random import generate_householder_vector, apply_householder

# Context variable for conditional hook activation
# Default False means hooks pass through unchanged (for inference)
_poc_forward_active: ContextVar[bool] = ContextVar('poc_forward_active', default=False)


@contextmanager
def poc_forward_context():
    """Context manager for PoC forward passes.
    
    Hooks only transform hidden states when this context is active.
    This allows inference and PoC to coexist without interference.
    
    Usage:
        with poc_forward_context():
            hidden_states = model(...)  # Hooks will transform
    """
    token = _poc_forward_active.set(True)
    try:
        yield
    finally:
        _poc_forward_active.reset(token)


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active."""
    return _poc_forward_active.get()


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers.
    
    These hooks apply the same transform to all nonces in a round (determined
    by block_hash). Combined with per-nonce hidden state transforms, this
    provides strong structure breaking.
    
    Usage:
        # At round init
        hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)
        
        # Run forward passes...
        
        # At round end
        hooks.detach()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.reflection_vectors: List[torch.Tensor] = []
        self.block_hash = block_hash
        # self._setup(model, block_hash, device, hidden_size)
    
    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        # Try common patterns for different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama, Qwen, Mistral style
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            return list(model.transformer.h)
        elif hasattr(model, 'layers'):
            # Direct layers attribute
            return list(model.layers)
        return []
    
    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Setup hooks on all transformer layers."""
        layers = self._find_layers(model)
        self.num_total_layers = len(layers)
        
        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            self.reflection_vectors.append(v)
            
            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)
    
    def _create_hook(self, layer_idx: int):
        """Create a forward hook that applies Householder reflection.
        
        Hook only transforms when poc_forward_context is active.
        This allows inference to proceed unaffected when PoC hooks are registered.
        
        vLLM decoder layers typically return (hidden_states, residual).
        We must transform BOTH to prevent residual connections from
        preserving untransformed values.
        """
        def hook(module, input, output):
            # Early exit if not in PoC forward context - pass through unchanged
            if not is_poc_forward_active():
                return output
            
            v = self.reflection_vectors[layer_idx]
            
            def transform(x):
                # Apply Householder reflection (preserves magnitude)
                return apply_householder(x, v.to(x.dtype))
            
            if isinstance(output, tuple):
                if len(output) >= 2:
                    # (hidden_states, residual, ...) format - transform both
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    transformed_hidden = transform(hidden)
                    transformed_residual = transform(residual)
                    return (transformed_hidden, transformed_residual) + rest
                else:
                    # Single element tuple
                    hidden = output[0]
                    transformed = transform(hidden)
                    return (transformed,)
            else:
                transformed = transform(output)
                return transformed
        
        return hook
    
    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.reflection_vectors = []
    
    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)
