"""PoC model runner - simplified forward pass for vLLM v0.16.

This mimics vLLM's /chat/completion TP synchronization:
- TP rank0 (driver) broadcasts metadata to all TP workers
- Non-driver TP workers block until they receive the broadcast
- All TP ranks then enter model forward together (NCCL collectives align)

Key changes from v0.9.1:
- Removed _create_prefill_attn_metadata (v0.16 has a completely different
  attention system based on CommonAttentionMetadata + per-layer builders)
- Uses set_forward_context with attn_metadata=None (PoC forward doesn't
  need KV cache or proper attention scheduling)
- Updated worker attribute paths for v0.16 (vllm_config.model_config.dtype)
"""
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors

from .gpu_random import (
    generate_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

# Default k_dim (can be overridden per-request)
DEFAULT_K_DIM = 12


def _ensure_layer_hooks(worker, block_hash: str, hidden_size: int) -> None:
    """Ensure layer hooks are installed on the worker for the given block_hash.

    Caches hooks on worker._poc_layer_hooks. If block_hash changes, detaches
    old hooks and installs new ones (per-round transform changes).
    """
    model = worker.model_runner.model
    device = worker.device

    existing_hook = getattr(worker, '_poc_layer_hooks', None)

    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()

    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass on a worker.

    Mimics /chat/completion TP synchronization:
    - TP rank0 broadcasts PoC metadata
    - Non-driver ranks block until broadcast received
    - All ranks enter forward together (NCCL ops align)

    Returns:
        Dict with nonces and vectors (FP16 numpy arrays for encoding).
        Returns None for non-last PP ranks.
    """
    device = worker.device
    dtype = worker.vllm_config.model_config.dtype
    model = worker.model_runner.model
    worker_vllm_config = worker.vllm_config

    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0

    # =========================================================================
    # TP SYNC: Rendezvous + CPU-only gate (no NCCL)
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)

        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])

    batch_size = len(nonces)

    # Generate embeddings on first PP rank, receive intermediate tensors on others
    intermediate_tensors = None
    inputs_embeds = None

    pp_group = get_pp_group()

    if pp_group.is_first_rank:
        inputs_embeds = generate_inputs(
            block_hash, public_key, nonces,
            dim=hidden_size, seq_len=seq_len,
            device=device, dtype=dtype,
        )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )

    # Create positions tensor
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # =========================================================================
    # TP SYNC: Pre-forward rendezvous
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)

    torch.cuda.synchronize()

    # Ensure layer hooks are installed for this block_hash (lazy + cached)
    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Forward pass with PoC context (activates layer hook transformations)
    # In v0.16, set_forward_context requires attn_metadata and vllm_config.
    # We pass attn_metadata=None (profile-run mode): attention layers return
    # zeros, so hidden states are driven by embeddings + MLP + layer norms +
    # Householder layer hooks. This is deterministic and sufficient for PoC
    # artifact generation. skip_compiled=True bypasses torch.compile graphs.
    with set_forward_context(None, worker_vllm_config, skip_compiled=True):
        with poc_forward_context():
            hidden_states = model(
                input_ids=None,
                positions=positions.flatten(),
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
            )

    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None

    # Extract last token hidden state and compute in FP32
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()

    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

    # Per-nonce k-dim pick + Haar rotation (via Householder chain, no cuSOLVER)
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16 for artifact encoding (compute was in FP32)
    vectors_f16 = yk.half().cpu().numpy()

    return {
        "nonces": nonces,
        "vectors": vectors_f16,  # FP16 numpy array, shape [batch_size, k_dim]
    }
