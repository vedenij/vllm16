"""PoC model runner - simplified forward pass for vLLM v0.16.

This mimics vLLM's /chat/completion TP synchronization:
- TP rank0 (driver) broadcasts metadata to all TP workers
- Non-driver TP workers block until they receive the broadcast
- All TP ranks then enter model forward together (NCCL collectives align)

Key changes from v0.9.1:
- v0.16 separates KV cache update from attention forward. Decoder attention
  always reads K/V from KV cache (no direct Q/K/V path).
- We create real FlashAttentionMetadata with real slot mappings and block
  tables so K/V gets written to cache and read back during attention.
- attn_metadata is dict[str, AttentionMetadata] (per-layer)
- slot_mapping is dict[str, torch.Tensor] (per-layer)
"""
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from .gpu_random import (
    generate_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

# Default k_dim (can be overridden per-request)
DEFAULT_K_DIM = 12


def _create_poc_attn_context(worker, batch_size, seq_len, device):
    """Create real attention metadata + slot mappings for PoC forward.

    In v0.16, decoder attention always reads K/V from the KV cache via
    block_table. We borrow the first N blocks of the already-allocated
    cache to store K/V during the forward pass.

    Returns:
        (attn_metadata_dict, slot_mapping_dict) — both are
        dict[str, ...] keyed by attention layer name.
    """
    vllm_config = worker.vllm_config
    # static_forward_context maps layer_name -> various modules (Attention,
    # MoE, Mamba, etc). Filter to only Attention layers.
    forward_ctx = vllm_config.compilation_config.static_forward_context
    attn_layers = {
        name: layer for name, layer in forward_ctx.items()
        if isinstance(layer, Attention)
    }

    # Get block_size from the first attention layer's KV cache
    first_layer_name = next(iter(attn_layers))
    first_layer = attn_layers[first_layer_name]
    kv_cache = first_layer.kv_cache[0]  # virtual_engine=0
    # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
    block_size = kv_cache.shape[2]

    num_tokens = batch_size * seq_len
    blocks_per_req = (seq_len + block_size - 1) // block_size
    num_blocks_needed = batch_size * blocks_per_req

    assert num_blocks_needed <= kv_cache.shape[1], (
        f"PoC needs {num_blocks_needed} blocks but only "
        f"{kv_cache.shape[1]} available in KV cache"
    )

    # slot_mapping: linear mapping [0, num_tokens) -> slots in blocks
    # slot i maps to block (i // block_size), offset (i % block_size)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # block_table: each request gets its own contiguous blocks
    block_table = torch.zeros(
        batch_size, blocks_per_req, dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        start_block = i * blocks_per_req
        block_table[i] = torch.arange(
            start_block, start_block + blocks_per_req,
            dtype=torch.int32, device=device
        )

    # query_start_loc: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
    query_start_loc = torch.arange(
        0, num_tokens + 1, seq_len, dtype=torch.int32, device=device
    )
    seq_lens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )

    attn_metadata = FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )

    # Build per-layer dicts (all attention layers share the same metadata)
    attn_metadata_dict = {}
    slot_mapping_dict = {}
    for layer_name in attn_layers:
        attn_metadata_dict[layer_name] = attn_metadata
        slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


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

    # Create real attention metadata so attention computes properly.
    # In v0.16, decoder attention reads K/V from KV cache via block_table.
    # We borrow blocks from the already-allocated cache, provide valid
    # slot mappings so K/V gets written, and valid block_table so
    # attention reads K/V back. This matches v0.9.1 behavior where
    # flash_attn_varlen_func computed real attention with Q/K/V.
    attn_metadata_dict, slot_mapping_dict = _create_poc_attn_context(
        worker, batch_size, seq_len, device
    )

    # Forward pass with PoC context (activates layer hook transformations)
    # skip_compiled=True bypasses torch.compile graphs.
    with set_forward_context(
        attn_metadata_dict, worker_vllm_config,
        slot_mapping=slot_mapping_dict, skip_compiled=True,
    ):
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
