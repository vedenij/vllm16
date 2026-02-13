"""PoC model runner - simplified forward pass for vLLM v0.16.

Uses direct_qkv=True in FlashAttentionMetadata so that attention calls
flash_attn_varlen_func with raw Q/K/V tensors (no KV cache), matching
v0.9.1 prefill path for bit-exact reproducibility.

collective_rpc passes identical arguments to all TP workers, so no
barriers or broadcast needed.

- attn_metadata is dict[str, AttentionMetadata] (per-layer)
- slot_mapping_dict is empty (no KV cache writes)
"""
import time
import torch
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.logger import init_logger

from .gpu_random import (
    generate_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

logger = init_logger(__name__)

# Default k_dim (can be overridden per-request)
DEFAULT_K_DIM = 12


def _create_poc_attn_context(worker, batch_size, seq_len, device):
    """Create attention metadata for PoC direct Q/K/V forward.

    Uses direct_qkv=True so FlashAttention calls flash_attn_varlen_func
    with raw Q/K/V tensors (no KV cache), matching v0.9.1 prefill path
    for bit-exact reproducibility.

    Returns:
        (attn_metadata_dict, slot_mapping_dict) — attn_metadata_dict is
        dict[str, FlashAttentionMetadata], slot_mapping_dict is empty
        (no KV cache writes needed).
    """
    vllm_config = worker.vllm_config
    forward_ctx = vllm_config.compilation_config.static_forward_context
    attn_layers = {
        name: layer for name, layer in forward_ctx.items()
        if isinstance(layer, Attention)
    }

    logger.info(f"[PoC] _create_poc_attn_context: "
                f"{len(forward_ctx)} total layers, "
                f"{len(attn_layers)} attention layers")

    if not attn_layers:
        raise RuntimeError(
            f"No Attention layers found in static_forward_context. "
            f"Layer types: {[type(v).__name__ for v in forward_ctx.values()][:10]}"
        )

    num_tokens = batch_size * seq_len

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
        block_table=torch.empty(0, dtype=torch.int32, device=device),
        slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
        direct_qkv=True,
    )

    attn_metadata_dict = {name: attn_metadata for name in attn_layers}
    slot_mapping_dict = {}  # Empty — skip KV cache writes

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
            logger.info(f"[PoC] layer hooks cached for block_hash={block_hash[:16]}...")
            return
        logger.info(f"[PoC] detaching old hooks, installing new ones")
        existing_hook.detach()

    logger.info(f"[PoC] installing layer hooks for block_hash={block_hash[:16]}... "
                f"hidden_size={hidden_size}")
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook
    logger.info(f"[PoC] installed {hook.num_layers} layer hooks")


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

    Called via collective_rpc which passes identical args to all TP workers.
    Uses direct_qkv=True for bit-exact match with v0.9.1.

    Returns:
        Dict with nonces and vectors (FP16 numpy arrays for encoding).
        Returns None for non-last PP ranks.
    """
    t0 = time.time()
    device = worker.device
    dtype = worker.vllm_config.model_config.dtype
    model = worker.model_runner.model
    worker_vllm_config = worker.vllm_config

    tp_group = get_tp_group()
    rank = tp_group.rank_in_group

    logger.info(f"[PoC][rank={rank}] execute_poc_forward START "
                f"batch={len(nonces)} seq_len={seq_len} hidden={hidden_size}")

    try:
        # collective_rpc passes identical arguments to all TP workers,
        # so no barriers or broadcast needed.
        batch_size = len(nonces)

        # Generate embeddings on first PP rank
        intermediate_tensors = None
        inputs_embeds = None

        pp_group = get_pp_group()

        if pp_group.is_first_rank:
            logger.info(f"[PoC][rank={rank}] generating inputs...")
            inputs_embeds = generate_inputs(
                block_hash, public_key, nonces,
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
            logger.info(f"[PoC][rank={rank}] inputs generated "
                        f"shape={inputs_embeds.shape} (+{time.time()-t0:.2f}s)")
        else:
            intermediate_tensors = IntermediateTensors(
                pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
            )

        # Create positions tensor
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        torch.cuda.synchronize()

        # Ensure layer hooks are installed for this block_hash (lazy + cached)
        logger.info(f"[PoC][rank={rank}] ensuring layer hooks...")
        _ensure_layer_hooks(worker, block_hash, hidden_size)
        logger.info(f"[PoC][rank={rank}] layer hooks done (+{time.time()-t0:.2f}s)")

        # Create real attention metadata
        logger.info(f"[PoC][rank={rank}] creating attn context...")
        attn_metadata_dict, slot_mapping_dict = _create_poc_attn_context(
            worker, batch_size, seq_len, device
        )
        logger.info(f"[PoC][rank={rank}] attn context done: "
                    f"{len(attn_metadata_dict)} layers (+{time.time()-t0:.2f}s)")

        # Forward pass
        logger.info(f"[PoC][rank={rank}] starting model forward...")
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

        torch.cuda.synchronize()
        logger.info(f"[PoC][rank={rank}] model forward done (+{time.time()-t0:.2f}s)")

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

        # Per-nonce k-dim pick + Haar rotation
        logger.info(f"[PoC][rank={rank}] post-processing...")
        indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
        xk = torch.gather(last_hidden, 1, indices)
        yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

        # Normalize output vectors
        yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

        # Convert to FP16 for artifact encoding
        vectors_f16 = yk.half().cpu().numpy()

        logger.info(f"[PoC][rank={rank}] execute_poc_forward DONE "
                    f"total={time.time()-t0:.2f}s")

        return {
            "nonces": nonces,
            "vectors": vectors_f16,
        }

    except Exception as e:
        logger.error(f"[PoC][rank={rank}] execute_poc_forward FAILED "
                     f"after {time.time()-t0:.2f}s: {e}", exc_info=True)
        raise
