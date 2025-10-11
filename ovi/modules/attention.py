# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings
from typing import Optional

try:
    import flash_attn_interface  # type: ignore
    FLASH_ATTN_3_AVAILABLE = True
    _FLASH_ATTN_3_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    flash_attn_interface = None  # type: ignore
    FLASH_ATTN_3_AVAILABLE = False
    _FLASH_ATTN_3_ERROR = exc

try:
    import flash_attn  # type: ignore
    FLASH_ATTN_2_AVAILABLE = True
    _FLASH_ATTN_2_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    flash_attn = None  # type: ignore
    FLASH_ATTN_2_AVAILABLE = False
    _FLASH_ATTN_2_ERROR = exc

try:
    from sageattention import sageattn as _sageattn  # type: ignore
    SAGE_ATTENTION_AVAILABLE = True
    _SAGE_ATTENTION_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    _sageattn = None  # type: ignore
    SAGE_ATTENTION_AVAILABLE = False
    _SAGE_ATTENTION_ERROR = exc

__all__ = [
    'flash_attention',
    'attention',
    'attention_with_weights',
    'safe_flash_attention',
    'set_attention_backend',
    'get_attention_backend',
    'available_attention_backends',
]

_BACKEND_AVAILABILITY = {
    'flash_attn_3': FLASH_ATTN_3_AVAILABLE,
    'flash_attn_2': FLASH_ATTN_2_AVAILABLE,
    'sage': SAGE_ATTENTION_AVAILABLE,
    'sdpa': True,
}

_BACKEND_ERRORS = {
    'flash_attn_3': _FLASH_ATTN_3_ERROR,
    'flash_attn_2': _FLASH_ATTN_2_ERROR,
    'sage': _SAGE_ATTENTION_ERROR,
    'sdpa': None,
}

_BACKEND_ALIASES = {
    'auto': 'auto',
    'default': 'auto',
    'flash': 'flash_attn_2',
    'flashattention': 'flash_attn_2',
    'flash_attn': 'flash_attn_2',
    'flash_attention': 'flash_attn_2',
    'flash3': 'flash_attn_3',
    'flash2': 'flash_attn_2',
    'sage': 'sage',
    'sageattn': 'sage',
    'sageattention': 'sage',
    'sage_attn': 'sage',
    'sage attention': 'sage',
    'sage3': 'sage',
    'sdpa': 'sdpa',
}

_SELECTED_BACKEND = 'auto'


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    # Note: Removed strict asserts to allow more flexibility
    # assert dtype in half_dtypes
    # assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)

        if isinstance(x, tuple):
            x = x[0]
        x = x.unflatten(0, (b, lq))

    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        raise RuntimeError(
            "flash_attention() called but Flash Attention is not available. "
            "Please install flash-attn package or use attention_mode='sdpa' instead."
        )

    # output
    return x.type(out_dtype)


def attention_with_weights(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    average_for_q=False,
    total_video_latent_frames = 21
):
    """
    Compute attention with explicit attention weights for visualization.
    Returns both output and attention weights.
    """
    out_dtype = q.dtype
    
    # Handle sequence lengths
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    
    if q_lens is None:
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        # Ensure q_lens is on the same device as q
        q_lens = q_lens.to(q.device)
        
    if k_lens is None:
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        # Ensure k_lens is on the same device as k
        k_lens = k_lens.to(k.device)
    
    # Apply q_scale if provided
    if q_scale is not None:
        q = q * q_scale
    
    # Compute attention weights manually
    # q: [B, Lq, Nq, C], k: [B, Lk, Nk, C]
    scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)
    
    # Compute scores: [B, Nq, Lq, Lk]
    scores = torch.einsum('blhd,bshd->bhls', q, k) * scale
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(lq, lk, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Mask for k_lens (columns)
    k_mask = torch.arange(lk, device=k.device).unsqueeze(0) >= k_lens.unsqueeze(1)  # [B, Lk]
    scores.masked_fill_(k_mask.unsqueeze(1).unsqueeze(2), float('-inf'))  # [B, 1, 1, Lk]
    
    # Mask for q_lens (rows) 
    q_mask = torch.arange(lq, device=q.device).unsqueeze(0) >= q_lens.unsqueeze(1)  # [B, Lq]
    scores.masked_fill_(q_mask.unsqueeze(1).unsqueeze(3), float('-inf'))  # [B, 1, Lq, 1]
    
    # Compute attention weights
    attn_weights = torch.softmax(scores, dim=-1)  # [B, Nq, Lq, Lk]
    assert attn_weights.shape[0] == 1, "Batch size > 1 not supported for attention visualization."
    
    # Average attention weights to reduce memory usage before returning
    # Average across batch dimension (should be 1) and query heads and query sequence length
    # This gives us attention weight per video token: [Lk]
    if average_for_q:
        #avg_attn_weights = torch.mean(attn_weights, dim=(0, 1, 3))  # [Lq]
        avg_attn_weights = torch.max(attn_weights, dim=3)[0].mean(dim=(0, 1))  # [Lq]
    else:
        if 0:
            avg_attn_weights = torch.mean(attn_weights, dim=(0, 1, 2))  # [Lk]
        elif 1:
            B, H, Lq, Lk = attn_weights.shape  # [1, H, Lq, Lk]
            per_frame_seq_len = Lk // total_video_latent_frames
            per_frame_aud_len = Lq // total_video_latent_frames

            avg_attn_weights = torch.zeros((Lk,), device=attn_weights.device, dtype=attn_weights.dtype)

            eps = 1e-8  # numerical stability
            for i in range(total_video_latent_frames):
                start_idx_v = i * per_frame_seq_len
                end_idx_v   = (i + 1) * per_frame_seq_len

                start_idx_a = i * per_frame_aud_len
                end_idx_a   = (i + 1) * per_frame_aud_len

                # attn_chunk: [H, La, Lv]
                attn_chunk = attn_weights[0, :, start_idx_a:end_idx_a, start_idx_v:end_idx_v]

                # ---- Head informativeness via (low) entropy over Lv ----
                # Normalize within the Lv slice per (head, query) to make a proper distribution
                p = attn_chunk / (attn_chunk.sum(dim=-1, keepdim=True) + eps)          # [H, La, Lv]
                entropy = -(p * (p + eps).log()).sum(dim=-1).mean(dim=1)               # [H]

                # Convert to positive head weights (lower entropy -> larger weight)
                saliency = 1.0 / (entropy + 1e-6)                                      # [H]
                head_w = saliency / (saliency.sum() + eps)                             # [H], sum=1

                # Reduce across audio queries first (pick strong responses), then weight heads
                per_head = torch.amax(attn_chunk, dim=1)                               # [H, Lv]
                weighted = (per_head * head_w[:, None]).sum(dim=0)                     # [Lv]

                avg_attn_weights[start_idx_v:end_idx_v] = weighted
        else:
            avg_attn_weights = torch.mean(attn_weights, dim=(0, 2)).max(dim=(0))[0]  # [Lk]
    
    # Compute output: [B, Lq, Nq, C]
    out = torch.einsum('bhls,bshd->blhd', attn_weights, v)
    
    return out.to(out_dtype), avg_attn_weights.to(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    attention_mode='sdpa',
):
    """
    Unified attention function with multiple backend support.

    attention_mode options:
    - 'sdpa': PyTorch scaled_dot_product_attention (default, no extra dependencies)
    - 'flash_attn_2': Flash Attention 2 (requires flash-attn package)
    - 'flash_attn_3': Flash Attention 3 (requires flash-attn 3.x package)
    """
    if "flash" in attention_mode:
        if attention_mode == 'flash_attn_2':
            fa_version = 2
        elif attention_mode == 'flash_attn_3':
            fa_version = 3
        else:
            fa_version = None
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    elif attention_mode == 'sage':
        if not SAGE_ATTENTION_AVAILABLE:
            err = _SAGE_ATTENTION_ERROR
            if err is not None:
                raise RuntimeError("SageAttention backend requested but failed to initialise.") from err
            raise RuntimeError("SageAttention backend requested but it is not available.")

        b, lq, _, _ = q.shape
        lk = k.size(1)
        out_dtype = q.dtype
        device = q.device

        if q_lens is None:
            q_lens_tensor = torch.full((b,), lq, dtype=torch.int32, device=device)
        else:
            q_lens_tensor = q_lens.to(dtype=torch.int32, device=device)

        if k_lens is None:
            k_lens_tensor = torch.full((b,), lk, dtype=torch.int32, device=device)
        else:
            k_lens_tensor = k_lens.to(dtype=torch.int32, device=device)

        target_dtype = q.dtype
        if k.dtype != target_dtype or k.device != device:
            k = k.to(device=device, dtype=target_dtype)
        if v.dtype != target_dtype or v.device != device:
            v = v.to(device=device, dtype=target_dtype)

        scale = softmax_scale

        q_heads = q.transpose(1, 2).contiguous()
        k_heads = k.transpose(1, 2).contiguous()
        v_heads = v.transpose(1, 2).contiguous()

        same_q = bool(torch.all(q_lens_tensor == lq))
        same_k = bool(torch.all(k_lens_tensor == lk))

        if same_q and same_k:
            attn_out = _sageattn(
                q_heads,
                k_heads,
                v_heads,
                causal=causal,
                scale=scale,
                dropout_p=dropout_p,
                window_size=window_size,
                deterministic=deterministic,
            )
        else:
            attn_out = q_heads.new_zeros(q_heads.shape)
            q_lens_list = q_lens_tensor.int().cpu().tolist()
            k_lens_list = k_lens_tensor.int().cpu().tolist()
            for idx in range(b):
                q_len = int(q_lens_list[idx])
                k_len = int(k_lens_list[idx])
                if q_len <= 0 or k_len <= 0:
                    continue
                attn_slice = _sageattn(
                    q_heads[idx:idx + 1, :, :q_len, :].contiguous(),
                    k_heads[idx:idx + 1, :, :k_len, :].contiguous(),
                    v_heads[idx:idx + 1, :, :k_len, :].contiguous(),
                    causal=causal,
                    scale=scale,
                    dropout_p=dropout_p,
                    window_size=window_size,
                    deterministic=deterministic,
                )
                attn_out[idx, :, :q_len, :] = attn_slice[0]

        return attn_out.transpose(1, 2).contiguous().to(out_dtype)
    elif attention_mode == 'sdpa':
        # PyTorch native attention - always available, no extra dependencies
        b, lq, _, _ = q.shape
        lk = k.size(1)
        device = q.device

        if q_lens is not None or k_lens is not None:
            # Respect per-sample padding by slicing to the real lengths before invoking SDPA.
            q_lens_tensor = (
                q_lens.to(dtype=torch.int32, device=device)
                if q_lens is not None
                else torch.full((b,), lq, dtype=torch.int32, device=device)
            )
            k_lens_tensor = (
                k_lens.to(dtype=torch.int32, device=device)
                if k_lens is not None
                else torch.full((b,), lk, dtype=torch.int32, device=device)
            )

            same_q = bool(torch.all(q_lens_tensor == lq))
            same_k = bool(torch.all(k_lens_tensor == lk))

            if not (same_q and same_k):
                if k.dtype != q.dtype or k.device != device:
                    k = k.to(device=device, dtype=q.dtype)
                if v.dtype != q.dtype or v.device != device:
                    v = v.to(device=device, dtype=q.dtype)

                q_heads = q.transpose(1, 2).contiguous()
                k_heads = k.transpose(1, 2).contiguous()
                v_heads = v.transpose(1, 2).contiguous()

                out_heads = q_heads.new_zeros(q_heads.shape)
                q_lens_list = q_lens_tensor.int().cpu().tolist()
                k_lens_list = k_lens_tensor.int().cpu().tolist()

                for idx in range(b):
                    q_len = int(q_lens_list[idx])
                    k_len = int(k_lens_list[idx])
                    if q_len <= 0 or k_len <= 0:
                        continue
                    slice_out = torch.nn.functional.scaled_dot_product_attention(
                        q_heads[idx:idx + 1, :, :q_len, :].contiguous(),
                        k_heads[idx:idx + 1, :, :k_len, :].contiguous(),
                        v_heads[idx:idx + 1, :, :k_len, :].contiguous(),
                        is_causal=causal,
                        dropout_p=dropout_p,
                    )
                    out_heads[idx, :, :q_len, :] = slice_out[0]

                return out_heads.transpose(1, 2).contiguous()

        # Handle dtype mismatch
        if not (q.dtype == k.dtype == v.dtype):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2).to(q.dtype),
                v.transpose(1, 2).to(q.dtype),
                is_causal=causal,
                dropout_p=dropout_p
            ).transpose(1, 2).contiguous()

        # Standard path
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=causal,
            dropout_p=dropout_p
        ).transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown attention_mode: {attention_mode}")


def _normalize_backend_name(name: Optional[str]) -> str:
    if not name:
        return 'auto'
    lowered = name.lower()
    return _BACKEND_ALIASES.get(lowered, lowered)


def _resolve_backend(requested: Optional[str]) -> str:
    backend = _normalize_backend_name(requested)
    if backend == 'auto':
        for candidate in ('flash_attn_3', 'flash_attn_2', 'sage', 'sdpa'):
            if _BACKEND_AVAILABILITY.get(candidate, False):
                return candidate
        raise RuntimeError('No attention backend is available. Install flash-attn or use PyTorch 2.0+ for SDPA.')
    if backend not in _BACKEND_AVAILABILITY:
        raise ValueError(f"Unknown attention backend '{requested}'.")
    if not _BACKEND_AVAILABILITY[backend]:
        err = _BACKEND_ERRORS.get(backend)
        if err is not None:
            raise RuntimeError(f"Attention backend '{backend}' failed to initialise: {err}") from err
        raise RuntimeError(f"Attention backend '{backend}' is not available.")
    return backend


def set_attention_backend(name: str) -> str:
    global _SELECTED_BACKEND
    resolved = _resolve_backend(name)
    _SELECTED_BACKEND = _normalize_backend_name(name)
    return resolved


def get_attention_backend() -> str:
    return _SELECTED_BACKEND


def available_attention_backends(include_auto: bool = True) -> list[str]:
    priority = ('flash_attn_3', 'flash_attn_2', 'sage', 'sdpa')
    names = [name for name in priority if _BACKEND_AVAILABILITY.get(name, False)]
    extra = [
        name
        for name, available in _BACKEND_AVAILABILITY.items()
        if available and name not in priority
    ]
    names.extend(sorted(set(extra)))
    if include_auto:
        return ['auto'] + names
    return names


def safe_flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    backend = _resolve_backend(_SELECTED_BACKEND)
    if backend == 'flash_attn_3':
        return flash_attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=3,
        )
    if backend == 'flash_attn_2':
        return flash_attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=2,
        )
    if backend == 'sage':
        return attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            attention_mode='sage',
        )
    return attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        attention_mode='sdpa',
    )
