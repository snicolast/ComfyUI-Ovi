# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    import flash_attn_interface  # type: ignore
    FLASH_ATTN_3_AVAILABLE = True
    _FLASH_ATTN_3_ERROR = None
except Exception as exc:  # pragma: no cover - import-time optional dependency
    flash_attn_interface = None  # type: ignore
    FLASH_ATTN_3_AVAILABLE = False
    _FLASH_ATTN_3_ERROR = exc

try:
    import flash_attn  # type: ignore
    FLASH_ATTN_2_AVAILABLE = True
    _FLASH_ATTN_2_ERROR = None
except Exception as exc:  # pragma: no cover - import-time optional dependency
    flash_attn = None  # type: ignore
    FLASH_ATTN_2_AVAILABLE = False
    _FLASH_ATTN_2_ERROR = exc

try:
    from sageattention import sageattn  # type: ignore
    SAGE_ATTENTION_AVAILABLE = True
    _SAGE_ATTENTION_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    sageattn = None  # type: ignore
    SAGE_ATTENTION_AVAILABLE = False
    _SAGE_ATTENTION_ERROR = exc

try:
    from sageattn import sageattn_blackwell  # type: ignore
    SAGE_ATTENTION_BLACKWELL_AVAILABLE = True
    _SAGE_ATTENTION_BLACKWELL_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    sageattn_blackwell = None  # type: ignore
    SAGE_ATTENTION_BLACKWELL_AVAILABLE = False
    _SAGE_ATTENTION_BLACKWELL_ERROR = exc

__all__ = [
    'flash_attention',
    'attention',
    'attention_with_weights',
    'set_attention_backend',
    'get_attention_backend',
    'available_attention_backends',
]

_HALF_DTYPES = (torch.float16, torch.bfloat16)
_AUTO_PRIORITY = (
    'flash_attn_3',
    'flash_attn_2',
    'sage_attn_blackwell',
    'sage_attn',
    'sdpa',
)

_BACKEND_AVAILABILITY: Dict[str, bool] = {
    'flash_attn_3': FLASH_ATTN_3_AVAILABLE,
    'flash_attn_2': FLASH_ATTN_2_AVAILABLE,
    'sage_attn': SAGE_ATTENTION_AVAILABLE,
    'sage_attn_blackwell': SAGE_ATTENTION_BLACKWELL_AVAILABLE,
    'sdpa': True,
}

_BACKEND_ERRORS = {
    'flash_attn_3': _FLASH_ATTN_3_ERROR,
    'flash_attn_2': _FLASH_ATTN_2_ERROR,
    'sage_attn': _SAGE_ATTENTION_ERROR,
    'sage_attn_blackwell': _SAGE_ATTENTION_BLACKWELL_ERROR,
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
    'sage': 'sage_attn',
    'sageattention': 'sage_attn',
    'sage3': 'sage_attn_blackwell',
    'sdpa': 'sdpa',
}

_SELECTED_BACKEND = 'auto'


def available_attention_backends(include_auto: bool = True) -> List[str]:
    """Return the attention backends that are currently usable."""
    names = [name for name, available in _BACKEND_AVAILABILITY.items() if available]
    if include_auto:
        return ['auto'] + names
    return names


def get_attention_backend() -> str:
    return _SELECTED_BACKEND


def set_attention_backend(name: str) -> str:
    """Select the attention backend to use for subsequent calls.

    Args:
        name: One of ``auto`` or a backend returned by
            :func:`available_attention_backends`.

    Returns:
        The resolved backend name that will be used.
    """
    global _SELECTED_BACKEND
    key = _normalize_backend_name(name)
    if key != 'auto' and not _BACKEND_AVAILABILITY.get(key, False):
        error = _BACKEND_ERRORS.get(key)
        if error is not None:
            raise RuntimeError(
                f"Attention backend '{name}' is unavailable: {error}"
            ) from error
        raise RuntimeError(f"Attention backend '{name}' is unavailable on this system.")
    resolved = _resolve_backend(key)
    _SELECTED_BACKEND = key
    return resolved


def _normalize_backend_name(name: Optional[str]) -> str:
    if not name:
        return 'auto'
    lowered = name.lower()
    return _BACKEND_ALIASES.get(lowered, lowered)


def _resolve_backend(requested: Optional[str]) -> str:
    backend = _normalize_backend_name(requested)
    if backend == 'auto':
        for candidate in _AUTO_PRIORITY:
            if _BACKEND_AVAILABILITY.get(candidate, False):
                return candidate
        raise RuntimeError(
            'No attention backend is available. Install flash-attn, sageattention, or use PyTorch 2.0+ for SDPA.'
        )
    if backend not in _BACKEND_AVAILABILITY:
        raise ValueError(f"Unknown attention backend '{requested}'.")
    if not _BACKEND_AVAILABILITY[backend]:
        error = _BACKEND_ERRORS.get(backend)
        if error is not None:
            raise RuntimeError(
                f"Attention backend '{backend}' failed to initialize: {error}"
            ) from error
        raise RuntimeError(f"Attention backend '{backend}' is not available.")
    return backend


def _normalize_lens(lens: Optional[torch.Tensor], default_len: int, batch: int, device: torch.device) -> torch.Tensor:
    if lens is None:
        return torch.full((batch,), default_len, dtype=torch.int32, device=device)
    if isinstance(lens, torch.Tensor):
        return lens.to(device=device, dtype=torch.int32)
    return torch.tensor(lens, dtype=torch.int32, device=device)


def _build_padding_mask(
    q_lens: Optional[torch.Tensor],
    k_lens: Optional[torch.Tensor],
    lq: int,
    lk: int,
    batch: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if q_lens is None and k_lens is None:
        return None

    q_lens_t = _normalize_lens(q_lens, lq, batch, device)
    k_lens_t = _normalize_lens(k_lens, lk, batch, device)

    q_valid = torch.arange(lq, device=device).unsqueeze(0) < q_lens_t.unsqueeze(1)
    k_valid = torch.arange(lk, device=device).unsqueeze(0) < k_lens_t.unsqueeze(1)

    valid = q_valid.unsqueeze(2) & k_valid.unsqueeze(1)
    return ~valid  # True where we should mask


def _flash_attn_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor],
    k_lens: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: Optional[float],
    q_scale: Optional[float],
    causal: bool,
    window_size: tuple,
    deterministic: bool,
    dtype: torch.dtype,
    version: int,
) -> torch.Tensor:
    if q.device.type != 'cuda':
        raise RuntimeError('FlashAttention kernels require CUDA tensors.')
    if q.size(-1) > 256:
        raise RuntimeError(f'FlashAttention requires head_dim <= 256, got {q.size(-1)}.')
    if dtype not in _HALF_DTYPES:
        raise ValueError(f'FlashAttention expects dtype float16/bfloat16, got {dtype}.')

    if version == 3 and not FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError('flash_attn_interface (FlashAttention-3) is not available.')
    if version == 2 and not FLASH_ATTN_2_AVAILABLE:
        raise RuntimeError('flash_attn (FlashAttention-2) is not available.')

    b, lq, lk = q.size(0), q.size(1), k.size(1)
    out_dtype = q.dtype

    def _half(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype in _HALF_DTYPES else x.to(dtype)

    q_lens_t = _normalize_lens(q_lens, lq, b, q.device)
    k_lens_t = _normalize_lens(k_lens, lk, b, k.device)
    q_seq = q_lens_t.tolist()
    k_seq = k_lens_t.tolist()

    if q_lens is None:
        q_flat = _half(q.flatten(0, 1))
    else:
        q_flat = _half(torch.cat([u[:seq] for u, seq in zip(q, q_seq)], dim=0))

    if k_lens is None:
        k_flat = _half(k.flatten(0, 1))
        v_flat = _half(v.flatten(0, 1))
    else:
        k_flat = _half(torch.cat([u[:seq] for u, seq in zip(k, k_seq)], dim=0))
        v_flat = _half(torch.cat([u[:seq] for u, seq in zip(v, k_seq)], dim=0))

    q_flat = q_flat.to(v_flat.dtype)
    k_flat = k_flat.to(v_flat.dtype)

    if q_scale is not None:
        q_flat = q_flat * q_scale

    cu_q = torch.zeros(b + 1, dtype=torch.int32, device=q.device)
    cu_q[1:] = q_lens_t
    cu_q = torch.cumsum(cu_q, dim=0)

    cu_k = torch.zeros(b + 1, dtype=torch.int32, device=k.device)
    cu_k[1:] = k_lens_t
    cu_k = torch.cumsum(cu_k, dim=0)

    if version == 3:
        outputs = flash_attn_interface.flash_attn_varlen_func(  # type: ignore[attr-defined]
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        x = outputs.unflatten(0, (b, lq))
    else:
        try:
            outputs = flash_attn.flash_attn_varlen_func(  # type: ignore[attr-defined]
                q=q_flat,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "cu_seqlens" in message or "dtype int32" in message:
                warnings.warn(
                    "FlashAttention varlen kernels rejected the sequence lengths; falling back to SDPA backend.",
                    RuntimeWarning,
                )
                return _sdpa_attention(
                    q,
                    k,
                    v,
                    q_lens,
                    k_lens,
                    dropout_p,
                    softmax_scale,
                    q_scale,
                    causal,
                    window_size,
                    deterministic,
                )
            raise
        x = outputs.unflatten(0, (b, lq))

    return x.to(out_dtype)


def _sage_attn_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor],
    k_lens: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: Optional[float],
    q_scale: Optional[float],
    causal: bool,
    window_size: tuple,
) -> torch.Tensor:
    if sageattn is None:
        raise RuntimeError('sageattention backend is not installed.')
    if window_size != (-1, -1):
        warnings.warn('SageAttention backend ignores window_size; proceeding with full attention.')

    if q_scale is not None:
        q = q * q_scale

    attn_mask = _build_padding_mask(q_lens, k_lens, q.size(1), k.size(1), q.size(0), q.device)
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1)

    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)
    if q.dtype == torch.float32:
        output = sageattn(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal,
            tensor_layout='NHD',
        )
        return output.to(torch.float32)

    return sageattn(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        tensor_layout='NHD',
    )


def _sage_attn_blackwell_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor],
    k_lens: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: Optional[float],
    q_scale: Optional[float],
    causal: bool,
    window_size: tuple,
) -> torch.Tensor:
    if sageattn_blackwell is None:
        raise RuntimeError('sageattn_blackwell backend is not installed.')
    if window_size != (-1, -1):
        warnings.warn('SageAttention Blackwell backend ignores window_size; proceeding with full attention.')

    if q_scale is not None:
        q = q * q_scale

    attn_mask = _build_padding_mask(q_lens, k_lens, q.size(1), k.size(1), q.size(0), q.device)
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1)

    return sageattn_blackwell(  # type: ignore[operator]
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        tensor_layout='HND',
        per_block_mean=False,
    ).transpose(1, 2).contiguous()


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor],
    k_lens: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: Optional[float],
    q_scale: Optional[float],
    causal: bool,
    window_size: tuple,
    deterministic: bool,
) -> torch.Tensor:
    if window_size != (-1, -1):
        warnings.warn('SDPA backend does not support windowed attention; falling back to global attention.')

    b, lq, heads, _ = q.shape
    lk = k.size(1)
    out_dtype = q.dtype

    if q_scale is not None:
        q = q * q_scale

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    if softmax_scale is not None:
        q_t = q_t * softmax_scale

    attn_mask = _build_padding_mask(q_lens, k_lens, lq, lk, b, q.device)
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1)

    out = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
    )

    return out.transpose(1, 2).contiguous().to(out_dtype)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: Optional[int] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Unified attention entry point with multiple backend implementations."""
    requested_backend = backend
    if requested_backend is None and version is not None:
        if version == 3:
            requested_backend = 'flash_attn_3'
        elif version == 2:
            requested_backend = 'flash_attn_2'
        else:
            warnings.warn(f'Unknown flash attention version {version}; defaulting to auto backend selection.')

    backend_impl = _resolve_backend(requested_backend)

    if backend_impl == 'flash_attn_3':
        return _flash_attn_backend(
            q,
            k,
            v,
            q_lens,
            k_lens,
            dropout_p,
            softmax_scale,
            q_scale,
            causal,
            window_size,
            deterministic,
            dtype,
            version=3,
        )
    if backend_impl == 'flash_attn_2':
        return _flash_attn_backend(
            q,
            k,
            v,
            q_lens,
            k_lens,
            dropout_p,
            softmax_scale,
            q_scale,
            causal,
            window_size,
            deterministic,
            dtype,
            version=2,
        )
    if backend_impl == 'sage_attn_blackwell':
        return _sage_attn_blackwell_backend(
            q,
            k,
            v,
            q_lens,
            k_lens,
            dropout_p,
            softmax_scale,
            q_scale,
            causal,
            window_size,
        )
    if backend_impl == 'sage_attn':
        return _sage_attn_backend(
            q,
            k,
            v,
            q_lens,
            k_lens,
            dropout_p,
            softmax_scale,
            q_scale,
            causal,
            window_size,
        )
    return _sdpa_attention(
        q,
        k,
        v,
        q_lens,
        k_lens,
        dropout_p,
        softmax_scale,
        q_scale,
        causal,
        window_size,
        deterministic,
    )


def attention_with_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    average_for_q: bool = False,
    total_video_latent_frames: int = 21,
) -> tuple:
    """Compute attention and return both outputs and attention weights for visualisation."""
    out_dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    if q_scale is not None:
        q = q * q_scale

    scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)
    scores = torch.einsum('blhd,bshd->bhls', q, k) * scale

    padding_mask = _build_padding_mask(q_lens, k_lens, lq, lk, b, q.device)
    if causal:
        causal_mask = torch.triu(torch.ones(lq, lk, device=q.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0)
        padding_mask = causal_mask if padding_mask is None else (padding_mask | causal_mask)

    if padding_mask is not None:
        scores = scores.masked_fill(padding_mask.unsqueeze(1), float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    if attn_weights.size(0) != 1:
        raise AssertionError('attention_with_weights currently supports batch size == 1 for visualisation paths.')

    if average_for_q:
        avg_attn_weights = torch.max(attn_weights, dim=3)[0].mean(dim=(0, 1))
    else:
        _, heads, q_tokens, v_tokens = attn_weights.shape
        per_frame_seq_len = v_tokens // total_video_latent_frames
        per_frame_aud_len = q_tokens // total_video_latent_frames

        avg_attn_weights = torch.zeros((v_tokens,), device=attn_weights.device, dtype=attn_weights.dtype)
        eps = 1e-8
        for i in range(total_video_latent_frames):
            start_idx_v = i * per_frame_seq_len
            end_idx_v = (i + 1) * per_frame_seq_len
            start_idx_a = i * per_frame_aud_len
            end_idx_a = (i + 1) * per_frame_aud_len

            attn_chunk = attn_weights[0, :, start_idx_a:end_idx_a, start_idx_v:end_idx_v]
            p = attn_chunk / (attn_chunk.sum(dim=-1, keepdim=True) + eps)
            entropy = -(p * (p + eps).log()).sum(dim=-1).mean(dim=1)
            saliency = 1.0 / (entropy + 1e-6)
            head_w = saliency / (saliency.sum() + eps)

            per_head = torch.amax(attn_chunk, dim=1)
            weighted = (per_head * head_w[:, None]).sum(dim=0)
            avg_attn_weights[start_idx_v:end_idx_v] = weighted

    out = torch.einsum('bhls,bshd->blhd', attn_weights, v)
    return out.to(out_dtype), avg_attn_weights.to(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version: Optional[int] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
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
        backend=backend,
    )
