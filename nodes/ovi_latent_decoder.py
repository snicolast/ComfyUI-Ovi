"""ComfyUI node that decodes OVI latent tensors into IMAGE and AUDIO outputs."""
from __future__ import annotations

import numpy as np
import torch

from ovi.ovi_fusion_engine import OviFusionEngine

DEFAULT_SAMPLE_RATE = 16000


def _video_to_tensor(video_numpy: np.ndarray) -> torch.Tensor:
    """Convert (C, F, H, W) array in [-1,1] to (F, H, W, C) tensor in [0,1]."""
    tensor = torch.from_numpy(video_numpy).float()
    tensor = tensor.permute(1, 2, 3, 0)  # F, H, W, C
    tensor = ((tensor + 1.0) * 0.5).clamp(0.0, 1.0)
    return tensor


def _audio_to_comfy(audio_numpy: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict:
    """Convert numpy audio array to ComfyUI AUDIO dict."""
    waveform = torch.from_numpy(audio_numpy).float().flatten()
    waveform = waveform.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, T]
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


class OviLatentDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("OVI_ENGINE",),
                "video_latents": ("OVI_VIDEO_LATENTS",),
                "audio_latents": ("OVI_AUDIO_LATENTS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "decode"
    CATEGORY = "Ovi"

    def decode(
        self,
        components: OviFusionEngine,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ):
        if not isinstance(components, OviFusionEngine):
            raise TypeError("components input must come from OviEngineLoader")

        decoded_video, decoded_audio = components.decode_latents(
            video_latents=video_latents,
            audio_latents=audio_latents,
        )

        if decoded_video is None or decoded_audio is None:
            raise RuntimeError("OVI decode failed. Check console logs for details.")

        video_tensor = _video_to_tensor(decoded_video)
        audio_dict = _audio_to_comfy(decoded_audio, DEFAULT_SAMPLE_RATE)

        return (video_tensor, audio_dict)
