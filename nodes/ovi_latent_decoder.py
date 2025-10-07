"""ComfyUI node that decodes OVI latent tensors into IMAGE and AUDIO outputs."""
from __future__ import annotations

import torch

from ovi.ovi_fusion_engine import OviFusionEngine
from comfy.utils import ProgressBar

DEFAULT_SAMPLE_RATE = 16000


def _video_to_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
    """Convert (C, F, H, W) tensor in [0,1] to (F, H, W, C) tensor."""
    if not isinstance(video_tensor, torch.Tensor):
        raise TypeError("video tensor must be a torch.Tensor")
    tensor = video_tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    tensor = tensor.to(torch.float32)
    if tensor.dim() != 4:
        raise ValueError("decoded video must have shape [C, F, H, W]")
    tensor = tensor.clamp(0.0, 1.0)
    return tensor.permute(1, 2, 3, 0).contiguous()


def _audio_to_comfy(audio_tensor: torch.Tensor, sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict:
    """Convert torch audio tensor to ComfyUI AUDIO dict."""
    if not isinstance(audio_tensor, torch.Tensor):
        raise TypeError("audio tensor must be a torch.Tensor")
    waveform = audio_tensor.detach()
    if waveform.device.type != "cpu":
        waveform = waveform.cpu()
    waveform = waveform.to(torch.float32).flatten().unsqueeze(0).unsqueeze(0)  # [B=1, C=1, T]
    return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}


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

        pbar = ProgressBar(total=2)
        decoded_video, decoded_audio = components.decode_latents(
            video_latents=video_latents,
            audio_latents=audio_latents,
            to_cpu=True,
        )
        pbar.update(1)

        if decoded_video is None or decoded_audio is None:
            raise RuntimeError("OVI decode failed. Check console logs for details.")

        video_tensor = _video_to_tensor(decoded_video)
        audio_dict = _audio_to_comfy(decoded_audio, DEFAULT_SAMPLE_RATE)
        pbar.update(1)

        return (video_tensor, audio_dict)
