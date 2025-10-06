"""ComfyUI node that wraps OviFusionEngine.generate for T2V / I2V workflows."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from ovi.ovi_fusion_engine import OviFusionEngine

DEFAULT_SAMPLE_RATE = 16000


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor (B, H, W, C) in [0,1] to a PIL RGB image."""
    if image is None:
        return None
    if isinstance(image, list):
        image = image[0]
    if image.ndim == 4:
        image = image[0]
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


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


class OviVideoGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("OVI_ENGINE",),
                "text_prompt": ("STRING", {"multiline": True}),
                "video_height": ("INT", {"default": 512, "min": 128, "max": 1280, "step": 32}),
                "video_width": ("INT", {"default": 960, "min": 128, "max": 1280, "step": 32}),
                "seed": ("INT", {"default": 100, "min": 0, "max": 1_000_000}),
                "solver_name": (["unipc", "euler", "dpm++"], {"default": "unipc"}),
                "sample_steps": ("INT", {"default": 50, "min": 20, "max": 160}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "video_guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "slg_layer": ("INT", {"default": 11, "min": -1, "max": 30}),
                "video_negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "audio_negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "first_frame_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "generate"
    CATEGORY = "Ovi"

    def generate(
        self,
        components: OviFusionEngine,
        text_prompt: str,
        video_height: int,
        video_width: int,
        seed: int,
        solver_name: str,
        sample_steps: int,
        shift: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        first_frame_image: Optional[torch.Tensor] = None,
    ):
        if not isinstance(components, OviFusionEngine):
            raise TypeError("components input must come from OviEngineLoader")

        init_image = _tensor_to_pil(first_frame_image) if first_frame_image is not None else None
        result = components.generate(
            text_prompt=text_prompt,
            image_path=init_image,
            video_frame_height_width=[video_height, video_width],
            seed=seed,
            solver_name=solver_name,
            sample_steps=sample_steps,
            shift=shift,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            slg_layer=slg_layer,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
        )

        if result is None:
            raise RuntimeError("OVI generation failed. Check console logs for details.")

        video_numpy, audio_numpy, _ = result
        video_tensor = _video_to_tensor(video_numpy)
        audio_dict = _audio_to_comfy(audio_numpy, DEFAULT_SAMPLE_RATE)

        return (video_tensor, audio_dict)
