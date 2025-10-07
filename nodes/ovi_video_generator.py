"""ComfyUI node that wraps OviFusionEngine.generate for T2V / I2V workflows."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from ovi.ovi_fusion_engine import OviFusionEngine


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

    RETURN_TYPES = ("OVI_VIDEO_LATENTS", "OVI_AUDIO_LATENTS", "OVI_ENGINE")
    RETURN_NAMES = ("video_latents", "audio_latents", "components")
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

        video_latents, audio_latents = result
        if not isinstance(video_latents, torch.Tensor) or not isinstance(audio_latents, torch.Tensor):
            raise RuntimeError("OVI engine returned invalid latents. Check console logs for details.")

        return (video_latents.detach(), audio_latents.detach(), components)
