"""Load Wan VAE and UMT5 encoder using ComfyUI-style selectors."""
from __future__ import annotations

from pathlib import Path

import folder_paths
import torch

from ovi.modules.t5 import T5EncoderModel
from ovi.modules.vae2_2 import Wan2_2_VAE


class OviWanComponentLoader:
    @classmethod
    def INPUT_TYPES(cls):
        vae_files = folder_paths.get_filename_list('vae') or ['']
        text_files = folder_paths.get_filename_list('text_encoders') or ['']
        default_vae = 'Wan2.2_VAE.pth' if 'Wan2.2_VAE.pth' in vae_files else vae_files[0]
        default_umt5 = 'models_t5_umt5-xxl-enc-bf16.pth' if 'models_t5_umt5-xxl-enc-bf16.pth' in text_files else text_files[0]
        return {
            "required": {
                "engine": ("OVI_ENGINE",),
                "vae_file": (vae_files, {"default": default_vae}),
                "umt5_file": (text_files, {"default": default_umt5}),
            },
        }

    RETURN_TYPES = ("OVI_ENGINE",)
    RETURN_NAMES = ("components",)
    FUNCTION = "load"
    CATEGORY = "Ovi"

    def load(self, engine, vae_file: str, umt5_file: str, tokenizer: str = ''):
        from ovi.ovi_fusion_engine import OviFusionEngine

        if not isinstance(engine, OviFusionEngine):
            raise TypeError("engine input must come from OviEngineLoader")

        vae_path = Path(folder_paths.get_full_path_or_raise('vae', vae_file)).resolve()
        umt5_path = Path(folder_paths.get_full_path_or_raise('text_encoders', umt5_file)).resolve()

        wan_device = engine.device if not getattr(engine, "cpu_offload", False) else 'cpu'
        wan_vae = Wan2_2_VAE(device=wan_device, vae_pth=str(vae_path))
        wan_vae.model.requires_grad_(False).eval()

        tokenizer_path = Path(engine.get_config().ckpt_dir) / 'google' / 'umt5-xxl'
        if not tokenizer_path.exists():
            raise FileNotFoundError(f'Wan tokenizer not found at {tokenizer_path}. Run OviEngineLoader with auto_download or place it manually.')

        text_device = engine.device if not getattr(engine, "cpu_offload", False) else 'cpu'
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=text_device,
            checkpoint_path=str(umt5_path),
            tokenizer_path=str(tokenizer_path),
            shard_fn=None,
        )

        engine.override_models(video_vae=wan_vae, text_model=text_encoder)
        return (engine,)
