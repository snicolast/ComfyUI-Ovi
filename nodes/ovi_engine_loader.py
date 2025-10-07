"""ComfyUI node for initializing and caching the OviFusionEngine."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import logging

import torch

import comfy.model_management as model_management
from omegaconf import OmegaConf

from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from ovi.utils.checkpoint_manager import ensure_checkpoints, MissingDependencyError, DownloadError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT_DIR = str((REPO_ROOT / "ckpts").resolve())

MODEL_VARIANT_CHOICES = [
    ("Ovi-11B-bf16.safetensors", False, "bf16"),
    ("Ovi-11B-fp8.safetensors", True, "fp8"),
]
MODEL_VARIANT_LABELS = [choice[0] for choice in MODEL_VARIANT_CHOICES]
_MODEL_VARIANT_LOOKUP = {label: (fp8, key) for label, fp8, key in MODEL_VARIANT_CHOICES}

# Global cache so multiple graphs can reuse the same heavy engine instance.
_ENGINE_CACHE: Dict[Tuple[int, bool, bool], OviFusionEngine] = {}


def _clear_engine_cache():
    for engine in list(_ENGINE_CACHE.values()):
        try:
            engine.unload()
        except Exception as exc:
            logging.warning('Failed to unload OVI engine: %s', exc)
    _ENGINE_CACHE.clear()


def _register_unload_hook():
    if getattr(model_management, '_ovi_unload_hook', False):
        return
    original_unload = model_management.unload_all_models

    def wrapped_unload_all_models(*args, **kwargs):
        _clear_engine_cache()
        return original_unload(*args, **kwargs)

    model_management.unload_all_models = wrapped_unload_all_models
    model_management._ovi_unload_hook = True


_register_unload_hook()



def _build_config(cpu_offload: bool, fp8: bool) -> OmegaConf:
    """Clone DEFAULT_CONFIG without mutating the module-level object."""
    base = OmegaConf.to_container(DEFAULT_CONFIG, resolve=True)
    config = OmegaConf.create(base)
    config.ckpt_dir = DEFAULT_CKPT_DIR
    config.cpu_offload = bool(cpu_offload)
    config.fp8 = bool(fp8)
    return config


class OviEngineLoader:
    CACHEABLE = False
    @classmethod
    def INPUT_TYPES(cls):
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device_inputs = {}
        if gpu_count > 1:
            device_choices = [
                f"{idx}: {torch.cuda.get_device_name(idx)}" for idx in range(gpu_count)
            ]
            device_inputs["device"] = (device_choices, {"default": device_choices[0]})
        elif gpu_count == 1:
            device_inputs["device"] = ("INT", {"default": 0, "min": 0, "max": 0})
        return {
            "required": {
                "model_precision": (MODEL_VARIANT_LABELS, {"default": MODEL_VARIANT_LABELS[0]}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                **device_inputs,
            }
        }

    RETURN_TYPES = ("OVI_ENGINE",)
    RETURN_NAMES = ("engine",)
    FUNCTION = "load"
    CATEGORY = "Ovi"

    def load(self, model_precision: str, cpu_offload: bool, device=0):
        if model_precision not in _MODEL_VARIANT_LOOKUP:
            raise ValueError(f"Unknown model precision selection '{model_precision}'.")
        fp8, variant_key = _MODEL_VARIANT_LOOKUP[model_precision]
        if isinstance(device, str):
            device = int(device.split(":")[0].strip())

        config = _build_config(cpu_offload, fp8)
        try:
            ensure_checkpoints(config.ckpt_dir, download=True, variants=(variant_key,))
        except MissingDependencyError as exc:
            raise RuntimeError(
                "huggingface_hub package is required for initial downloads."
            ) from exc
        except DownloadError as exc:
            raise RuntimeError(str(exc)) from exc

        cache_key = (device, config.cpu_offload, fp8)

        engine = _ENGINE_CACHE.get(cache_key)
        if engine is None or getattr(engine, 'model', None) is None:
            engine = OviFusionEngine(config=config, device=device)
            _ENGINE_CACHE[cache_key] = engine
            available = engine.available_attention_backends()
            logging.info(
                'OVI engine attention backends: %s (current: %s)',
                ', '.join(available),
                engine.get_attention_backend(resolved=True),
            )
        else:
            engine.config = config

        return (engine,)
