"""ComfyUI helper node to pick OVI attention backend."""
from __future__ import annotations

import logging
from typing import List

from ovi.modules.attention import available_attention_backends
from ovi.ovi_fusion_engine import OviFusionEngine


def _choices() -> List[str]:
    options = available_attention_backends()
    return options if options else ['auto']


def _default_choice(options: List[str]) -> str:
    if 'auto' in options:
        return 'auto'
    return options[0]


class OviAttentionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        options = _choices()
        default = _default_choice(options)
        return {
            "required": {
                "components": ("OVI_ENGINE",),
                "attention_backend": (options, {"default": default}),
            }
        }

    RETURN_TYPES = ("OVI_ENGINE",)
    RETURN_NAMES = ("components",)
    FUNCTION = "set_backend"
    CATEGORY = "Ovi"

    def set_backend(self, components: OviFusionEngine, attention_backend: str):
        if not isinstance(components, OviFusionEngine):
            raise TypeError("components input must come from OviEngineLoader")
        resolved = components.set_attention_backend(attention_backend)
        resolved_display = components.get_attention_backend(resolved=True)
        if attention_backend.lower() == 'auto':
            print(f"[OVI] Attention backend auto resolved to {resolved_display}")
        logging.info(
            'OVI attention backend requested %s, resolved %s',
            attention_backend,
            resolved_display,
        )
        return (components,)
