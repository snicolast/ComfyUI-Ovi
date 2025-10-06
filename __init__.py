"""Register ComfyUI-Ovi custom nodes."""

import os
import sys

_PACKAGE_ROOT = os.path.dirname(__file__)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from .nodes.ovi_engine_loader import OviEngineLoader
from .nodes.ovi_video_generator import OviVideoGenerator
from .nodes.ovi_attention_selector import OviAttentionSelector
from .nodes.ovi_wan_component_loader import OviWanComponentLoader

NODE_CLASS_MAPPINGS = {
    "OviEngineLoader": OviEngineLoader,
    "OviVideoGenerator": OviVideoGenerator,
    "OviAttentionSelector": OviAttentionSelector,
    "OviWanComponentLoader": OviWanComponentLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OviEngineLoader": "OVI Engine Loader",
    "OviVideoGenerator": "OVI Video Generator",
    "OviAttentionSelector": "OVI Attention Selector",
    "OviWanComponentLoader": "OVI Wan Component Loader",
}
