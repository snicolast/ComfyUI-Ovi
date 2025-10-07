"Per-node modules will live beside this package."

from .ovi_engine_loader import OviEngineLoader
from .ovi_video_generator import OviVideoGenerator
from .ovi_attention_selector import OviAttentionSelector
from .ovi_wan_component_loader import OviWanComponentLoader
from .ovi_latent_decoder import OviLatentDecoder

NODE_CLASS_MAPPINGS = {
    "OviEngineLoader": OviEngineLoader,
    "OviVideoGenerator": OviVideoGenerator,
    "OviAttentionSelector": OviAttentionSelector,
    "OviWanComponentLoader": OviWanComponentLoader,
    "OviLatentDecoder": OviLatentDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OviEngineLoader": "OVI Engine Loader",
    "OviVideoGenerator": "OVI Video Generator",
    "OviAttentionSelector": "OVI Attention Selector",
    "OviWanComponentLoader": "OVI Wan Component Loader",
    "OviLatentDecoder": "OVI Latent Decoder",
}
