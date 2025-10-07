import torch
import os
import json
from pathlib import Path

from safetensors.torch import load_file

from ovi.modules.fusion import FusionModel
from ovi.modules.t5 import T5EncoderModel
from ovi.modules.vae2_2 import Wan2_2_VAE
from ovi.modules.mmaudio.features_utils import FeaturesUtils

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def init_wan_vae_2_2(ckpt_dir, rank=0):
    device = rank
    device_index = 0
    if isinstance(device, int):
        device_index = device
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            device_index = int(device.split(":", 1)[1])
        except (ValueError, IndexError):
            device_index = 0

    if torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "device_count") and device_index < torch.cuda.device_count():
                torch.cuda.set_device(device_index)
            supports_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        except Exception:
            supports_bf16 = False
    else:
        supports_bf16 = False

    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    vae_model = Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        dtype=dtype,
        device=device,
    )

    return vae_model


def init_mmaudio_vae(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
    bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config['tod_vae_ckpt'] = tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae


def init_fusion_score_model_ovi(rank: int = 0, meta_init=False):
    video_config_path = CONFIG_ROOT / "model" / "dit" / "video.json"
    audio_config_path = CONFIG_ROOT / "model" / "dit" / "audio.json"

    if not video_config_path.exists():
        raise FileNotFoundError(f"Missing video config at {video_config_path}")
    if not audio_config_path.exists():
        raise FileNotFoundError(f"Missing audio config at {audio_config_path}")

    with video_config_path.open() as f:
        video_config = json.load(f)

    with audio_config_path.open() as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)

    params_all = sum(p.numel() for p in fusion_model.parameters())

    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config


def init_text_model(ckpt_dir, rank):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        shard_fn=None)

    return text_encoder


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False):
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else:
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        missing, unexpected = model.load_state_dict(df, strict=True, assign=from_meta)

        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else:
        raise RuntimeError("{checkpoint=} does not exists'")
