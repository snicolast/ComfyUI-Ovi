import os
import sys
import uuid
import cv2
import glob
import torch
import logging
import folder_paths
import comfy.model_management as model_management
import folder_paths
from textwrap import indent
import torch.nn as nn
from diffusers import FluxPipeline
from tqdm import tqdm
from ovi.distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from ovi.utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from ovi.utils.processing_utils import clean_text, preprocess_image_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible
from ovi.modules.attention import available_attention_backends, get_attention_backend, set_attention_backend
from ovi.utils.checkpoint_manager import (
    OVI_MODEL_SOURCE_NAME,
    OVI_MODEL_TARGET_NAME,
    OVI_MODEL_FP8_SOURCE_NAME,
    OVI_MODEL_FP8_TARGET_NAME,
)

from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = OmegaConf.load(PACKAGE_ROOT / 'configs' / 'inference' / 'inference_fusion.yaml')
try:
    _DIFFUSION_MODEL_DIRS = [Path(p) for p in folder_paths.get_folder_paths('diffusion_models')]
except Exception:
    _DIFFUSION_MODEL_DIRS = []
OVI_MODEL_CANDIDATES_BF16 = [Path(p) / OVI_MODEL_TARGET_NAME for p in _DIFFUSION_MODEL_DIRS]
OVI_MODEL_CANDIDATES_FP8 = [Path(p) / OVI_MODEL_FP8_TARGET_NAME for p in _DIFFUSION_MODEL_DIRS]


class OviFusionEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16):
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        self.config = config
        meta_init = True
        self.cpu_offload = config.get("cpu_offload", False)
        self.fp8 = bool(config.get("fp8", False))
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")
        if self.fp8:
            logging.info("FP8 quantized fusion model requested.")

        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)

        if not meta_init:
            if not self.fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
    
        # Load VAEs
        vae_path = Path(config.ckpt_dir) / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth"
        if vae_path.exists():
            vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
            vae_model_video.model.requires_grad_(False).eval()
            vae_model_video.model = vae_model_video.model.bfloat16()
            self.vae_model_video = vae_model_video
        else:
            self.vae_model_video = None

        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()

        # Load T5 text model
        text_model_path = Path(config.ckpt_dir) / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth"
        if text_model_path.exists():
            self.text_model = init_text_model(config.ckpt_dir, rank=device)
            if config.get("shard_text_model", False):
                raise NotImplementedError("Sharding text model is not implemented yet.")
            if self.cpu_offload:
                self.offload_to_cpu(self.text_model.model)
        else:
            self.text_model = None

        # Find fusion ckpt in the same dir used by other components
        checkpoint_path = None
        fusion_candidates = OVI_MODEL_CANDIDATES_FP8 if self.fp8 else OVI_MODEL_CANDIDATES_BF16
        fallback_source = OVI_MODEL_FP8_SOURCE_NAME if self.fp8 else OVI_MODEL_SOURCE_NAME
        candidate_paths = list(fusion_candidates) + [Path(config.ckpt_dir) / 'Ovi' / fallback_source]
        for candidate in candidate_paths:
            if candidate.exists():
                checkpoint_path = candidate
                break

        if checkpoint_path is None:
            raise RuntimeError('No fusion checkpoint found. Please download Ovi-11B-bf16.safetensors.')

        load_fusion_checkpoint(model, checkpoint_path=str(checkpoint_path), from_meta=meta_init)

        if meta_init:
            if not self.fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()
        self.model = model
        self._requested_attention_backend = 'auto'
        try:
            self._check_cancel()
            self._resolved_attention_backend = set_attention_backend('auto')
        except RuntimeError as exc:
            available = ', '.join(available_attention_backends(include_auto=False))
            raise RuntimeError(
                f"Failed to initialise attention backend (requested 'auto'). Available backends: {available or 'none'}"
            ) from exc

        ## Load t2i as part of pipeline
        if hasattr(self, 'image_model'):
            self.image_model = None
        if hasattr(self, 'image_model'):
            self.image_model = None
        # Fixed attributes, non-configurable
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.audio_latent_length = 157
        self.video_latent_length = 31

        # Track external overrides so they can be restored after reloads.
        self._override_video_vae = getattr(self, "_override_video_vae", None)
        self._override_text_model = getattr(self, "_override_text_model", None)

        logging.info(f"OVI Fusion Engine initialized, cpu_offload={self.cpu_offload}. GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    def ensure_loaded(self):
        """Reload weights in-place if they were released via unload()."""
        model_missing = getattr(self, "model", None) is None
        audio_vae_missing = getattr(self, "vae_model_audio", None) is None
        video_vae_missing = getattr(self, "vae_model_video", None) is None
        text_model_missing = getattr(self, "text_model", None) is None

        # If only missing overridden modules, restore from cached overrides.
        if video_vae_missing and self._override_video_vae is not None:
            self.override_models(video_vae=self._override_video_vae)
            video_vae_missing = False
        if text_model_missing and self._override_text_model is not None:
            self.override_models(text_model=self._override_text_model)
            text_model_missing = False

        if not (model_missing or audio_vae_missing or video_vae_missing or text_model_missing):
            return

        overrides = (self._override_video_vae, self._override_text_model)
        logging.info(
            "Reinitialising OVI Fusion Engine after unload for device %s (fp8=%s, cpu_offload=%s).",
            self.device,
            getattr(self, "fp8", False),
            getattr(self, "cpu_offload", False),
        )
        self.__class__.__init__(self, config=self.config, device=self.device, target_dtype=self.target_dtype)
        if overrides[0] is not None or overrides[1] is not None:
            self.override_models(video_vae=overrides[0], text_model=overrides[1])

    def set_attention_backend(self, backend: str) -> str:
        resolved = set_attention_backend(backend)
        if backend.lower() == 'auto' and resolved == 'auto':
            options = available_attention_backends(include_auto=False)
            if options:
                # Fall back to the first actual backend when attention module leaves auto unresolved
                resolved = options[0]
        self._requested_attention_backend = backend
        self._resolved_attention_backend = resolved
        logging.info(
            'OVI attention backend set to %s (requested %s)',
            resolved,
            backend,
        )
        return resolved

    def get_config(self):
        return self.config

    def resolve_ckpt_path(self, path: str):
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        base = Path(self.config.ckpt_dir)
        return (base / candidate).resolve()

    def override_models(self, video_vae=None, text_model=None):
        if video_vae is not None:
            self.vae_model_video = video_vae
            self._override_video_vae = video_vae
            self._set_video_vae_device(self.device)
            if self.cpu_offload:
                self._set_video_vae_device("cpu")
        if text_model is not None:
            model_obj = getattr(text_model, "model", text_model)
            try:
                model_obj = model_obj.to(device=self.device)
            except Exception:
                pass
            if hasattr(text_model, "model"):
                text_model.model = model_obj
            self.text_model = text_model
            self._override_text_model = text_model
            if self.cpu_offload:
                try:
                    self.offload_to_cpu(text_model.model)
                except Exception:
                    pass

    def _check_cancel(self):
        model_management.throw_exception_if_processing_interrupted()

    def _set_video_vae_device(self, device: str):
        video_vae = self._require_video_vae()
        if hasattr(video_vae, "model"):
            target_dtype = getattr(video_vae, "dtype", self.target_dtype)
            if device == "cpu":
                target_dtype = torch.float32
            video_vae.model = video_vae.model.to(device=device, dtype=target_dtype).eval()
            if device == "cpu":
                try:
                    self.offload_to_cpu(video_vae.model)
                except Exception:
                    pass
        if isinstance(getattr(video_vae, "scale", None), list):
            video_vae.scale = [
                tensor.to(device, dtype=torch.bfloat16)
                if isinstance(tensor, torch.Tensor) else tensor
                for tensor in video_vae.scale
            ]
        return video_vae

    def _require_video_vae(self):
        if self.vae_model_video is None:
            raise RuntimeError('Wan video VAE is not loaded. Please add OviWanComponentLoader to your workflow.')
        return self.vae_model_video

    def _require_text_model(self):
        if self.text_model is None:
            raise RuntimeError('Wan text encoder is not loaded. Please add OviWanComponentLoader to your workflow.')
        return self.text_model

    def unload(self):
        modules = [
            getattr(self, 'model', None),
            getattr(self, 'vae_model_video', None),
            getattr(self, 'vae_model_audio', None),
            getattr(self, 'text_model', None),
            getattr(self, 'image_model', None),
        ]

        for module in modules:
            if module is None:
                continue
            candidates = [module]
            if hasattr(module, 'model'):
                model_attr = getattr(module, 'model', None)
                if model_attr is not None:
                    candidates.append(model_attr)
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    candidate.to('cpu')
                except Exception:
                    pass

        if hasattr(self, 'model'):
            self.model = None
        if hasattr(self, 'vae_model_video'):
            self.vae_model_video = None
        if hasattr(self, 'vae_model_audio'):
            self.vae_model_audio = None
        if hasattr(self, 'text_model'):
            self.text_model = None
        if hasattr(self, 'image_model'):
            self.image_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def get_attention_backend(self, resolved: bool = False) -> str:
        if resolved:
            return getattr(self, '_resolved_attention_backend', self._requested_attention_backend)
        return self._requested_attention_backend

    @staticmethod
    def available_attention_backends(include_auto: bool = True):
        return available_attention_backends(include_auto=include_auto)

    @torch.inference_mode()
    def generate(self,
                    text_prompt, 
                    image_path=None,
                    video_frame_height_width=None,
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_guidance_scale=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    video_negative_prompt="",
                    audio_negative_prompt=""
                ):

        try:
            self.ensure_loaded()
            resolved_backend = set_attention_backend(self._requested_attention_backend)
            self._resolved_attention_backend = resolved_backend
        except RuntimeError as exc:
            available = ', '.join(available_attention_backends())
            raise RuntimeError(
                f"Failed to select attention backend '{self._requested_attention_backend}': {exc}. Available backends: {available or 'none'}"
            ) from exc

        params = {
            "Text Prompt": text_prompt,
            "Image Path": image_path if image_path else "None (T2V mode)",
            "Frame Height Width": video_frame_height_width,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Video Guidance Scale": video_guidance_scale,
            "Audio Guidance Scale": audio_guidance_scale,
            "Attention Backend": resolved_backend,
            "SLG Layer": slg_layer,
            "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt,
        }

        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")
        try:
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )

            first_frame = None
            is_i2v = False

            if image_path is not None:
                first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype)
                is_i2v = first_frame is not None
            else:
                assert video_frame_height_width is not None, f"If mode=t2v or t2i2v, video_frame_height_width must be provided."
                video_h, video_w = video_frame_height_width
                video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area = 720 * 720)
                video_latent_h, video_latent_w = video_h // 16, video_w // 16
                image_model = getattr(self, 'image_model', None)
                if image_model is not None:
                    image_h, image_w = scale_hw_to_area_divisible(video_h, video_w, area = 1024 * 1024)
                    generated_frame = image_model(
                        clean_text(text_prompt),
                        height=image_h,
                        width=image_w,
                        guidance_scale=4.5,
                        generator=torch.Generator().manual_seed(seed),
                    ).images[0]
                    first_frame = preprocess_image_tensor(generated_frame, self.device, self.target_dtype)
                    is_i2v = first_frame is not None
                else:
                    print(f"Pure T2V mode: calculated video latent size: {video_latent_h} x {video_latent_w}")

            text_model = self._require_text_model()
            previous_device = getattr(text_model, "device", self.device)
            if self.cpu_offload:
                text_model.model = text_model.model.to(self.device)
                text_model.device = self.device
            text_embeddings = text_model([text_prompt, video_negative_prompt, audio_negative_prompt], text_model.device)
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]

            if self.cpu_offload:
                self.offload_to_cpu(text_model.model)
                text_model.device = previous_device

            # Split embeddings
            text_embeddings_audio_pos = text_embeddings[0]
            text_embeddings_video_pos = text_embeddings[0] 

            text_embeddings_video_neg = text_embeddings[1]
            text_embeddings_audio_neg = text_embeddings[2]

            if is_i2v:              
                with torch.no_grad():
                    self._check_cancel()
                    video_vae = self._set_video_vae_device(self.device) if self.cpu_offload else self._require_video_vae()
                    first_frame_tensor = first_frame.to(device=self.device, dtype=torch.bfloat16)
                    latents_images = video_vae.wrapped_encode(first_frame_tensor[:, :, None]).to(self.target_dtype).squeeze(0) # c 1 h w 
                    if self.cpu_offload:
                        self._set_video_vae_device("cpu")
                latents_images = latents_images.to(self.target_dtype)
                video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]

            video_noise = torch.randn((self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # c, f, h, w
            audio_noise = torch.randn((self.audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
            
            # Calculate sequence lengths from actual latents
            max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
            _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
            max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
            
            # Sampling loop
            if self.cpu_offload:
                self.model = self.model.to(self.device)
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
                    self._check_cancel()
                    timestep_input = torch.full((1,), t_v, device=self.device)

                    if is_i2v:
                        video_noise[:, :1] = latents_images

                    # Positive (conditional) forward pass
                    pos_forward_args = {
                        'audio_context': [text_embeddings_audio_pos],
                        'vid_context': [text_embeddings_video_pos],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v
                    }

                    pred_vid_pos, pred_audio_pos = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        **pos_forward_args
                    )
                    
                    # Negative (unconditional) forward pass  
                    neg_forward_args = {
                        'audio_context': [text_embeddings_audio_neg],
                        'vid_context': [text_embeddings_video_neg],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v,
                        'slg_layer': slg_layer
                    }
                    
                    pred_vid_neg, pred_audio_neg = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        **neg_forward_args
                    )

                    # Apply classifier-free guidance
                    pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                    # Update noise using scheduler
                    video_noise = scheduler_video.step(
                        pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    audio_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                if self.cpu_offload:
                    self.offload_to_cpu(self.model)

            if is_i2v:
                video_noise[:, :1] = latents_images

            video_latents = video_noise.detach()
            audio_latents = audio_noise.detach()

            if self.cpu_offload:
                video_latents = video_latents.to("cpu")
                audio_latents = audio_latents.to("cpu")

            return video_latents, audio_latents


        except Exception as e:
            logging.error(traceback.format_exc())
            return None

    @torch.inference_mode()
    def decode_latents(
        self,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        to_cpu: bool = True,
    ):
        if video_latents is None and audio_latents is None:
            raise ValueError("At least one of video_latents or audio_latents must be provided.")

        self.ensure_loaded()

        decoded_video = None
        decoded_audio = None
        video_vae = None

        try:
            if audio_latents is not None:
                if not isinstance(audio_latents, torch.Tensor):
                    raise TypeError("audio_latents must be a torch.Tensor.")
                audio_tensor = audio_latents.to(self.target_dtype)
                if audio_tensor.dim() != 2:
                    raise ValueError("audio_latents must have shape [length, channels].")
                if audio_tensor.device != self.device:
                    audio_tensor = audio_tensor.to(self.device)
                self._check_cancel()
                if self.cpu_offload:
                    self.vae_model_audio = self.vae_model_audio.to(self.device)
                audio_latents_for_vae = audio_tensor.unsqueeze(0).transpose(1, 2)  # 1, c, l
                decoded_audio = (
                    self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                    .squeeze()
                    .to(torch.float32)
                )
                if to_cpu:
                    decoded_audio = decoded_audio.cpu()

            if video_latents is not None:
                if not isinstance(video_latents, torch.Tensor):
                    raise TypeError("video_latents must be a torch.Tensor.")
                video_tensor = video_latents.to(self.target_dtype)
                if video_tensor.dim() != 4:
                    raise ValueError("video_latents must have shape [channels, frames, height, width].")
                if video_tensor.device != self.device:
                    video_tensor = video_tensor.to(self.device)
                self._check_cancel()
                if self.cpu_offload:
                    video_vae = self._set_video_vae_device(self.device)
                else:
                    video_vae = self._require_video_vae()
                decoded_video = video_vae.decode_latents(
                    video_tensor,
                    device=self.device,
                    normalize=True,
                    return_cpu=to_cpu,
                    dtype=torch.float32,
                )

        finally:
            if self.cpu_offload:
                if audio_latents is not None:
                    self.vae_model_audio = self.vae_model_audio.to("cpu")
                if video_latents is not None and video_vae is not None:
                    self._set_video_vae_device("cpu")

        return decoded_video, decoded_audio

    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps

