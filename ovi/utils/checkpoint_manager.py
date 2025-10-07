"""Utilities to verify and download OVI checkpoints on demand."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import folder_paths
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError as exc:  # pragma: no cover - handled at runtime
    HfApi = None
    hf_hub_download = None
    HF_IMPORT_ERROR = exc
else:
    HF_IMPORT_ERROR = None

def _find_comfy_root():
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        models_dir = parent / 'models'
        if models_dir.is_dir():
            return parent
    # fallback to repo root two levels up (custom_nodes/ComfyUI-Ovi)
    return Path(__file__).resolve().parents[3]

try:
    _DIFFUSION_MODEL_DIRS = [Path(p) for p in folder_paths.get_folder_paths('diffusion_models')]
except Exception: # pragma: no cover
    _DIFFUSION_MODEL_DIRS = []

def _select_diffusion_dir():
    for candidate in _DIFFUSION_MODEL_DIRS:
        if 'diffusion' in candidate.name.lower() or 'diffusion_models' in candidate.as_posix():
            return candidate
    return _DIFFUSION_MODEL_DIRS[0] if _DIFFUSION_MODEL_DIRS else (COMFY_ROOT / 'models' / 'diffusion_models')

COMFY_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'models').is_dir()), Path(__file__).resolve().parents[3])
DIFFUSION_MODELS_DIR = _select_diffusion_dir().resolve()
OVI_MODEL_REPO = 'chetwinlow1/Ovi'
OVI_MODEL_SOURCE_NAME = 'model.safetensors'
OVI_MODEL_TARGET_NAME = 'Ovi-11B-bf16.safetensors'
OVI_MODEL_FP8_REPO = 'rkfg/Ovi-fp8_quantized'
OVI_MODEL_FP8_SOURCE_NAME = 'model_fp8_e4m3fn.safetensors'
OVI_MODEL_FP8_TARGET_NAME = 'Ovi-11B-fp8.safetensors'

class MissingDependencyError(RuntimeError):
    """Raised when optional runtime dependencies are absent."""


class DownloadError(RuntimeError):
    """Raised when downloads fail."""


_BASE_DOWNLOAD_JOBS = (
    {
        "repo_id": "Wan-AI/Wan2.2-TI2V-5B",
        "subdir": "",
        "patterns": [
            "google/umt5-xxl/*",
        ],
    },
    {
        "repo_id": "hkchengrex/MMAudio",
        "subdir": "MMAudio",
        "patterns": [
            "ext_weights/best_netG.pt",
            "ext_weights/v1-16.pth",
        ],
    },
)

_REPO_FILE_CACHE: dict[str, List[str]] = {}

_OVI_VARIANT_JOBS = {
    "bf16": {
        "repo_id": OVI_MODEL_REPO,
        "subdir": "Ovi",
        "patterns": [OVI_MODEL_SOURCE_NAME],
        "target_name": OVI_MODEL_TARGET_NAME,
        "source_name": OVI_MODEL_SOURCE_NAME,
    },
    "fp8": {
        "repo_id": OVI_MODEL_FP8_REPO,
        "subdir": "Ovi",
        "patterns": [OVI_MODEL_FP8_SOURCE_NAME],
        "target_name": OVI_MODEL_FP8_TARGET_NAME,
        "source_name": OVI_MODEL_FP8_SOURCE_NAME,
    },
}


def _check_huggingface_cli():
    """Check if hf CLI is available."""
    try:
        result = subprocess.run(['hf', '--help'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _list_repo_files(repo_id: str) -> List[str]:
    if repo_id in _REPO_FILE_CACHE:
        return _REPO_FILE_CACHE[repo_id]
    if HfApi is None:
        raise MissingDependencyError(
            "huggingface_hub is required for auto-downloads"
        ) from HF_IMPORT_ERROR
    api = HfApi()
    files = api.list_repo_files(repo_id)
    _REPO_FILE_CACHE[repo_id] = files
    return files


def _expand_patterns(repo_id: str, patterns: Iterable[str]) -> List[str]:
    files = _list_repo_files(repo_id)
    expanded: list[str] = []
    for pattern in patterns:
        pattern = pattern.replace("\\", "/").strip()
        if not pattern:
            continue
        if any(ch in pattern for ch in "*?["):
            matches = [f for f in files if fnmatch(f, pattern)]
            if not matches:
                logging.warning("Pattern %s did not match any files in %s", pattern, repo_id)
            expanded.extend(matches)
        else:
            expanded.append(pattern)
    seen = set()
    result = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _missing_files(base_dir: Path, relative_files: Iterable[str]) -> list[Path]:
    missing = []
    for rel in relative_files:
        path = base_dir / rel
        if not path.exists():
            missing.append(path)
    return missing


def _missing_patterns_locally(base_dir: Path, patterns: Iterable[str]) -> list[Path]:
    """Check whether patterns already exist locally without hitting remote APIs."""
    missing: list[Path] = []
    for pattern in patterns:
        normalized = pattern.replace("\\", "/").strip()
        if not normalized:
            continue
        has_wildcard = any(ch in normalized for ch in "*?[")
        if has_wildcard:
            matches = list(base_dir.glob(normalized))
            if matches:
                continue
            missing.append(base_dir / normalized)
        else:
            path = base_dir / normalized
            if not path.exists():
                missing.append(path)
    return missing


def _download_file(repo_id: str, relative_path: str, target_dir: Path) -> None:
    """Download file using hf CLI with progress bars, fallback to hf_hub_download."""
    # Try hf CLI first for progress bars
    if _check_huggingface_cli():
        logging.info("    Fetching %s (with progress bar)", relative_path)
        
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Build hf download command
            cmd = [
                'hf', 'download',
                repo_id,
                relative_path,
                '--local-dir', str(target_dir)
            ]
            
            # Run the command and show progress
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            
            if result.returncode != 0:
                raise DownloadError(f"hf CLI failed with return code {result.returncode}")
                
            return  # Success, exit early
            
        except subprocess.CalledProcessError as exc:
            logging.warning("hf CLI failed, falling back to hf_hub_download: %s", exc)
        except FileNotFoundError:
            logging.warning("hf CLI not found during execution, falling back to hf_hub_download")
        except Exception as exc:
            logging.warning("Error using hf CLI, falling back to hf_hub_download: %s", exc)
    
    # Fallback to original method
    if hf_hub_download is None:
        raise MissingDependencyError(
            "huggingface_hub is required for auto-downloads"
        ) from HF_IMPORT_ERROR

    logging.info("    Fetching %s", relative_path)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=relative_path,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=False,
        )
    except Exception as exc:  # pragma: no cover
        raise DownloadError(
            f"Failed downloading {relative_path} from {repo_id}: {exc}"
        ) from exc


def ensure_checkpoints(
    ckpt_dir: str | os.PathLike[str],
    download: bool = True,
    variants: Iterable[str] = ("bf16",),
) -> None:
    """Ensure required checkpoints exist, downloading from HF if permitted."""
    if isinstance(variants, str):
        variants = (variants,)
    normalized_variants: list[str] = []
    for variant in variants:
        key = variant.lower()
        if key not in _OVI_VARIANT_JOBS:
            raise ValueError(f"Unknown OVI model variant '{variant}'. Expected one of {sorted(_OVI_VARIANT_JOBS)}.")
        if key not in normalized_variants:
            normalized_variants.append(key)

    jobs = list(_BASE_DOWNLOAD_JOBS)
    jobs.extend(_OVI_VARIANT_JOBS[key] for key in normalized_variants)

    ckpt_path = Path(ckpt_dir).expanduser().resolve()
    ckpt_path.mkdir(parents=True, exist_ok=True)

    missing_any = False
    for job in jobs:
        target_dir = ckpt_path / job["subdir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        target_name = job.get("target_name")
        source_name = job.get("source_name")

        if target_name and source_name:
            dest_file = DIFFUSION_MODELS_DIR / target_name
            source_file = target_dir / source_name
            if dest_file.exists():
                continue
            if source_file.exists():
                DIFFUSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    if dest_file.exists():
                        dest_file.unlink()
                except OSError:
                    pass
                shutil.move(str(source_file), str(dest_file))
                continue

        local_missing = _missing_patterns_locally(target_dir, job["patterns"])

        if not local_missing and not (target_name and source_name):
            continue

        remote_listing_failed = False
        try:
            required_paths = _expand_patterns(job["repo_id"], job["patterns"])
        except Exception as exc:  # pragma: no cover - network/mirror failures
            logging.warning(
                "Failed to list files for %s via HuggingFace API: %s. Falling back to local file check.",
                job["repo_id"],
                exc,
            )
            required_paths = list(job["patterns"])
            remote_listing_failed = True

        if not required_paths:
            logging.warning("No matching files resolved for %s", job["repo_id"])
            continue

        if target_name and source_name:
            dest_path = DIFFUSION_MODELS_DIR / target_name
            missing = [] if dest_path.exists() else [target_dir / source_name]
        elif remote_listing_failed:
            missing = _missing_patterns_locally(target_dir, required_paths)
        else:
            missing = _missing_files(target_dir, required_paths)

        if missing:
            missing_any = True
            if not download:
                raise FileNotFoundError(
                    f"Missing required checkpoint files: {[str(p.relative_to(ckpt_path)) for p in missing]}"
                )
            if remote_listing_failed:
                raise DownloadError(
                    f"Unable to reach Hugging Face to download {job['repo_id']} and required files are missing: "
                    f"{[str(p.relative_to(ckpt_path)) for p in missing]}"
                )

            logging.info(
                "Downloading %s to %s (missing %d files)",
                job["repo_id"],
                target_dir,
                len(missing),
            )
            for missing_path in missing:
                missing_path.parent.mkdir(parents=True, exist_ok=True)
                rel_path = missing_path.relative_to(target_dir).as_posix()
                _download_file(job["repo_id"], rel_path, target_dir)

            if target_name and source_name:
                DIFFUSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
                source_file = target_dir / source_name
                dest_file = DIFFUSION_MODELS_DIR / target_name
                if source_file.exists():
                    try:
                        if dest_file.exists():
                            dest_file.unlink()
                    except OSError:
                        pass
                    shutil.move(str(source_file), str(dest_file))
                else:
                    logging.warning('Expected Ovi model file at %s but it was not downloaded.', source_file)

    if missing_any:
        logging.info("Checkpoint verification completed for %s", ckpt_path)
