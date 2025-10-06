import os
import tempfile
from typing import Optional

import numpy as np
from scipy.io import wavfile


def _import_moviepy():
    try:
        from moviepy.editor import ImageSequenceClip, AudioFileClip
    except ImportError as exc:
        raise RuntimeError(
            "moviepy is required to save MP4 files. Install it with `pip install moviepy` (and optionally `pip install imageio-ffmpeg`)."
        ) from exc
    return ImageSequenceClip, AudioFileClip


def save_video(
    output_path: str,
    video_numpy: np.ndarray,
    audio_numpy: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    fps: int = 24,
) -> str:
    """
    Combine a sequence of video frames with an optional audio track and save as an MP4.

    Args:
        output_path (str): Path to the output MP4 file.
        video_numpy (np.ndarray): Numpy array of frames. Shape (C, F, H, W).
                                  Values can be in range [-1, 1] or [0, 255].
        audio_numpy (Optional[np.ndarray]): 1D or 2D numpy array of audio samples, range [-1, 1].
        sample_rate (int): Sample rate of the audio in Hz. Defaults to 16000.
        fps (int): Frames per second for the video. Defaults to 24.

    Returns:
        str: Path to the saved MP4 file.
    """

    ImageSequenceClip, AudioFileClip = _import_moviepy()

    # Validate inputs
    assert isinstance(video_numpy, np.ndarray), "video_numpy must be a numpy array"
    assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
    assert video_numpy.shape[0] in {1, 3}, "video_numpy must have 1 or 3 channels"

    if audio_numpy is not None:
        assert isinstance(audio_numpy, np.ndarray), "audio_numpy must be a numpy array"
        assert np.abs(audio_numpy).max() <= 1.0, "audio_numpy values must be in range [-1, 1]"

    # Reorder dimensions: (C, F, H, W) -> (F, H, W, C)
    video_numpy = video_numpy.transpose(1, 2, 3, 0)

    # Normalize frames if values are in [-1, 1]
    if video_numpy.max() <= 1.0:
        video_numpy = np.clip(video_numpy, -1, 1)
        video_numpy = ((video_numpy + 1) / 2 * 255).astype(np.uint8)
    else:
        video_numpy = video_numpy.astype(np.uint8)

    # Convert numpy array to a list of frames
    frames = list(video_numpy)

    # Create video clip
    clip = ImageSequenceClip(frames, fps=fps)

    temp_audio_path: Optional[str] = None
    audio_clip: Optional[AudioFileClip] = None
    final_clip = clip
    try:
        # Add audio if provided. Using a NamedTemporaryFile with delete=False avoids Windows file locking.
        if audio_numpy is not None:
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            wavfile.write(
                temp_audio_path,
                sample_rate,
                (audio_numpy * 32767).astype(np.int16),
            )
            audio_clip = AudioFileClip(temp_audio_path)
            final_clip = clip.set_audio(audio_clip)

        # Write final video to disk
        final_clip.write_videofile(
            output_path, codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None
        )
    finally:
        if audio_clip is not None:
            audio_clip.close()
        if final_clip is not None and final_clip is not clip:
            final_clip.close()
        clip.close()
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return output_path
