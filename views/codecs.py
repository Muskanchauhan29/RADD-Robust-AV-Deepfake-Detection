"""Step 2b-ii: codec degradations through FFmpeg (MP3, AAC, H.264, H.265, chains).

Needs FFmpeg with libx264, libx265 and libmp3lame on the PATH. Tests skip themselves if it is missing.
Bitrates and CRF values are parameters; only CRF 23-40 (training) is from the paper.
The exact re-encoding chain used for held-out testing is undecided (D-14): no default is provided.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 16000


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd):
    if not have_ffmpeg():
        raise RuntimeError("FFmpeg not found on PATH; install it first (e.g. `brew install ffmpeg`).")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()}")


# ------------------------------------------------------------ audio
def read_wav(path) -> np.ndarray:
    sr, x = wavfile.read(str(path))
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    return x.astype(np.float32)


def write_wav(path, wav: np.ndarray) -> None:
    wavfile.write(str(path), SR, (np.clip(wav, -1, 1) * 32767).astype(np.int16))


def _fit(wav: np.ndarray, n: int) -> np.ndarray:
    """Codecs add padding/delay; trim or zero-pad back to n samples."""
    return wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))


def _audio_roundtrip(wav_in, wav_out, codec: str, ext: str, bitrate_k: int) -> None:
    n = len(read_wav(wav_in))
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / f"tmp.{ext}"
        _run(["-i", str(wav_in), "-c:a", codec, "-b:a", f"{bitrate_k}k", str(tmp)])
        _run(["-i", str(tmp), "-ar", str(SR), "-ac", "1", str(Path(d) / "dec.wav")])
        write_wav(wav_out, _fit(read_wav(Path(d) / "dec.wav"), n))


def mp3(wav_in, wav_out, bitrate_k: int) -> None:          # training family
    _audio_roundtrip(wav_in, wav_out, "libmp3lame", "mp3", bitrate_k)


def aac(wav_in, wav_out, bitrate_k: int) -> None:          # held-out family
    _audio_roundtrip(wav_in, wav_out, "aac", "m4a", bitrate_k)


# ------------------------------------------------------------ video
_EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"                # yuv420p needs even width/height


def _video(video_in, video_out, lib: str, crf: int) -> None:
    _run(["-i", str(video_in), "-an", "-vf", _EVEN, "-c:v", lib, "-crf", str(crf),
          "-preset", "veryfast", "-pix_fmt", "yuv420p", str(video_out)])


def h264(video_in, video_out, crf: int) -> None:            # training family, CRF 23-40
    _video(video_in, video_out, "libx264", crf)


def h265(video_in, video_out, crf: int) -> None:            # held-out family
    _video(video_in, video_out, "libx265", crf)


def reencode_chain(video_in, video_out, steps) -> None:
    """Apply several video encodes in a row. steps = [("h264", 28), ("h265", 32), ...].
    The held-out chain definition is open (D-14), so there is no default."""
    fn = {"h264": h264, "h265": h265}
    if not steps:
        raise ValueError("steps must not be empty")
    with tempfile.TemporaryDirectory() as d:
        cur = video_in
        for i, (name, crf) in enumerate(steps):
            nxt = video_out if i == len(steps) - 1 else Path(d) / f"s{i}.mp4"
            fn[name](cur, nxt, crf)
            cur = nxt