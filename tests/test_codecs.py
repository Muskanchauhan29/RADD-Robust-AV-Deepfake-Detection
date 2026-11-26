import json
import subprocess
import numpy as np
import pytest
from views import codecs as C

pytestmark = pytest.mark.skipif(not C.have_ffmpeg(), reason="FFmpeg not installed")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "clip.mp4"
    C._run(["-f", "lavfi", "-i", "testsrc2=size=130x98:rate=25:duration=2",   # odd-ish size on purpose
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)])
    return p


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    p = tmp_path_factory.mktemp("a") / "clip.wav"
    t = np.arange(2 * C.SR) / C.SR
    rng = np.random.default_rng(0)
    x = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1700 * t) + 0.02 * rng.standard_normal(len(t))
    C.write_wav(p, x.astype(np.float32))
    return p


@pytest.mark.parametrize("fn", [C.h264, C.h265])
def test_video_codec_keeps_50_frames_25fps(video, tmp_path, fn):
    out = tmp_path / "o.mp4"; fn(video, out, 32)
    s = probe(out)
    assert s["nb_read_frames"] == "50" and s["r_frame_rate"] == "25/1"
    assert int(s["width"]) % 2 == 0 and int(s["height"]) % 2 == 0


def test_higher_crf_gives_smaller_file(video, tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    C.h264(video, a, 23); C.h264(video, b, 40)
    assert b.stat().st_size < a.stat().st_size


@pytest.mark.parametrize("fn", [C.mp3, C.aac])
def test_audio_codec_keeps_length_and_16k_and_resembles_input(wav, tmp_path, fn):
    out = tmp_path / "o.wav"; fn(wav, out, 64)
    x, y = C.read_wav(wav), C.read_wav(out)
    assert len(y) == len(x)
    assert not np.allclose(x, y)
    assert np.corrcoef(x, y)[0, 1] > 0.8            # same content, just coded


def test_chain_applies_every_step_and_rejects_empty(video, tmp_path):
    out = tmp_path / "c.mp4"
    C.reencode_chain(video, out, [("h264", 28), ("h265", 32)])
    assert probe(out)["nb_read_frames"] == "50"
    with pytest.raises(ValueError):
        C.reencode_chain(video, out, [])