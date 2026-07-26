"""
Real, slow integration test of the CPU Whisper path — no mocking of faster_whisper itself.

Downloads and loads the actual WHISPER_CPU_MODEL_SIZE model on first run (small, ~500MB, cached
by huggingface_hub afterward) and runs real CPU inference. This is the test that caught the
original bug: without `vad_filter=True`, Whisper routinely hallucinates a short word on silence
instead of returning empty text.

Requires network access on first run and takes significantly longer than test_voice_assistant.py.
Not run by default — opt in explicitly:

    pytest tests/test_integration_cpu.py -m slow
"""
import io
import math
import sys
import types
from pathlib import Path

import pytest

av = pytest.importorskip("av", reason="pip install av numpy to run this test")
np = pytest.importorskip("numpy", reason="pip install av numpy to run this test")

VOICE_ASSISTANT_PATH = Path(__file__).parent.parent / "voice_assistant.py"

pytestmark = pytest.mark.slow


def _make_webm_opus(samples, sample_rate=48000):
    """Encodes float32 PCM to an in-memory webm/opus blob, like a browser's MediaRecorder does."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="webm")
    stream = container.add_stream("libopus", rate=sample_rate)
    frame = av.AudioFrame.from_ndarray(np.array([samples], dtype=np.float32), format="fltp", layout="mono")
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue()


def _silence(duration_s, sample_rate=48000):
    return [0.0] * int(duration_s * sample_rate)


@pytest.fixture(scope="module")
def real_cpu_app():
    src = VOICE_ASSISTANT_PATH.read_text()
    current = "USE_GPU = True" if "USE_GPU = True" in src else "USE_GPU = False"
    patched_src = src.replace(current, "USE_GPU = False", 1)

    module = types.ModuleType("voice_assistant_real_cpu")
    module.__file__ = str(VOICE_ASSISTANT_PATH)
    exec(compile(patched_src, str(VOICE_ASSISTANT_PATH), "exec"), module.__dict__)
    return module


def test_real_cpu_model_loads_with_configured_profile(real_cpu_app):
    assert real_cpu_app.WHISPER_DEVICE == "cpu"
    assert real_cpu_app.stt_model is not None


def test_real_silence_does_not_hallucinate_text(real_cpu_app):
    blob = _make_webm_opus(_silence(1.5))
    text = real_cpu_app.transcribe(blob)
    assert text == "", f"expected empty transcript for silence, got {text!r} (vad_filter regression?)"
