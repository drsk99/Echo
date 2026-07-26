"""
Fast tests for voice_assistant.py's config wiring and WebSocket orchestration.

The module does real work at import time (constructs a WhisperModel), so these tests fake out
faster_whisper.WhisperModel before importing it fresh in each test, via the `load_app` fixture.
That keeps the suite fast and independent of real GPU/CPU hardware, piper, or a running LLM
endpoint. For a real (slow) end-to-end CPU run, see test_integration_cpu.py instead.

Run with: pytest tests/test_voice_assistant.py
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

VOICE_ASSISTANT_PATH = Path(__file__).parent.parent / "voice_assistant.py"


class _FakeSegment:
    def __init__(self, text):
        self.text = text
        self.no_speech_prob = 0.1
        self.avg_logprob = -0.2


class _FakeWhisperModel:
    """Records how it was constructed and returns a canned transcript — no real hardware needed."""

    last_init_kwargs = {}

    def __init__(self, model_size_or_path, device=None, compute_type=None, **kwargs):
        _FakeWhisperModel.last_init_kwargs = {
            "model_size": model_size_or_path,
            "device": device,
            "compute_type": compute_type,
        }
        self.next_segments = [_FakeSegment("hello world")]

    def transcribe(self, path, language=None, vad_filter=False):
        return self.next_segments, None


@pytest.fixture
def load_app(monkeypatch):
    """Imports voice_assistant fresh, with WhisperModel faked out, for a given USE_GPU value."""

    def _load(use_gpu: bool):
        fake_module = types.ModuleType("faster_whisper")
        fake_module.WhisperModel = _FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

        src = VOICE_ASSISTANT_PATH.read_text()
        current = "USE_GPU = True" if "USE_GPU = True" in src else "USE_GPU = False"
        patched_src = src.replace(current, f"USE_GPU = {use_gpu}", 1)

        module = types.ModuleType("voice_assistant_under_test")
        module.__file__ = str(VOICE_ASSISTANT_PATH)
        exec(compile(patched_src, str(VOICE_ASSISTANT_PATH), "exec"), module.__dict__)
        return module

    return _load


def test_gpu_profile_wiring(load_app):
    va = load_app(use_gpu=True)
    assert (va.WHISPER_MODEL_SIZE, va.WHISPER_DEVICE, va.WHISPER_COMPUTE_TYPE) == ("medium", "cuda", "float16")
    assert _FakeWhisperModel.last_init_kwargs == {
        "model_size": "medium", "device": "cuda", "compute_type": "float16",
    }


def test_cpu_profile_wiring(load_app):
    va = load_app(use_gpu=False)
    assert (va.WHISPER_MODEL_SIZE, va.WHISPER_DEVICE, va.WHISPER_COMPUTE_TYPE) == ("small", "cpu", "int8")
    assert _FakeWhisperModel.last_init_kwargs == {
        "model_size": "small", "device": "cpu", "compute_type": "int8",
    }


def test_no_speech_detected(load_app):
    va = load_app(use_gpu=False)
    va.stt_model.next_segments = []  # what vad_filter=True returns for silence / non-speech
    client = TestClient(va.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"fake-audio")
        msg = ws.receive_json()
    assert msg == {"type": "error", "message": "no speech detected"}


def test_llm_outage_sends_error_and_keeps_connection_alive(load_app):
    """Regression test: an unhandled exception from query_llm() used to propagate out of
    ws_endpoint entirely, killing the connection with no message ever reaching the client."""
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    with patch.object(va.llm_client.chat.completions, "create", side_effect=RuntimeError("connection refused")):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            transcript_msg = ws.receive_json()
            error_msg = ws.receive_json()

            # the connection must still work for the next turn
            ws.send_bytes(b"fake-audio")
            second_transcript_msg = ws.receive_json()

    assert transcript_msg == {"type": "transcript", "text": "hello world"}
    assert error_msg["type"] == "error" and "language model" in error_msg["message"]
    assert second_transcript_msg["type"] == "transcript"


def test_tts_failure_sends_error(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="a reply"))]
    with patch.object(va.llm_client.chat.completions, "create", return_value=fake_completion), \
         patch.object(va, "synthesize", side_effect=FileNotFoundError("no piper binary")):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            transcript_msg = ws.receive_json()
            reply_msg = ws.receive_json()
            error_msg = ws.receive_json()

    assert reply_msg == {"type": "reply_text", "text": "a reply"}
    assert error_msg["type"] == "error" and "text-to-speech" in error_msg["message"]


def test_happy_path(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="a reply"))]
    with patch.object(va.llm_client.chat.completions, "create", return_value=fake_completion), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            msgs = [ws.receive_json() for _ in range(3)]

    assert [m["type"] for m in msgs] == ["transcript", "reply_text", "reply_audio"]


def test_synthesize_wraps_pcm_in_valid_wav(load_app):
    """Validates the WAV-wrapping logic itself; doesn't check against real piper output
    (unavailable in this environment) — see docs/SETUP.md for the sample-rate caveat that
    implies (PIPER_MODEL's actual sample rate isn't verified against the hardcoded 22050)."""
    va = load_app(use_gpu=False)
    import io
    import subprocess
    import wave

    fake_pcm = b"\x00\x01" * 100
    with patch.object(subprocess, "run", return_value=MagicMock(stdout=fake_pcm)):
        wav_bytes = va.synthesize("hello")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.readframes(wf.getnframes()) == fake_pcm
