"""
Fast tests for voice_assistant.py's config wiring and WebSocket orchestration.

The module does real work at import time (constructs a WhisperModel), so these tests fake out
faster_whisper.WhisperModel before importing it fresh in each test, via the `load_app` fixture.
That keeps the suite fast and independent of real GPU/CPU hardware, piper, or a running LLM
endpoint. For a real (slow) end-to-end CPU run, see test_integration_cpu.py instead.

Run with: pytest tests/test_voice_assistant.py
"""
import json
import sys
import threading
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
    implies (PIPER_MODEL's actual sample rate isn't verified against PIPER_SAMPLE_RATE)."""
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
        assert wf.getframerate() == va.PIPER_SAMPLE_RATE
        assert wf.readframes(wf.getnframes()) == fake_pcm


def test_synthesize_logs_and_reraises_on_piper_process_error(load_app):
    va = load_app(use_gpu=False)
    import subprocess

    err = subprocess.CalledProcessError(1, ["piper"], output=b"", stderr=b"model not found")
    with patch.object(subprocess, "run", side_effect=err):
        with pytest.raises(subprocess.CalledProcessError):
            va.synthesize("hello")


def test_piper_process_error_surfaces_as_tts_error_over_ws(load_app):
    va = load_app(use_gpu=False)
    import subprocess

    client = TestClient(va.app)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="a reply"))]
    err = subprocess.CalledProcessError(1, ["piper"], output=b"", stderr=b"model not found")
    with patch.object(va.llm_client.chat.completions, "create", return_value=fake_completion), \
         patch.object(subprocess, "run", side_effect=err):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            ws.receive_json()  # transcript
            ws.receive_json()  # reply_text
            error_msg = ws.receive_json()

    assert error_msg == {"type": "error", "message": "text-to-speech failed"}


def test_stt_failure_cleans_up_temp_file(load_app):
    """Regression test: transcribe() used to skip Path(path).unlink() entirely when
    stt_model.transcribe() raised, leaking a temp .webm file per failed transcription."""
    va = load_app(use_gpu=False)
    import glob
    import tempfile as _tempfile

    pattern = str(Path(_tempfile.gettempdir()) / "*.webm")
    before = set(glob.glob(pattern))
    with patch.object(va.stt_model, "transcribe", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            va.transcribe(b"fake-audio")
    after = set(glob.glob(pattern))

    assert after == before


def test_multi_sentence_reply_sends_one_audio_chunk_per_sentence(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="First sentence. Second sentence."))]
    with patch.object(va.llm_client.chat.completions, "create", return_value=fake_completion), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            msgs = [ws.receive_json() for _ in range(4)]  # transcript, reply_text, 2x reply_audio

    audio_msgs = [m for m in msgs if m["type"] == "reply_audio"]
    assert len(audio_msgs) == 2
    assert [m["final"] for m in audio_msgs] == [False, True]


def test_index_requires_token_when_auth_configured(load_app):
    va = load_app(use_gpu=False)
    va.AUTH_TOKEN = "secret"
    client = TestClient(va.app)

    assert client.get("/").status_code == 401
    assert client.get("/?token=wrong").status_code == 401
    assert client.get("/?token=secret").status_code == 200


def test_ws_rejects_missing_or_wrong_token_when_auth_configured(load_app):
    va = load_app(use_gpu=False)
    va.AUTH_TOKEN = "secret"
    client = TestClient(va.app)

    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=wrong"):
            pass


def test_ws_accepts_correct_token_when_auth_configured(load_app):
    va = load_app(use_gpu=False)
    va.AUTH_TOKEN = "secret"
    client = TestClient(va.app)

    with client.websocket_connect("/ws?token=secret") as ws:
        ws.send_bytes(b"fake-audio")
        msg = ws.receive_json()

    assert msg == {"type": "transcript", "text": "hello world"}


# ---- conversation memory ----

def test_conversation_history_is_sent_back_to_the_llm(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    calls = []

    def fake_create(model, messages):
        calls.append(messages)
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content=f"reply {len(calls)}"))]
        return completion

    with patch.object(va.llm_client.chat.completions, "create", side_effect=fake_create), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            [ws.receive_json() for _ in range(3)]  # transcript, reply_text, reply_audio
            ws.send_bytes(b"fake-audio")
            [ws.receive_json() for _ in range(3)]

    assert calls[0] == [
        {"role": "system", "content": va.PERSONAS[va.DEFAULT_PERSONA]},
        {"role": "user", "content": "hello world"},
    ]
    assert calls[1] == [
        {"role": "system", "content": va.PERSONAS[va.DEFAULT_PERSONA]},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "hello world"},
    ]


# ---- text input fallback ----

def test_text_input_skips_stt_and_reaches_the_llm(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="a reply"))]
    with patch.object(va.llm_client.chat.completions, "create", return_value=fake_completion), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "), \
         patch.object(va, "transcribe") as mock_transcribe:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "text_input", "text": "typed question"}))
            transcript_msg = ws.receive_json()
            reply_msg = ws.receive_json()

    mock_transcribe.assert_not_called()
    assert transcript_msg == {"type": "transcript", "text": "typed question"}
    assert reply_msg == {"type": "reply_text", "text": "a reply"}


def test_blank_text_input_is_ignored(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="a reply"))]
    with patch.object(va.llm_client.chat.completions, "create", return_value=fake_completion), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "text_input", "text": "   "}))
            ws.send_bytes(b"fake-audio")
            msg = ws.receive_json()  # the blank text_input produced nothing; this is the audio turn

    assert msg == {"type": "transcript", "text": "hello world"}


# ---- persona switching ----

def test_set_persona_changes_system_prompt_and_resets_history(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    calls = []

    def fake_create(model, messages):
        calls.append(messages)
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="arr"))]
        return completion

    with patch.object(va.llm_client.chat.completions, "create", side_effect=fake_create), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "set_persona", "persona": "pirate"}))
            ack = ws.receive_json()
            ws.send_bytes(b"fake-audio")
            [ws.receive_json() for _ in range(3)]

    assert ack == {"type": "persona_set", "persona": "pirate"}
    assert calls[0][0] == {"role": "system", "content": va.PERSONAS["pirate"]}


def test_unknown_persona_is_rejected(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "set_persona", "persona": "nonexistent"}))
        ws.send_bytes(b"fake-audio")
        msg = ws.receive_json()  # the rejected persona switch produced nothing; this is the audio turn

    assert msg == {"type": "transcript", "text": "hello world"}


# ---- rate limiting ----

def test_rate_limit_blocks_excess_turns(load_app):
    va = load_app(use_gpu=False)
    va.RATE_LIMIT_TURNS_PER_MINUTE = 1
    client = TestClient(va.app)
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="a reply"))]
    with patch.object(va.llm_client.chat.completions, "create", return_value=fake_completion), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            [ws.receive_json() for _ in range(3)]  # transcript, reply_text, reply_audio
            ws.send_bytes(b"fake-audio")
            second_msg = ws.receive_json()

    assert second_msg == {"type": "error", "message": "rate limit exceeded — slow down"}


# ---- interrupt / barge-in ----

def test_new_utterance_interrupts_in_flight_turn(load_app):
    va = load_app(use_gpu=False)
    client = TestClient(va.app)
    release = threading.Event()

    def slow_create(model, messages):
        release.wait(timeout=5)
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="reply"))]
        return completion

    with patch.object(va.llm_client.chat.completions, "create", side_effect=slow_create), \
         patch.object(va, "synthesize", return_value=b"RIFFxxxxWAVEfmt "):
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(b"fake-audio")
            first_transcript = ws.receive_json()  # first turn now blocked inside query_llm

            ws.send_bytes(b"fake-audio")
            interrupted_msg = ws.receive_json()
            release.set()  # let the (now cancelled) first LLM call's background thread finish
            second_transcript = ws.receive_json()

    assert first_transcript == {"type": "transcript", "text": "hello world"}
    assert interrupted_msg == {"type": "interrupted"}
    assert second_transcript == {"type": "transcript", "text": "hello world"}
