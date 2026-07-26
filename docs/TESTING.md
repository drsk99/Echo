# Testing

`voice_assistant.py` does real work at import time — it loads a Whisper model and constructs an
OpenAI client — so the test suite fakes out `faster_whisper.WhisperModel` before importing the
module fresh in each test (see the `load_app` fixture in `tests/test_voice_assistant.py`). That
keeps the suite fast and independent of real GPU/CPU hardware, a piper install, or a running LLM
endpoint.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run the fast suite

```bash
pytest tests/
```

Covers, all with `WhisperModel` faked out:

- **Config wiring** — `USE_GPU = True`/`False` each produce the correct `(model_size, device,
  compute_type)` triple passed into `WhisperModel(...)`.
- **No speech detected** — an empty transcript (what `vad_filter=True` produces for silence)
  sends `{"type": "error", "message": "no speech detected"}` and nothing else.
- **LLM outage** — if the LLM endpoint is unreachable, the client gets a
  `{"type": "error", ...}` message naming the failure, *and the WebSocket connection stays open
  for the next turn*. This is a regression test: earlier, an unhandled exception from the LLM
  call propagated out of the handler and silently killed the connection, leaving the browser
  stuck on "processing…" with no explanation.
- **TTS failure** — same pattern, for a missing/failing piper binary.
- **Happy path** — a full turn with the LLM and piper mocked to succeed produces `transcript` →
  `reply_text` → `reply_audio` in order.
- **WAV wrapping** — `synthesize()`'s raw-PCM-to-WAV framing is structurally correct.
- **Piper process errors** — a `CalledProcessError` from piper (binary ran, exited non-zero) is
  logged with its stderr and surfaces to the client as `{"type": "error", "message": "text-to-speech failed"}`.
- **STT temp-file cleanup** — a failed `stt_model.transcribe()` call still removes the temp
  `.webm` file (regression test for a leak).
- **Multi-sentence replies** — a reply with more than one sentence produces one `reply_audio`
  message per sentence, with `final: true` only on the last.
- **Auth** — `AUTH_TOKEN` set: `GET /` and `/ws` reject a missing/wrong `?token=`, accept the
  correct one.
- **Conversation memory** — a second turn's LLM call includes the first turn's user/assistant
  messages plus the active persona's system prompt, per `ConnectionState.build_messages()`.
- **Text input** — a `{"type": "text_input", ...}` message skips `transcribe()` entirely and goes
  straight to the LLM; blank text is ignored.
- **Personas** — `set_persona` swaps the system prompt sent to the LLM and resets history; an
  unknown persona key is ignored.
- **Rate limiting** — exceeding `RATE_LIMIT_TURNS_PER_MINUTE` on one connection returns
  `{"type": "error", "message": "rate limit exceeded — slow down"}` instead of processing the turn.
- **Barge-in** — a new utterance arriving while a turn is still in flight cancels that turn,
  sends `{"type": "interrupted"}`, and processes the new one.

## Run the real (slow) CPU integration test

```bash
pytest tests/test_integration_cpu.py -m slow
```

This one does *not* fake `WhisperModel` — it downloads and loads the actual
`WHISPER_CPU_MODEL_SIZE` model (small, ~500MB, cached by `huggingface_hub` after the first run)
and runs real CPU inference on synthetic audio. It's what caught the original `vad_filter` bug:
without it, Whisper routinely hallucinates a short word ("You", "Thank you") on silence instead
of returning empty text, which meant near-silent clips were getting sent to the LLM as if they
were real speech. Requires network access on first run; deselected by default (see `pytest.ini`)
since it's much slower than the mocked suite.

There's no equivalent real-GPU integration test — this environment doesn't have CUDA hardware.
The GPU profile is covered by `test_gpu_profile_wiring`, which faked `WhisperModel` to confirm
`USE_GPU = True` wires `device="cuda", compute_type="float16", model="medium"` correctly; it does
not (and can't, without a GPU) verify that ctranslate2 actually runs on one.

## Known gaps testing didn't cover

- **Piper's actual sample rate.** `synthesize()` wraps piper's raw PCM output at
  `PIPER_SAMPLE_RATE` (configurable, default `22050`). Most piper voices are 22050 Hz, but it
  varies by voice model (check the voice's `.onnx.json` for its `sample_rate`) — a mismatched
  value won't error, it'll just play back at the wrong pitch/speed. No piper binary is available
  in this environment to verify against real output; if you hear pitched audio, check this first.
- **Whisper transcription accuracy on real speech.** All tests here use synthetic (silent or
  tone) audio — there's no real speech sample in this environment to validate transcription
  quality against, only that the pipeline mechanically works.
