# Architecture

Everything — server and frontend — lives in one file, `voice_assistant.py`. FastAPI serves the
page at `GET /` and handles the voice loop over a WebSocket at `/ws`.

## Request flow

```
Browser                          FastAPI (voice_assistant.py)
────────                         ─────────────────────────────
mic → MediaRecorder
  (webm blob per utterance)
        │
        │  WS: binary audio blob
        ▼
                                  transcribe()      faster-whisper STT
                                        │            (CPU or GPU, per USE_GPU)
                                        ▼
        ◄─────────────────────  {"type": "transcript", "text": ...}
                                        │
                                  query_llm()        OpenAI-compatible chat
                                        │            completion via llama-swap
                                        ▼
        ◄─────────────────────  {"type": "reply_text", "text": ...}
                                        │
                                  synthesize()       piper subprocess → WAV
                                        │
        ◄─────────────────────  {"type": "reply_audio", "audio_b64": ...}
        │
  <audio> playback
```

If `transcribe()` returns empty text (e.g. background noise with no speech), the server sends
`{"type": "error", "message": "no speech detected"}` and waits for the next utterance instead of
calling the LLM.

## WebSocket message protocol

**Client → server:** a single binary WebM/Opus blob per utterance (one message per turn, not a
continuous stream).

**Server → client:** JSON text messages, in this order per turn:

| Type | Fields | Meaning |
|---|---|---|
| `transcript` | `text` | What Whisper heard. |
| `reply_text` | `text` | The LLM's reply, before speech synthesis. |
| `reply_audio` | `audio_b64` | Base64-encoded WAV of the spoken reply. |
| `error` | `message` | Something recoverable went wrong (e.g. no speech detected); the turn ends without a reply. |

## Frontend

The page returned by `GET /` is a single self-contained HTML document (inline `<style>` and
`<script>`, no build step, no external assets):

- **Recording** — `MediaRecorder` captures mic audio as WebM/Opus. In push-to-talk mode, recording
  starts/stops with the talk button (mouse, touch, and keyboard). In continuous mode, a small
  client-side VAD (RMS energy over a `Web Audio` `AnalyserNode`) starts recording when it detects
  voice and stops it after a configurable silence gap (`SILENCE_MS`).
- **Status visualization** — a `<canvas>` element animates a small set of glowing nodes connected
  by lines when close together; state (idle/listening/thinking/speaking) changes their motion
  speed, glow intensity, and size rather than color, since the UI is intentionally monochrome.
- **Status/error handling** — a single `setStatus(text, isError)` helper drives the status text
  (`role="status" aria-live="polite"`, so screen readers announce changes), with a distinct
  visual treatment for error states. Mic permission failures and unsupported-browser cases are
  caught explicitly rather than failing silently.
- **State machine** — `mode` ("push" or "continuous") and `busy` (waiting on a server round-trip)
  gate what the recorder and VAD loop are allowed to do at any moment, so a new recording can't
  start while a reply is still being synthesized or played back.

## Server internals

Three synchronous, blocking functions do the actual work and are each run in the default
executor (`loop.run_in_executor`) so they don't block the asyncio event loop while the model or
subprocess runs:

- `transcribe(audio_bytes)` — writes the blob to a temp file and runs it through the
  `faster_whisper.WhisperModel` loaded at startup.
- `query_llm(text)` — a single non-streaming chat completion call against `LLAMA_SWAP_URL`.
- `synthesize(text)` — pipes text into the `piper` binary as a subprocess and wraps its raw PCM
  output in a WAV container.

The Whisper model and the OpenAI client are constructed once at import time (`stt_model`,
`llm_client`), not per-request.
