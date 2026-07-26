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

The reply is split into sentences (`split_sentences()`) and synthesized/sent one sentence at a
time, rather than as a single `reply_audio` message for the whole reply. This lets the client
start playing the first sentence while piper is still synthesizing later ones, instead of waiting
on the full reply before any audio arrives.

## WebSocket message protocol

**Client → server:**

| Message | Meaning |
|---|---|
| binary WebM/Opus blob | One utterance, transcribed via `transcribe()`. |
| `{"type": "text_input", "text": "..."}` | Typed input — skips STT, goes straight to the LLM with the same conversation history and persona as a spoken turn. |
| `{"type": "set_persona", "persona": "..."}` | Switches the system prompt to `PERSONAS[persona]` and clears conversation history (a persona starts with a clean slate). Unknown keys are ignored. |

If `AUTH_TOKEN` is set, the connection URL must include `?token=<value>` or the server closes it
(code `4401`) before accepting.

**Server → client:** JSON text messages. For a turn (audio or text input), in this order:

| Type | Fields | Meaning |
|---|---|---|
| `transcript` | `text` | What Whisper heard, or the typed text for a `text_input` turn. |
| `reply_text` | `text` | The LLM's reply, before speech synthesis. |
| `reply_audio` | `audio_b64`, `final` | Base64-encoded WAV of one sentence of the spoken reply; sent once per sentence. `final` is `true` on the last chunk of the turn. |
| `error` | `message` | Something recoverable went wrong (e.g. no speech detected, rate limit exceeded); the turn ends without a reply. |
| `interrupted` | — | A new utterance or `text_input` arrived while a turn was still in flight; the previous turn was cancelled server-side and the client should stop any playback in progress (barge-in). |
| `persona_set` | `persona` | Acknowledges a successful `set_persona`. |

## Conversation memory & personas

`ConnectionState` (one instance per WebSocket connection) tracks the active persona and a running
`history` of `(user, assistant)` message pairs, trimmed to the last `MAX_HISTORY_TURNS` pairs
(default 6; set to `0` for the old fully-stateless behavior). Each turn's LLM call is built from
`state.build_messages(user_text)`: `[system prompt for the active persona] + trimmed history +
[this turn's user message]`. `PERSONAS` in the config block maps a key to a system prompt;
`set_persona` switches it and resets history, since mixing conversation context across personas
mid-conversation would be incoherent. History and persona live only in memory for the lifetime of
the connection — nothing is persisted, and a reconnect starts fresh.

## Barge-in (interrupting a reply)

Turn processing (`_run_turn`) runs as its own `asyncio.Task` rather than being awaited inline in
the receive loop, specifically so a new incoming message can interrupt it: `ws_endpoint`'s loop
calls `ws.receive()` continuously, and if a new audio blob or `text_input` arrives while the
previous turn's task is still running, that task is cancelled (`current_task.cancel()`) and the
client is told via `{"type": "interrupted"}` before the new turn starts. On the client, continuous
mode's VAD and push-to-talk's mic button both trigger the same local cleanup (`stopPlaybackQueue()`)
the moment new speech/typing starts, so local playback stops immediately rather than waiting for
the server round-trip.

## Rate limiting

`ConnectionState.allow_turn()` implements a simple rolling 60-second window, capping each
connection to `RATE_LIMIT_TURNS_PER_MINUTE` turns (default 20). Once exceeded, new turns get
`{"type": "error", "message": "rate limit exceeded — slow down"}` instead of being processed. This
is a per-connection guard against a runaway or misbehaving client, not a general abuse-prevention
system (it resets on reconnect).

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
  `faster_whisper.WhisperModel` loaded at startup. The temp file is always removed afterward
  (`finally`), including when transcription itself raises. The actual `stt_model.transcribe()`
  call is serialized with `stt_lock`, since concurrent calls into the same model instance from
  multiple executor threads aren't documented as safe by faster-whisper/ctranslate2.
- `query_llm(messages)` — a single non-streaming chat completion call against `LLAMA_SWAP_URL`,
  given the full message list built by `ConnectionState.build_messages()` (system prompt + trimmed
  history + this turn's user message).
- `synthesize(text)` — pipes text into the `piper` binary as a subprocess and wraps its raw PCM
  output in a WAV container at `PIPER_SAMPLE_RATE`. A `CalledProcessError` (piper ran but failed)
  is logged with piper's stderr before being re-raised, so the operator can see *why* piper failed
  rather than just that it did.

The Whisper model and the OpenAI client are constructed once at import time (`stt_model`,
`llm_client`), not per-request.

## Access control

There's no user accounts or session model — just an optional shared secret (`AUTH_TOKEN`). When
set, `_check_auth()` gates both `GET /` (401 without a valid `?token=`) and `/ws` (closed with
code `4401` before `accept()`). This is meant for a private network (e.g. a tailnet) where you
want a lightweight gate against anyone else on that network, not for internet-facing deployment.
