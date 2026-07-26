# Architecture

Everything — server and frontend — lives in one file, `voice_assistant.py`. FastAPI serves the
page at `GET /` and handles the entire voice/text loop over a single WebSocket at `/ws`. There is
no database, no build step, and no state that survives a process restart or a dropped connection:
all per-user state (conversation history, active persona, rate-limit counters) lives in memory,
scoped to one WebSocket connection, for as long as that connection is open.

This document walks through the system in three parts: the life of one turn end-to-end, the
server internals function by function, and the frontend's state machine.

## The life of one turn

A "turn" is one round trip: the user says or types something, and the assistant replies in text
and audio. Here's what happens for a spoken turn in push-to-talk mode:

1. **Recording.** The user holds the talk button. `pushStart()` grabs (or reuses) the mic stream
   via `getUserMedia` and starts a `MediaRecorder` capturing WebM/Opus. Releasing the button calls
   `mediaRecorder.stop()`.
2. **Upload.** `MediaRecorder.onstop` fires, the recorded chunks are assembled into a `Blob`, and
   — if it's bigger than the 1000-byte noise floor — sent as a single binary WebSocket message.
   The client immediately shows `"processing…"` and sets `busy = true`.
3. **Receive.** `ws_endpoint` on the server is sitting in a loop on `await ws.receive()`. It gets
   the binary message, and — since no turn is currently in flight — starts a background task,
   `_run_turn(ws, state, audio_bytes=..., user_text=None)`, and immediately loops back to
   `ws.receive()` again. This is the crux of how barge-in works: the receive loop is never
   blocked waiting for a turn to finish (see "Concurrency model" below).
4. **STT.** Inside `_run_turn`, `transcribe(audio_bytes)` runs in the default executor (a thread
   pool, so it doesn't block the asyncio event loop). It writes the blob to a temp `.webm` file,
   runs it through the shared `faster_whisper.WhisperModel` with `vad_filter=True`, and joins the
   resulting segments into one string. The temp file is always deleted afterward, success or
   failure (`finally: Path(path).unlink(...)`).
5. **Transcript sent.** The server sends `{"type": "transcript", "text": ...}`. If STT produced
   nothing (silence, filtered out by `vad_filter`), it sends `{"type": "error", "message": "no
   speech detected"}` instead and the turn ends there — no LLM call for silence.
6. **LLM.** `state.build_messages(user_text)` assembles `[system prompt for the active persona] +
   trimmed conversation history + [this turn's user message]`, and `query_llm(messages)` — also
   run in the executor — sends it to the OpenAI-compatible endpoint as a single non-streaming
   chat completion. The reply is recorded into `state.history` via `state.record_turn(...)`
   before anything else happens with it, so even if TTS below fails, memory of the exchange is
   preserved for the next turn.
7. **Reply text sent.** The server sends `{"type": "reply_text", "text": ...}` immediately — the
   client shows the reply text before audio is ready.
8. **TTS, sentence by sentence.** `split_sentences(reply)` breaks the reply on sentence
   boundaries. Each sentence is synthesized independently via `synthesize()` (which shells out to
   the `piper` binary and wraps its raw PCM output in a WAV container) and sent as its own
   `{"type": "reply_audio", "audio_b64": ..., "final": ...}` message the moment it's ready. This
   means playback of sentence 1 can start on the client while piper is still working on sentence
   2 — the client never waits for the entire reply to be synthesized before hearing anything.
9. **Playback.** The client's `enqueueAudio()` pushes each chunk onto `audioQueue` and
   `playNextInQueue()` plays them back to back via the single `<audio>` element, only resetting
   `busy = false` and the status text once the chunk marked `final: true` finishes.

A **text-typed** turn (via the text box) and a **persona switch** follow the same WebSocket, just
as JSON text messages instead of binary blobs — see the protocol table below. A text turn skips
step 4 entirely (`_run_turn` is called with `user_text` already set and `audio_bytes=None`) but is
otherwise identical from step 5 onward — memory, persona, and sentence-chunked TTS all apply the
same way regardless of whether the turn started as speech or as typed text.

## WebSocket message protocol

**Client → server:**

| Message | Meaning |
|---|---|
| binary WebM/Opus blob | One utterance, transcribed via `transcribe()`. |
| `{"type": "text_input", "text": "..."}` | Typed input — skips STT, goes straight to the LLM with the same conversation history and persona as a spoken turn. Blank/whitespace-only text is silently ignored. |
| `{"type": "set_persona", "persona": "..."}` | Switches the system prompt to `PERSONAS[persona]` and clears conversation history (a persona starts with a clean slate). Unrecognized keys are silently ignored. |

If `AUTH_TOKEN` is set, the connection URL must include `?token=<value>` or the server closes it
(code `4401`) before accepting.

**Server → client:** JSON text messages. For a turn (audio or text input), in this order:

| Type | Fields | Meaning |
|---|---|---|
| `transcript` | `text` | What Whisper heard, or the typed text for a `text_input` turn. |
| `reply_text` | `text` | The LLM's reply, before speech synthesis. |
| `reply_audio` | `audio_b64`, `final` | Base64-encoded WAV of one sentence of the spoken reply; sent once per sentence. `final` is `true` on the last chunk of the turn. |
| `error` | `message` | Something recoverable went wrong (no speech detected, STT/LLM/TTS failure, rate limit exceeded); the turn ends without a reply. |
| `interrupted` | — | A new utterance or `text_input` arrived while a turn was still in flight; the previous turn was cancelled server-side and the client should stop any playback in progress (barge-in). |
| `persona_set` | `persona` | Acknowledges a successful `set_persona`. |

## Server internals

### `transcribe(audio_bytes) -> str`

Writes the blob to a `NamedTemporaryFile(suffix=".webm")`, runs `stt_model.transcribe(path,
language="en", vad_filter=True)`, joins segment texts, and always removes the temp file in a
`finally` block — including when `stt_model.transcribe()` itself raises. `vad_filter=True` is
load-bearing: without it, Whisper routinely hallucinates a short word ("You", "Thank you") on
pure silence instead of returning empty text, which would send every silent clip to the LLM as if
it were real speech.

The actual model call is wrapped in `stt_lock` (`threading.Lock`), not because of any known bug,
but because faster-whisper/ctranslate2 doesn't document the shared `WhisperModel` instance as
safe for concurrent `.transcribe()` calls from multiple threads — and the WS handler dispatches
each call into a shared executor thread pool, so two connections transcribing at once is a real
possibility. Serializing avoids finding out the hard way.

### `query_llm(messages) -> str`

A single non-streaming chat completion call: `llm_client.chat.completions.create(model=
LLAMA_SWAP_MODEL, messages=messages)`, returning the stripped reply content. `messages` is the
full list built by `ConnectionState.build_messages()` — system prompt, trimmed history, new user
message — so this function itself has no notion of memory or personas; that logic lives entirely
in `ConnectionState`.

### `synthesize(text) -> bytes`

Pipes `text` into the `piper` binary (`piper --model PIPER_MODEL --output-raw`) via
`subprocess.run(..., check=True)` and wraps its raw 16-bit mono PCM stdout in a WAV container at
`PIPER_SAMPLE_RATE`. If piper exits non-zero, the `CalledProcessError` is logged with piper's
`stderr` (which is what actually names the failure — a bad model path, a corrupt `.onnx`) before
being re-raised for the caller to turn into a client-facing error.

### `split_sentences(text) -> list[str]`

A single regex (`(?<=[.!?])\s+`) splitting on sentence-ending punctuation followed by whitespace.
Falls back to `[text]` (the whole reply as one chunk) if the split produces nothing usable — e.g.
a reply with no terminal punctuation at all.

### `ConnectionState`

One instance is created per WebSocket connection (`state = ConnectionState()` inside
`ws_endpoint`) and lives for that connection's lifetime. It owns three things:

- **`persona`** — the active key into `PERSONAS`, starting at `DEFAULT_PERSONA`.
- **`history`** — a flat list of alternating `{"role": "user"/"assistant", "content": ...}`
  dicts, no system prompt (that's added fresh per call from the current persona). `record_turn()`
  appends a (user, assistant) pair and trims to the last `MAX_HISTORY_TURNS * 2` entries;
  `build_messages()` does the same trim independently (defensive — the two are meant to always
  agree, but `build_messages()` doesn't rely on `record_turn()` having already trimmed). Setting
  `MAX_HISTORY_TURNS = 0` makes `build_messages()` always pass an empty history, i.e. fully
  stateless turns.
- **`turn_timestamps`** — a list of `time.monotonic()` timestamps, one per turn actually started.
  `allow_turn()` filters this to the last 60 seconds, and returns `False` without recording a new
  timestamp once `RATE_LIMIT_TURNS_PER_MINUTE` are already in that window — a simple rolling-window
  rate limiter, not a token bucket, so it's slightly bursty at window edges but requires no extra
  bookkeeping.

`set_persona(persona)` swaps `self.persona` and resets `self.history = []`. History is wiped on
persona change deliberately: sending old context built for one persona's voice into a new
persona's system prompt would produce incoherent replies.

### `_run_turn(ws, state, *, audio_bytes, user_text)`

The actual turn pipeline — STT (if `audio_bytes` given) → LLM → TTS — as described in "The life
of one turn" above. Each stage's `except Exception` catches everything *except*
`asyncio.CancelledError`, which is explicitly re-raised (`except asyncio.CancelledError: raise`)
so that cancelling this task (for barge-in) actually unwinds it instead of being swallowed by the
stage's own error handling and reported to the client as, say, a "text-to-speech failed" error.

### `ws_endpoint(ws, token)`

The connection's main loop. After the `AUTH_TOKEN` check and `ws.accept()`, it:

1. Calls `await ws.receive()` directly (not the `receive_bytes()`/`receive_json()` convenience
   wrappers), because a single turn-agnostic receive loop needs to handle three message shapes:
   binary audio, `text_input` JSON, and `set_persona` JSON. It also has to detect
   `{"type": "websocket.disconnect"}` itself, since raw `receive()` doesn't raise
   `WebSocketDisconnect` the way the convenience methods do.
2. For `set_persona`, updates `state` and acknowledges — this never touches `current_task`, so it
   doesn't interrupt an in-flight turn.
3. For a new turn (binary audio, or a non-blank `text_input`): if a previous turn's task
   (`current_task`) is still running, cancels it and sends `{"type": "interrupted"}` first. Then
   checks `state.allow_turn()`; if the rate limit is exceeded, sends the rate-limit error and
   does *not* start a new task. Otherwise starts `_run_turn` as a new `asyncio.Task` and keeps
   looping — it does not `await` this task, which is what lets the loop go straight back to
   `ws.receive()` and stay responsive to the *next* interrupt.
4. On disconnect (either the explicit check in step 1, or a `WebSocketDisconnect` exception), the
   `finally` block cancels any still-running `current_task` so it doesn't keep doing STT/LLM/TTS
   work for a client that's already gone.

### `index(token)`

Serves the single self-contained HTML page, gated by the same `_check_auth()` used by `/ws`.
Returns `401 Unauthorized` (not the page) if `AUTH_TOKEN` is set and the token is missing or
wrong.

## Concurrency model

The server is a single asyncio event loop running one `ws_endpoint` coroutine per connection.
Three kinds of concurrency are in play, and it's worth being explicit about how they interact:

- **The event loop itself** is single-threaded and cooperative — nothing here uses `async` for
  parallelism, only for not blocking on I/O (`ws.receive()`, `ws.send_json()`) or on the executor
  handoff below.
- **`loop.run_in_executor(None, fn, ...)`** offloads the three genuinely blocking, CPU/subprocess-
  bound calls (`transcribe`, `query_llm`, `synthesize`) onto Python's default `ThreadPoolExecutor`
  so they don't stall the event loop for other connections while they run. This pool is shared
  across all connections and is unbounded in the sense that nothing here caps how many turns can
  be "in flight" across all clients at once — `RATE_LIMIT_TURNS_PER_MINUTE` bounds each
  *connection*, not the server as a whole.
- **`asyncio.Task` + cancellation** is what makes barge-in possible. Turn processing
  (`_run_turn`) is never `await`-ed directly in the receive loop; it's wrapped in
  `asyncio.create_task(...)` and stored as `current_task`, so the loop can go back to
  `ws.receive()` immediately after starting it. When a new message arrives, `current_task.cancel()`
  raises `CancelledError` inside `_run_turn` at its next `await` point (typically mid-`run_in_
  executor`) — note that this only cancels the *awaiting* coroutine; if the cancellation lands
  while the executor thread is actively running `transcribe`/`query_llm`/`synthesize`, that
  underlying call keeps running to completion in its background thread (Python can't forcibly
  kill a running thread), it just gets discarded — its result is never awaited or sent to the
  client. This is a deliberate simplification: the wasted work is bounded to one turn's worth of
  STT/LLM/TTS and isn't worth the complexity of real cancellation-token plumbing into
  `faster-whisper`/`openai`/`subprocess`.

## Access control

There's no user accounts or session model — just an optional shared secret (`AUTH_TOKEN`). When
set, `_check_auth()` gates both `GET /` (401 without a valid `?token=`) and `/ws` (closed with
code `4401` before `accept()`). This is meant for a private network (e.g. a tailnet) where you
want a lightweight gate against anyone else on that network, not for internet-facing deployment.

## Frontend

The page returned by `GET /` is a single self-contained HTML document (inline `<style>` and
`<script>`, no build step, no external assets).

### DOM & controls

- **Persona `<select>`** — options must match the server's `PERSONAS` keys (kept in sync
  manually; there's no dynamic endpoint listing them). Changing it sends `set_persona` over the
  open WebSocket immediately; on reconnect, `connect()` re-sends the currently-selected persona
  (if not `"default"`) once the socket opens, so a reconnect doesn't silently drop back to the
  default persona.
- **Mode toggle** (`push` / `continuous`) — `setMode()` tears down whichever mode is active
  (`pushStop()` or `stopContinuous()`) before starting the other, and toggles the talk button's
  visibility (hidden in continuous mode, since there's nothing to hold).
- **Talk button** — mouse, touch, and keyboard (`Space`/`Enter`) all map to the same
  `pushStart()`/`pushStop()` pair.
- **Text input row** — `sendTextInput()` reads and clears the input, updates the transcript/
  status/orb exactly as a spoken turn's transcript message would (since the server won't echo
  typed text back as a separate "you typed" event — the `transcript` message it does send would
  otherwise duplicate what's already in the input), and sends `{"type": "text_input", "text":
  ...}`.
- **`<audio id="player">`** — the single audio element all reply chunks play through, sequenced
  by the playback queue described below.

### State variables

- **`mode`** (`"push"` | `"continuous"`) — which recording mode is active.
- **`busy`** — true from the moment audio/text is sent until the *final* reply-audio chunk
  finishes playing (or an error/interrupt cancels that wait). Continuous mode's VAD only starts a
  *fresh* recording while not busy, except for the barge-in case below.
- **`speaking`** / **`silenceStart`** — continuous mode's VAD state: whether the mic is currently
  judged to be picking up voice, and when the current silence run started (used against
  `SILENCE_MS` to decide when an utterance has ended).
- **`vadRunning`** — whether the continuous-mode analysis loop should keep scheduling itself via
  `requestAnimationFrame`.
- **`audioQueue`** / **`playingQueue`** — the sentence-audio playback queue (see below).

### Voice activity detection (continuous mode)

`startContinuous()` opens a `Web Audio` `AnalyserNode` on the mic stream. `vadLoop()` runs once
per animation frame: it computes the RMS energy of the current audio frame and compares it to
`VOICE_THRESHOLD`. Crossing above the threshold while not already `speaking` starts a new
recording (`startRecorder()`); staying below it while `speaking` starts a silence timer, and once
that silence has lasted `SILENCE_MS` the recorder is stopped, which triggers the upload in
`mediaRecorder.onstop`. `VOICE_THRESHOLD` and `SILENCE_MS` are the two constants to retune for a
noisy room or a quiet mic.

### Barge-in (client side)

Voice detected (continuous mode) or the talk button pressed (push-to-talk) or text sent, while
`busy` is true, calls `stopPlaybackQueue()` before doing anything else: it pauses the `<audio>`
element, clears its `onended` handler (so a late `onended` firing from the just-paused clip
doesn't advance the queue or flip `busy` back), empties `audioQueue`, and resets `busy = false`.
This happens *before* the new recording/message is even sent, so local playback stops instantly
rather than waiting for the server's `{"type": "interrupted"}` acknowledgment — that message (see
above) exists to handle the symmetric case of the *server* cancelling a turn (because a new one
arrived) and needing to tell an otherwise-unaware client to stop.

### Playback queue

`enqueueAudio(audioB64, isFinal)` pushes each `reply_audio` chunk onto `audioQueue` and starts
`playNextInQueue()` if nothing is currently playing. `playNextInQueue()` plays one chunk via
`player.src` + `player.play()`, and on `onended` either advances to the next queued chunk or —
if this was the `final` chunk — resets `busy` and the status/orb back to idle/listening. This
queue exists specifically because replies now arrive as multiple sentence-sized audio messages
(see `split_sentences()` server-side) rather than one message per reply; without it, reassigning
`player.src` while a previous chunk was still playing would cut it off.

### Status visualization

The `<canvas id="orb">` animates a small set of glowing nodes connected by lines when close
together (`drawOrb()`, driven by `requestAnimationFrame`). `setOrb(state)` sets one of `""` /
`listening` / `thinking` / `speaking`, each mapped in `ORB_PALETTES` to a distinct animation
speed, glow intensity, size multiplier, and (for `thinking`) a flicker effect — deliberately
grayscale-only differentiation (speed/glow/size, not color) since the whole UI is monochrome.

### Connection lifecycle

`connect()` opens the WebSocket (appending `?token=...` if one is present in the page's own URL
query string), and `ws.onclose` schedules a reconnect after 1.5s with a status message. There's
no capped backoff — a persistently unreachable server retries every 1.5s indefinitely, which is
intentional for a small self-hosted tool (you want it to just come back the moment the server
does) but would need a backoff cap before this app was ever exposed to a flaky/adversarial network.
