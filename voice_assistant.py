"""
Single-file voice assistant.
Browser mic -> WebSocket -> faster-whisper STT -> llama-swap (OpenAI-compatible) -> piper TTS -> back to browser.

Requirements:
    pip install fastapi uvicorn[standard] faster-whisper openai websockets

External binary (TTS):
    piper (https://github.com/rhasspy/piper) — download a release binary + a voice .onnx model,
    set PIPER_BIN and PIPER_MODEL below.

Run:
    python voice_assistant.py
Then open http://<tailscale-hostname>:8000 from any device on your tailnet.
"""

import asyncio
import base64
import io
import logging
import re
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from openai import OpenAI

# ---- config ----
LLAMA_SWAP_URL = "http://localhost:8080/v1"   # your llama-swap endpoint
LLAMA_SWAP_MODEL = "your-model-name"          # model name as configured in llama-swap

# Shared secret required to use the app — leave empty to allow anyone who can reach HOST:PORT
# (fine on a private tailnet, not fine on an open network). When set, clients must pass it as
# ?token=... on both the page URL and the WebSocket URL.
AUTH_TOKEN = ""

# Whisper STT hardware profile — flip this to match whatever machine you're running on.
USE_GPU = True

WHISPER_GPU_MODEL_SIZE = "medium"      # GPU headroom — medium is a real accuracy step up over small
WHISPER_GPU_DEVICE = "cuda"
WHISPER_GPU_COMPUTE_TYPE = "float16"   # int8_float16 if VRAM is tight

WHISPER_CPU_MODEL_SIZE = "small"       # medium/large are too slow for real-time on CPU
WHISPER_CPU_DEVICE = "cpu"
WHISPER_CPU_COMPUTE_TYPE = "int8"

WHISPER_MODEL_SIZE = WHISPER_GPU_MODEL_SIZE if USE_GPU else WHISPER_CPU_MODEL_SIZE
WHISPER_DEVICE = WHISPER_GPU_DEVICE if USE_GPU else WHISPER_CPU_DEVICE
WHISPER_COMPUTE_TYPE = WHISPER_GPU_COMPUTE_TYPE if USE_GPU else WHISPER_CPU_COMPUTE_TYPE

PIPER_BIN = "/usr/local/bin/piper"
PIPER_MODEL = "/opt/piper/en_US-lessac-medium.onnx"
PIPER_SAMPLE_RATE = 22050   # must match your voice model's .onnx.json "sample_rate", or audio is pitched wrong
HOST = "0.0.0.0"
PORT = 8000
# ----------------

app = FastAPI()
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
llm_client = OpenAI(base_url=LLAMA_SWAP_URL, api_key="not-needed")

# faster-whisper/ctranslate2 doesn't document the shared WhisperModel instance as safe for
# concurrent .transcribe() calls from multiple threads, and the WS handler runs each call in a
# shared executor thread pool — serialize access rather than risk corrupted/interleaved results.
stt_lock = threading.Lock()

SYSTEM_PROMPT = "You are a concise voice assistant. Keep replies short — this will be read aloud."

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Splits a reply into sentence-sized chunks so TTS/playback can start before the whole
    reply is synthesized, instead of waiting on piper for the entire (possibly long) reply."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def _check_auth(token: str | None) -> bool:
    return not AUTH_TOKEN or token == AUTH_TOKEN


def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        # vad_filter matters more than it looks: without it, Whisper routinely hallucinates a
        # short word ("You", "Thank you") on pure silence instead of returning empty text, which
        # would otherwise send every silent clip to the LLM as if it were real speech.
        with stt_lock:
            segments, _ = stt_model.transcribe(path, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        Path(path).unlink(missing_ok=True)


def query_llm(text: str) -> str:
    resp = llm_client.chat.completions.create(
        model=LLAMA_SWAP_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip()


def synthesize(text: str) -> bytes:
    """Runs piper, returns raw WAV bytes."""
    try:
        result = subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # check=True's own message just says "exited with code N" — log piper's stderr too,
        # since that's what actually tells you it's a bad model path or malformed input.
        logging.error("piper exited with %s: %s", e.returncode, e.stderr.decode(errors="replace").strip())
        raise
    pcm = result.stdout  # raw 16-bit mono PCM at PIPER_SAMPLE_RATE
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(PIPER_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: str | None = Query(None)):
    if not _check_auth(token):
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            loop = asyncio.get_event_loop()

            # Each stage is wrapped separately so a downstream outage (LLM unreachable, piper
            # missing, a bad model file) sends the client a specific, recoverable error message
            # instead of raising out of the handler and silently killing the connection — which
            # left the client stuck showing "processing…" forever with no explanation.
            try:
                text = await loop.run_in_executor(None, transcribe, audio_bytes)
            except Exception:
                logging.exception("STT failed")
                await ws.send_json({"type": "error", "message": "speech-to-text failed"})
                continue

            if not text:
                await ws.send_json({"type": "error", "message": "no speech detected"})
                continue

            await ws.send_json({"type": "transcript", "text": text})

            try:
                reply = await loop.run_in_executor(None, query_llm, text)
            except Exception:
                logging.exception("LLM request failed")
                await ws.send_json({"type": "error", "message": "couldn't reach the language model"})
                continue

            await ws.send_json({"type": "reply_text", "text": reply})

            # Synthesize and send sentence-by-sentence rather than the whole reply at once, so
            # playback of the first sentence can start while later sentences are still being
            # synthesized, instead of the client waiting on piper for the entire reply.
            try:
                sentences = split_sentences(reply)
                for i, sentence in enumerate(sentences):
                    wav_bytes = await loop.run_in_executor(None, synthesize, sentence)
                    await ws.send_json({
                        "type": "reply_audio",
                        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                        "final": i == len(sentences) - 1,
                    })
            except Exception:
                logging.exception("TTS failed")
                await ws.send_json({"type": "error", "message": "text-to-speech failed"})
                continue
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
async def index(token: str | None = None):
    if not _check_auth(token):
        return HTMLResponse("Unauthorized", status_code=401)
    return """
<!DOCTYPE html>
<html>
<head>
<title>Voice</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: space-between;
    background: #0b0b0f; color: #eaeaf0; font-family: -apple-system, system-ui, sans-serif;
    padding: 32px 20px; overflow: hidden; position: relative;
  }
  /* ambient light behind the glass controls — flat black gives backdrop-filter nothing to blur */
  body::before, body::after {
    content: ""; position: absolute; z-index: 0; border-radius: 50%; filter: blur(90px); pointer-events: none;
  }
  body::before { width: 320px; height: 320px; opacity: .16; background: #fff; top: -110px; left: -90px; animation: driftA 20s ease-in-out infinite alternate; }
  body::after { width: 280px; height: 280px; opacity: .10; background: #fff; bottom: -110px; right: -80px; animation: driftB 24s ease-in-out infinite alternate; }
  @keyframes driftA { from { transform: translate(0, 0); } to { transform: translate(50px, 40px); } }
  @keyframes driftB { from { transform: translate(0, 0); } to { transform: translate(-40px, -50px); } }

  .top, .transcript-wrap, .controls { position: relative; z-index: 1; }

  .top { text-align: center; }
  .status { font-size: 15px; color: #9a9a9a; letter-spacing: .02em; height: 20px; transition: color .2s; }
  .status.error { color: #fff; font-weight: 600; animation: statusPulse 1.4s ease-in-out infinite; }
  @keyframes statusPulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
  .transcript-wrap { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 18px; width: 100%; max-width: 480px; }
  .bubble { font-size: 17px; line-height: 1.5; text-align: center; padding: 0 8px; min-height: 1.5em; transition: opacity .2s; }
  .bubble.you { color: #9a9a9a; }
  .bubble.reply { color: #f2f2f7; font-size: 19px; }

  .orb-wrap { position: relative; width: 200px; height: 200px; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; }
  .orb { width: 200px; height: 200px; transition: filter .3s ease; filter: drop-shadow(0 0 22px rgba(255,255,255,.28)); }
  .orb.listening { filter: drop-shadow(0 0 30px rgba(255,255,255,.4)); }
  .orb.speaking { filter: drop-shadow(0 0 38px rgba(255,255,255,.5)); }
  .orb.thinking { filter: drop-shadow(0 0 26px rgba(255,255,255,.35)); }

  .controls { display: flex; flex-direction: column; align-items: center; gap: 20px; width: 100%; }

  .mode-toggle {
    display: flex; gap: 4px; padding: 4px; border-radius: 999px;
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
    backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 6px 20px rgba(0,0,0,.3);
  }
  .mode-toggle button {
    border: 1px solid transparent; background: transparent; color: #9a9a9a; padding: 8px 18px; border-radius: 999px;
    font-size: 14px; cursor: pointer; transition: all .2s;
  }
  .mode-toggle button.active {
    background: linear-gradient(160deg, rgba(255,255,255,.24), rgba(255,255,255,.08));
    border-color: rgba(255,255,255,.22); color: #fff;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.24);
  }
  .mode-toggle button:disabled { opacity: .35; cursor: not-allowed; }
  .mode-toggle button:focus-visible { outline: 2px solid rgba(255,255,255,.85); outline-offset: 2px; }

  .talk-btn {
    width: 76px; height: 76px; border-radius: 50%; cursor: pointer;
    border: 1px solid rgba(255,255,255,.18);
    background: linear-gradient(160deg, rgba(255,255,255,.22), rgba(255,255,255,.07));
    backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
    color: white; display: flex; align-items: center; justify-content: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.3), inset 0 -14px 20px -12px rgba(0,0,0,.4), 0 10px 26px rgba(0,0,0,.45);
    user-select: none; -webkit-user-select: none; touch-action: none;
    transition: transform .1s, background .2s, box-shadow .2s, color .2s;
  }
  .talk-btn:active { transform: scale(0.92); }
  .talk-btn.recording {
    background: linear-gradient(160deg, rgba(255,255,255,.92), rgba(255,255,255,.78));
    border-color: rgba(255,255,255,.6);
    color: #0b0b0f;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.8), inset 0 -10px 16px -10px rgba(0,0,0,.15), 0 0 30px rgba(255,255,255,.3);
  }
  .talk-btn.hidden { display: none; }
  .talk-btn:disabled { opacity: .35; cursor: not-allowed; }
  .talk-btn:focus-visible { outline: 2px solid rgba(255,255,255,.85); outline-offset: 3px; }
  .talk-btn .icon-stop { display: none; }
  .talk-btn.recording .icon-mic { display: none; }
  .talk-btn.recording .icon-stop { display: block; }
  .hint { font-size: 13px; color: #7d7d7d; }
</style>
</head>
<body>
  <div class="top"><div class="status" id="status" role="status" aria-live="polite">connecting…</div></div>

  <div class="transcript-wrap">
    <div class="orb-wrap"><canvas class="orb" id="orb"></canvas></div>
    <div class="bubble you" id="transcript"></div>
    <div class="bubble reply" id="reply"></div>
  </div>

  <div class="controls">
    <div class="mode-toggle">
      <button id="modePush" class="active" aria-pressed="true">Push to talk</button>
      <button id="modeCont" aria-pressed="false">Continuous</button>
    </div>
    <button class="talk-btn" id="rec" aria-label="Hold to talk">
      <svg class="icon-mic" viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 15a3.5 3.5 0 0 0 3.5-3.5v-5a3.5 3.5 0 0 0-7 0v5A3.5 3.5 0 0 0 12 15Z"/>
        <path d="M7 11a5 5 0 0 0 10 0"/>
        <path d="M12 16v3"/>
        <path d="M9 19h6"/>
      </svg>
      <svg class="icon-stop" viewBox="0 0 24 24" width="22" height="22" fill="currentColor" aria-hidden="true">
        <rect x="6" y="6" width="12" height="12" rx="2"/>
      </svg>
    </button>
    <div class="hint" id="hint">hold to talk</div>
  </div>

  <audio id="player"></audio>

<script>
const statusEl = document.getElementById("status");
const orb = document.getElementById("orb");
const orbCtx = orb.getContext("2d");
const transcriptEl = document.getElementById("transcript");
const replyEl = document.getElementById("reply");
const recBtn = document.getElementById("rec");
const hint = document.getElementById("hint");
const modePushBtn = document.getElementById("modePush");
const modeContBtn = document.getElementById("modeCont");
const player = document.getElementById("player");

let mode = "push"; // "push" | "continuous"
let ws, mediaRecorder, stream, chunks = [];
let audioCtx, analyser, vadRunning = false, speaking = false, silenceStart = null;
let busy = false; // waiting on server round-trip — pause VAD while true

const SILENCE_MS = 900;      // gap of quiet that ends an utterance
const VOICE_THRESHOLD = 0.02; // RMS threshold, tune to your mic/room

// ---- animated node network (replaces the old static orb) ----
let orbState = "";

// states read only in grayscale — speed, glow, size, and flicker carry the meaning color used to
const ORB_PALETTES = {
  "": { rgb: [215, 215, 215], speed: 1, glow: 6, sizeMul: 1, flicker: false },
  listening: { rgb: [255, 255, 255], speed: 1.5, glow: 10, sizeMul: 1.1, flicker: false },
  thinking: { rgb: [255, 255, 255], speed: 2.6, glow: 9, sizeMul: 1, flicker: true },
  speaking: { rgb: [255, 255, 255], speed: 1.8, glow: 14, sizeMul: 1.25, flicker: false },
};

const ORB_NODES = Array.from({ length: 12 }, (_, i) => ({
  baseRadius: 22 + (i % 4) * 13,
  angle: (i / 12) * Math.PI * 2,
  dir: i % 2 === 0 ? 1 : -1,
  speed: 0.12 + (i % 5) * 0.03,
  size: 1.8 + (i % 3) * 0.6,
}));

function setOrb(state) {
  orb.className = "orb" + (state ? " " + state : "");
  orbState = state;
}

function resizeOrbCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const size = orb.clientWidth || 200;
  orb.width = size * dpr;
  orb.height = size * dpr;
  orbCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resizeOrbCanvas);
resizeOrbCanvas();

function drawOrb(t) {
  const size = orb.clientWidth || 200;
  const cx = size / 2, cy = size / 2;
  const palette = ORB_PALETTES[orbState] || ORB_PALETTES[""];
  const pulse = 1 + 0.12 * Math.sin(t / (orbState === "thinking" ? 260 : 900));

  orbCtx.clearRect(0, 0, size, size);

  const positions = ORB_NODES.map((n) => {
    const angle = n.angle + n.dir * n.speed * palette.speed * (t / 1000);
    const radius = n.baseRadius * pulse;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle), size: n.size };
  });

  const [r, g, b] = palette.rgb;
  const threshold = size * 0.34;
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const dx = positions[i].x - positions[j].x;
      const dy = positions[i].y - positions[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < threshold) {
        orbCtx.strokeStyle = `rgba(${r},${g},${b},${((1 - dist / threshold) * 0.5).toFixed(3)})`;
        orbCtx.lineWidth = 1;
        orbCtx.beginPath();
        orbCtx.moveTo(positions[i].x, positions[i].y);
        orbCtx.lineTo(positions[j].x, positions[j].y);
        orbCtx.stroke();
      }
    }
  }

  positions.forEach((p, i) => {
    const flicker = palette.flicker ? 0.55 + 0.45 * Math.abs(Math.sin(t / 120 + i)) : 1;
    orbCtx.beginPath();
    orbCtx.fillStyle = `rgba(${r},${g},${b},${(0.95 * flicker).toFixed(3)})`;
    orbCtx.shadowColor = `rgba(${r},${g},${b},${(0.9 * flicker).toFixed(3)})`;
    orbCtx.shadowBlur = palette.glow;
    orbCtx.arc(p.x, p.y, p.size * palette.sizeMul, 0, Math.PI * 2);
    orbCtx.fill();
  });
  orbCtx.shadowBlur = 0;

  requestAnimationFrame(drawOrb);
}
requestAnimationFrame(drawOrb);

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function connect() {
  const token = new URLSearchParams(location.search).get("token");
  const qs = token ? `?token=${encodeURIComponent(token)}` : "";
  ws = new WebSocket(`ws://${location.host}/ws${qs}`);
  ws.onopen = () => { setStatus(mode === "push" ? "ready" : "listening…"); };
  ws.onclose = () => { setStatus("disconnected — reconnecting…", true); setTimeout(connect, 1500); };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "transcript") {
      transcriptEl.textContent = msg.text;
      replyEl.textContent = "";
      setStatus("thinking…");
      setOrb("thinking");
    }
    if (msg.type === "reply_text") {
      replyEl.textContent = msg.text;
    }
    if (msg.type === "reply_audio") {
      enqueueAudio(msg.audio_b64, msg.final !== false);
    }
    if (msg.type === "error") {
      busy = false;
      audioQueue = [];
      setStatus(msg.message, true);
      setOrb(mode === "continuous" ? "listening" : "");
    }
  };
}

// Replies may arrive as several sentence-sized audio chunks in sequence (see server-side
// split_sentences) — queue them so playback is gapless instead of racing player.src reassignment.
let audioQueue = [];
let playingQueue = false;

function enqueueAudio(audioB64, isFinal) {
  audioQueue.push({ audioB64, isFinal });
  if (!playingQueue) playNextInQueue();
}

function playNextInQueue() {
  if (audioQueue.length === 0) { playingQueue = false; return; }
  playingQueue = true;
  const { audioB64, isFinal } = audioQueue.shift();
  player.src = "data:audio/wav;base64," + audioB64;
  setOrb("speaking");
  setStatus("speaking…");
  player.play();
  player.onended = () => {
    if (isFinal) {
      busy = false;
      setOrb(mode === "continuous" ? "listening" : "");
      setStatus(mode === "continuous" ? "listening…" : "ready");
    }
    playNextInQueue();
  };
}

function micSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
}

if (micSupported()) {
  connect();
} else {
  setStatus("This browser can't record audio — try the latest Chrome, Edge, or Firefox.", true);
  recBtn.disabled = true;
  modePushBtn.disabled = true;
  modeContBtn.disabled = true;
}

async function getStream() {
  if (stream) return stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return stream;
  } catch (err) {
    const message = err.name === "NotAllowedError" || err.name === "PermissionDeniedError"
      ? "Microphone access denied — allow it in your browser's site settings and reload."
      : `Couldn't access the microphone (${err.message}).`;
    setStatus(message, true);
    setOrb("");
    throw err;
  }
}

function startRecorder() {
  chunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    const blob = new Blob(chunks, { type: "audio/webm" });
    if (blob.size > 1000) {
      ws.send(blob);
      busy = true;
      setStatus("processing…");
      setOrb("thinking");
    } else if (mode === "continuous") {
      setOrb("listening");
    }
  };
  mediaRecorder.start();
}

// ---- push to talk ----
async function pushStart() {
  if (recBtn.classList.contains("recording")) return;
  try {
    await getStream();
  } catch {
    return;
  }
  startRecorder();
  recBtn.classList.add("recording");
  setOrb("listening");
  setStatus("listening…");
}
function pushStop() {
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  recBtn.classList.remove("recording");
}

// ---- continuous mode (client-side VAD) ----
async function startContinuous() {
  try {
    await getStream();
  } catch {
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  vadRunning = true;
  speaking = false;
  silenceStart = null;
  setOrb("listening");
  setStatus("listening…");
  vadLoop();
}

function vadLoop() {
  if (!vadRunning) return;
  const data = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
  const rms = Math.sqrt(sum / data.length);

  if (!busy) {
    if (rms > VOICE_THRESHOLD) {
      if (!speaking) {
        speaking = true;
        startRecorder();
        setOrb("listening");
      }
      silenceStart = null;
    } else if (speaking) {
      if (silenceStart === null) silenceStart = Date.now();
      else if (Date.now() - silenceStart > SILENCE_MS) {
        speaking = false;
        silenceStart = null;
        if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
      }
    }
  }
  requestAnimationFrame(vadLoop);
}

function stopContinuous() {
  vadRunning = false;
  speaking = false;
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  setOrb("");
}

// ---- mode switching ----
function setMode(newMode) {
  if (newMode === mode) return;
  if (mode === "continuous") stopContinuous();
  if (mode === "push") pushStop();
  mode = newMode;
  modePushBtn.classList.toggle("active", mode === "push");
  modeContBtn.classList.toggle("active", mode === "continuous");
  modePushBtn.setAttribute("aria-pressed", mode === "push");
  modeContBtn.setAttribute("aria-pressed", mode === "continuous");
  recBtn.classList.toggle("hidden", mode === "continuous");
  hint.textContent = mode === "push" ? "hold to talk" : "just start talking";
  if (mode === "continuous") startContinuous();
  else { setOrb(""); setStatus("ready"); }
}

modePushBtn.onclick = () => setMode("push");
modeContBtn.onclick = () => setMode("continuous");

recBtn.onmousedown = () => mode === "push" && pushStart();
recBtn.onmouseup = () => mode === "push" && pushStop();
recBtn.onmouseleave = () => mode === "push" && pushStop();
recBtn.ontouchstart = (e) => { e.preventDefault(); mode === "push" && pushStart(); };
recBtn.ontouchend = (e) => { e.preventDefault(); mode === "push" && pushStop(); };
recBtn.onkeydown = (e) => {
  if ((e.code === "Space" || e.code === "Enter") && mode === "push" && !e.repeat) {
    e.preventDefault();
    pushStart();
  }
};
recBtn.onkeyup = (e) => {
  if ((e.code === "Space" || e.code === "Enter") && mode === "push") {
    e.preventDefault();
    pushStop();
  }
};
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
