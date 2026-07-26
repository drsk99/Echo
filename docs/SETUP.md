# Setup & configuration

## 1. Install Python dependencies

```bash
pip install fastapi "uvicorn[standard]" faster-whisper openai websockets
```

`faster-whisper` pulls in `ctranslate2`; the GPU path additionally needs a working CUDA + cuDNN
install matched to your driver.

## 2. Install piper (TTS)

Download a release binary from [rhasspy/piper](https://github.com/rhasspy/piper) and a voice
model (a `.onnx` file plus its matching `.onnx.json`). Note the binary path and the model path —
you'll put both in the config block.

## 3. Point at an LLM endpoint

`voice_assistant.py` talks to any OpenAI-compatible chat completions endpoint. The project was
built against [llama-swap](https://github.com/mostlygeek/llama-swap), which hot-swaps local GGUF
models behind an OpenAI-compatible API, but any compatible server (llama.cpp's own server, vLLM,
LM Studio, etc.) works — just point `LLAMA_SWAP_URL` at it.

## 4. Configure `voice_assistant.py`

All configuration lives in one block near the top of the file:

| Variable | Purpose |
|---|---|
| `LLAMA_SWAP_URL` | Base URL of your OpenAI-compatible LLM endpoint (e.g. `http://localhost:8080/v1`). |
| `LLAMA_SWAP_MODEL` | Model name as configured on that endpoint. |
| `USE_GPU` | `True`/`False` — selects the GPU or CPU Whisper profile below. This is the only line you flip when moving the script between machines. |
| `WHISPER_GPU_MODEL_SIZE` / `WHISPER_GPU_DEVICE` / `WHISPER_GPU_COMPUTE_TYPE` | Whisper settings used when `USE_GPU = True`. Defaults: `medium` / `cuda` / `float16`. Drop to `int8_float16` if VRAM is tight. |
| `WHISPER_CPU_MODEL_SIZE` / `WHISPER_CPU_DEVICE` / `WHISPER_CPU_COMPUTE_TYPE` | Whisper settings used when `USE_GPU = False`. Defaults: `small` / `cpu` / `int8` — `medium`/`large` are too slow for real-time use on CPU. |
| `PIPER_BIN` | Path to the piper binary. |
| `PIPER_MODEL` | Path to the piper voice `.onnx` model. |
| `HOST` / `PORT` | Bind address for the FastAPI server. Defaults to `0.0.0.0:8000` so it's reachable from other devices on your network. |

Only the variable *names* are fixed — every value is yours to change; there's no separate config
file or environment variable layer.

## 5. Run it

```bash
python voice_assistant.py
```

Open `http://<host>:8000` from any device on the same network — for example a Tailscale
hostname, so your phone can reach a voice assistant running on a home server.

## Troubleshooting

- **"This browser can't record audio"** — the browser lacks `MediaRecorder`/`getUserMedia`
  support, or the page isn't loaded from a secure context (`https://` or a private-network
  origin some browsers treat as secure). Use an up-to-date Chrome, Edge, or Firefox.
- **"Microphone access denied"** — allow microphone access for the site in your browser's site
  settings, then reload.
- **Status shows "disconnected — reconnecting…"** — the FastAPI process isn't reachable; check
  it's running and that `HOST`/`PORT` and any firewall/Tailscale ACLs allow the connection.
- **LLM replies never arrive / connection errors in the server logs** — confirm
  `LLAMA_SWAP_URL` is reachable from the machine running `voice_assistant.py` and that
  `LLAMA_SWAP_MODEL` matches a model name the endpoint actually serves.
- **No audio comes back** — verify `PIPER_BIN` and `PIPER_MODEL` are correct, executable, and
  that `PIPER_BIN --model PIPER_MODEL --output-raw` produces PCM output when run manually with
  text piped to stdin.
