# Echo

A single-file, self-hosted voice assistant you talk to from a browser. Speech-to-text, the LLM
reply, and text-to-speech all run on your own hardware — nothing goes to a third-party cloud API.

```
Browser mic → WebSocket → faster-whisper (STT) → llama-swap (OpenAI-compatible LLM) → piper (TTS) → browser
```

## Features

- **Push-to-talk or continuous mode** — hold a button, or let client-side voice-activity detection
  start/stop recording for you.
- **Live status visualization** — an animated node network reflects idle / listening / thinking /
  speaking state in real time.
- **CPU/GPU toggle** — flip one flag to switch the Whisper model between a CUDA and a CPU profile;
  no separate config files to maintain.
- **Clean, accessible UI** — keyboard-operable controls, screen-reader-friendly status updates,
  clear error states (denied mic permission, unsupported browser, dropped connection).
- **Zero build step** — one Python file serves both the API and the entire frontend.

## Requirements

- Python 3.10+
- [piper](https://github.com/rhasspy/piper) — TTS binary plus a voice `.onnx` model
- An OpenAI-compatible LLM endpoint, e.g. [llama-swap](https://github.com/mostlygeek/llama-swap)
- An NVIDIA GPU (optional — see the CPU/GPU toggle in [docs/SETUP.md](docs/SETUP.md))

## Quick start

```bash
pip install fastapi "uvicorn[standard]" faster-whisper openai websockets
```

1. Download a piper release binary and a voice model.
2. Edit the config block at the top of `voice_assistant.py` (endpoint URLs, model paths, the
   `USE_GPU` flag) — see [docs/SETUP.md](docs/SETUP.md) for every option.
3. Run it:

   ```bash
   python voice_assistant.py
   ```
4. Open `http://<host>:8000` from any device on the same network (e.g. your Tailscale tailnet).

## Docs

- [Setup & configuration reference](docs/SETUP.md)
- [Architecture](docs/ARCHITECTURE.md)

## License

No license specified — personal project, all rights reserved.
