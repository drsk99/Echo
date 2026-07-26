# Echo

A single-file, self-hosted voice assistant you talk to from a browser. Speech-to-text, the LLM
reply, and text-to-speech all run on your own hardware — nothing goes to a third-party cloud API.

```
Browser mic → WebSocket → faster-whisper (STT) → llama-swap (OpenAI-compatible LLM) → piper (TTS) → browser
```

## Features

- **Push-to-talk or continuous mode** — hold a button, or let client-side voice-activity detection
  start/stop recording for you.
- **Type instead of speaking** — a text input alongside the mic for when talking isn't convenient.
- **Conversation memory** — recent turns are sent back to the LLM as context, so replies can refer
  to what you just said (configurable window, or disable for fully stateless turns).
- **Personas** — switch the assistant's system prompt (and start a fresh conversation) from a
  dropdown, no restart required.
- **Barge-in** — start talking (or typing) while the assistant is still replying and it stops
  immediately instead of finishing its sentence.
- **Live status visualization** — an animated node network reflects idle / listening / thinking /
  speaking state in real time.
- **CPU/GPU toggle** — flip one flag to switch the Whisper model between a CUDA and a CPU profile;
  no separate config files to maintain.
- **Optional access token** — gate the app behind a shared secret when it's reachable by more than
  just you.
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

Licensed under the [Apache License 2.0](LICENSE).
