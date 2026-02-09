# J.A.R.V.I.S. – Offline Voice Assistant on Jetson Orin Nano Super

A fully offline AI assistant modelled on **J.A.R.V.I.S.** from the Iron Man films (Paul Bettany's portrayal), running on the **Jetson Orin Nano Super Developer Kit (8GB)** with JetPack 6.x. Uses Bluetooth (e.g. Google Pixel Buds 2) for mic and TTS output, USB webcam for vision, and runs LLM (Ollama/Qwen3), STT (Faster-Whisper), TTS (Piper), and wake word (openWakeWord) locally.

Includes a **SvelteKit PWA** frontend served over the LAN and a **FastAPI WebSocket bridge** so you can chat, view the camera feed, and manage reminders from any device on the network.

## Performance

Optimised for **sub-2-second** chat and **sub-10-second** tool calls on the 8 GB Jetson:

| Tuning | Value | Why |
|--------|-------|-----|
| Model | `qwen3:1.7b` (Q4_K_M, 1.4 GB) | Native tool-calling, fits 100% GPU at 8192 ctx |
| Context window | **8192 tokens** | 100% GPU at 2.0 GB total; 3.5x faster than old 2048 (eliminates KV-cache thrashing) |
| Thinking | **Adaptive** — `think=false` for chat, `think=true` for tools | Qwen3 requires reasoning chain for tool routing; plain chat skips it for speed |
| Output cap | **512 tokens** (`num_predict`) | Qwen3 thinking tokens count toward budget; 256 starved content |
| Temperature | 0.6 | Faster convergence, deterministic |
| GPU offload | **100% GPU** (2.0 GB) | Old 2048 limit was sized for Llama3.2:3b; Qwen3:1.7b is 600 MB lighter |
| Tool routing | **Intent-based** | Only sends tool schemas when query matches keywords; prevents wasted think budget |
| Tool schemas | 4 (was 7) | Time, stats, reminders already injected into context |
| History | 4 turns | Less prefill work for the GPU |

Benchmarked latencies (warm, 100% GPU):

| Scenario | Latency | Mode |
|----------|---------|------|
| Greeting / status / time | **0.5–0.7 s** | `think=false`, no tools |
| Tool call (joke, reminder, sarcasm) | **3.6–8.4 s** | `think=true`, selected tools only |
| Vision query (pre-fetched) | **0.7 s** | `think=false`, scene in context |

Context size benchmarks (Jetson Orin Nano 8 GB, Qwen3:1.7b):

| num_ctx | Model Size | GPU% | Chat Latency |
|---------|-----------|------|-------------|
| 2048 | 1.6 GB | 100% | 12.9 s (KV thrashing) |
| 4096 | 1.7 GB | 100% | 4.1 s |
| **8192** | **2.0 GB** | **100%** | **3.5 s** ← production |
| 12288 | 2.3 GB | 100% | ~4 s (swap pressure) |
| 16384 | 2.6 GB | 30/70 CPU/GPU | slow (unacceptable) |

## Requirements

- **Hardware**: Jetson Orin Nano Super (8GB), 128 GB+ storage (microSD or SSD), USB webcam, Bluetooth earbuds (e.g. Pixel Buds 2)
- **OS**: JetPack 6.x (L4T R36.x), Ubuntu 22.04 base
- **RAM**: Keep total usage under ~7.5 GiB to avoid swap on microSD

## Power mode (MAXN_SUPER)

For best performance, use MAXN_SUPER and lock max clocks:

```bash
# Check current mode (should show MAXN_SUPER)
sudo nvpmodel -q

# Lock max CPU/GPU/EMC clocks
sudo jetson_clocks

# Monitor thermals and power
tegrastats
# or install: sudo pip3 install jetson-stats && jtop
```

## System packages (audio & Bluetooth)

```bash
sudo apt update
sudo apt install -y wireplumber pipewire-audio pipewire-pulse bluez-tools pulseaudio-utils
```

Set default sink/source for Pixel Buds via `pactl` or `wpctl` (after Wireplumber), or use your desktop sound settings.

## Bluetooth pairing (Pixel Buds 2)

1. Put the buds in pairing mode.
2. On the Jetson:

```bash
bluetoothctl
power on
scan on
# Find "Pixel Buds" (or your device), note the MAC address
pair <MAC>
trust <MAC>
connect <MAC>
```

3. **A2DP** is used for high-quality TTS output. For **mic input**, many buds need **HFP/HSP**. If the buds do not appear as an input device, switch the profile to HFP in `bluetoothctl` or via Blueman. If HFP is unreliable on JetPack 6.x, use a **USB microphone** as fallback and keep A2DP for output only.

## Python environment

```bash
cd /home/jarvis/Jarvis
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## CUDA and PyTorch (per NVIDIA)

For GPU-accelerated PyTorch (e.g. YOLO TensorRT export), follow [NVIDIA's official guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html):

1. **System packages** (once):
   ```bash
   sudo apt-get -y update
   sudo apt-get install -y python3-pip libopenblas-dev
   ```

2. **cuSPARSELt** (required for PyTorch 24.06+ on JetPack 6.x):
   ```bash
   bash scripts/install-cusparselt.sh
   ```

3. **CUDA in PATH** – ensure `/etc/profile.d/cuda.sh` exists:
   ```bash
   sudo bash scripts/install-cuda-path.sh
   ```

4. **PyTorch with CUDA** in the project venv:
   ```bash
   source venv/bin/activate
   . /etc/profile.d/cuda.sh
   bash scripts/install-pytorch-cuda-nvidia.sh
   ```

5. **Verify**:
   ```bash
   . /etc/profile.d/cuda.sh && python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
   You should see `CUDA: True` and device `Orin`.

## Ollama (local install)

Install Ollama locally as in [Jetson AI Lab – Ollama on Jetson](https://www.jetson-ai-lab.com/tutorials/ollama/):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the default model:

```bash
ollama pull qwen3:1.7b
```

### Ollama memory optimisation (8 GB Jetson)

On the Orin Nano, GPU and CPU share the same 7.6 GiB RAM. The project configures Ollama via a systemd drop-in (`/etc/systemd/system/ollama.service.d/gpu.conf`) for maximum GPU offload:

| Layer | Setting | Effect |
|-------|---------|--------|
| **systemd** | `OLLAMA_FLASH_ATTENTION=1` | Flash attention (dramatically less KV cache memory) |
| **systemd** | `OLLAMA_KV_CACHE_TYPE=q8_0` | 8-bit KV cache (halves memory vs f16) |
| **systemd** | `OLLAMA_NUM_PARALLEL=1` | No duplicate KV caches |
| **systemd** | `OLLAMA_MAX_LOADED_MODELS=1` | Only one model in GPU at a time |
| **systemd** | `OLLAMA_CONTEXT_LENGTH=8192` | Default context length (Qwen3:1.7b sweet spot) |
| **systemd** | `OLLAMA_GPU_OVERHEAD=1500000000` | Reserve ~1.5 GB for X11/GNOME/Cursor/YOLOE |
| **systemd** | `OLLAMA_KEEP_ALIVE=5m` | Unload model after 5 min idle |
| **Python** | `num_ctx=8192`, `num_predict=512` | Cap context and output per request |
| **Python** | `think=false` (chat) / `think=true` (tools) | Adaptive: Qwen3 needs reasoning for tool routing |
| **Python** | Intent-based tool routing | Only sends tool schemas when query matches keywords |
| **Python** | OOM recovery | On CUDA OOM: unload model, drop caches, retry with smaller ctx (8192→4096→2048→1024) |

Apply all settings:

```bash
sudo bash scripts/configure-ollama-systemd.sh
sudo systemctl daemon-reload && sudo systemctl restart ollama

# Verify: should show 100% GPU, context 8192
ollama ps
```

## First-time setup: download all models

After `pip install -r requirements.txt`, run the bootstrap script to download openWakeWord, Faster-Whisper (small), and the Piper voice. Optionally build the YOLOE-26N TensorRT engine.

```bash
source venv/bin/activate
bash scripts/bootstrap_models.sh
# Optional: build YOLOE-26N TensorRT engine (requires CUDA)
bash scripts/bootstrap_models.sh --with-yolo
```

Ensure Ollama has the default model: `ollama pull qwen3:1.7b`.

## Piper TTS (British male voice)

The **British male voice** (en_GB-alan-medium) is included under `models/voices/`:

- `models/voices/en_GB-alan-medium.onnx` – voice model
- `models/voices/en_GB-alan-medium.onnx.json` – config

To use another voice, set `JARVIS_TTS_VOICE` to the full path of a `.onnx` file, or add more voices from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).

## Vision (YOLOE-26N TensorRT)

To use TensorRT-accelerated **YOLOE-26N** (Ultralytics YOLOE, prompt-free nano):

```bash
source venv/bin/activate
. /etc/profile.d/cuda.sh
bash scripts/export_yolo_engine.sh
```

Output: `models/yoloe26n.engine`. Engine build can take several minutes on device.

**USB camera**: default is index `0` (`/dev/video0`). Set `JARVIS_CAMERA_DEVICE=/dev/video0` to force a device path, or `JARVIS_CAMERA_INDEX=1` for a second camera.

## Running Jarvis

```bash
source venv/bin/activate

# Validate config
python main.py --dry-run

# List audio devices
python main.py --test-audio

# Phase 1 test: wake word → play "Hello Sir"
python main.py --voice-only

# One-shot (no mic): text → LLM → TTS → play
python main.py --one-shot "What time is it?"

# Full loop: wake → STT → Ollama → TTS (with vision)
python main.py --e2e

# Agentic orchestrator: wake → STT → LLM with tools + context → TTS
python main.py --orchestrator

# Full-stack server: FastAPI + WebSocket bridge + PWA + orchestrator
python main.py --serve

# Live YOLOE camera preview (OpenCV window)
python main.py --yolo-visualize

# With status overlay
python main.py --e2e --gui
```

Stop with `Ctrl+C`.

### Server mode (`--serve`)

Runs FastAPI (uvicorn) + orchestrator in one process. Exposes:

- **WebSocket** (`/ws`) – bidirectional: send text queries from the PWA, receive status/reply/detections broadcasts
- **REST API** – `/health`, `/api/status`, `/api/stats`, `/api/reminders`
- **MJPEG stream** (`/stream`) – live camera + YOLOE overlay
- **PWA** – SvelteKit frontend served at `/` (build with `npm run build` in `pwa/`)

Connect from any device on the LAN: `http://<jetson-ip>:8000`.

### Orchestrator (agentic mode)

With `--orchestrator` or `--serve`, J.A.R.V.I.S. runs an async loop with **short- and long-term context**, **intent-based tool routing**, and **proactive** idle checks. Session summary and reminders are stored under `data/`.

**Intent-based tool routing**: instead of sending all tool schemas on every request (which wastes the 1.7b model's think budget on simple queries), the orchestrator uses deterministic keyword matching to decide which tools — if any — to include. This is the standard production pattern used by Alexa, Rasa, and Dialogflow for resource-constrained NLU.

| Query type | Route | Think | Latency |
|------------|-------|-------|---------|
| Greeting, status, time, goodnight | No tools, `think=false` | No | 0.5–0.7 s |
| "Tell me a joke" | `tell_joke` only, `think=true` | Yes | ~6 s |
| "Remind me to..." | `create_reminder` only, `think=true` | Yes | ~8 s |
| "Turn on sarcasm" | `toggle_sarcasm` only, `think=true` | Yes | ~4 s |
| "Re-scan the camera" | `vision_analyze` only, `think=true` | Yes | ~5 s |

Tools available to the LLM (via Ollama Hermes-style tool-calling):

| Tool | Description |
|------|-------------|
| `vision_analyze` | Re-scan camera with optional focus prompt |
| `create_reminder` | Save a reminder with optional time |
| `tell_joke` | Tell a witty one-liner (J.A.R.V.I.S.-style dry wit) |
| `toggle_sarcasm` | Toggle sarcasm mode |

Time, system stats, scene description, and pending reminders are **injected directly into the user context** — the LLM doesn't need tools for those.

## PWA Frontend

The SvelteKit PWA lives in `pwa/`. To build:

```bash
cd pwa
npm install
npm run build
cd ..
```

The built files in `pwa/build/` are served automatically by `--serve` mode. Components:

- **ChatPanel** – send text queries, view Jarvis replies
- **VoiceControls** – trigger wake/listen from the browser
- **CameraStream** – live MJPEG feed with YOLOE overlay
- **Dashboard** – Jetson GPU/CPU/thermal stats
- **Reminders** – create and view reminders
- **SettingsPanel** – sarcasm toggle, connection status

## Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Validate config and exit |
| `--test-audio` | List input devices and default sink/source |
| `--voice-only` | Wake word only; on trigger, play TTS "Hello Sir" |
| `--one-shot [PROMPT]` | Text → LLM → TTS → play (no mic needed) |
| `--e2e` | Full loop: wake → record → STT → LLM → TTS (with vision) |
| `--orchestrator` | Agentic loop: wake → STT → LLM with tools + context → TTS |
| `--serve` | Full-stack: FastAPI + WebSocket + PWA + orchestrator |
| `--yolo-visualize` | Live camera + YOLOE detections in OpenCV window |
| `--gui` | Show status overlay (Listening / Thinking / Speaking) |
| `--verbose` | Debug logging |

## Project layout

```
main.py              Entry point and CLI dispatcher
orchestrator.py      Async agentic loop (context, tools, proactive vision)
tools.py             Local tools (vision, status, time, reminders, joke, sarcasm)
memory.py            Session summary and persistence
config/              Settings (Jetson/Ollama tuning) and system prompts
audio/               Mic selection, recording, playback, Bluetooth hints
voice/               Wake word, STT (Faster-Whisper), TTS (Piper)
llm/                 Ollama client (OOM-hardened) and context builder
vision/              Camera, YOLOE-26N TensorRT, MediaPipe, scene description
utils/               Power mode, logging, reminders
gui/                 Optional Tkinter status overlay with vision preview
server/              FastAPI app, WebSocket bridge, MJPEG streaming
pwa/                 SvelteKit PWA frontend (Chat, Camera, Dashboard, Reminders)
scripts/             Setup and maintenance scripts
tests/               Unit and E2E tests (pytest)
models/              TTS voices, YOLOE TensorRT engines
data/                Session summaries and reminders (runtime)
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Default LLM model |
| `OLLAMA_NUM_CTX` | `8192` | Context window size (100% GPU sweet spot for Qwen3:1.7b) |
| `OLLAMA_NUM_CTX_MAX` | `8192` | Hard cap (12288 works but swap pressure; 16384 spills to CPU) |
| `OLLAMA_NUM_PREDICT` | `512` | Max output tokens (includes Qwen3 thinking tokens) |
| `OLLAMA_THINK` | `0` | Global think flag; `chat_with_tools` forces `true` when tools present |
| `OLLAMA_TEMPERATURE` | `0.6` | Sampling temperature |
| `JARVIS_CAMERA_INDEX` | `0` | Camera device index |
| `JARVIS_CAMERA_DEVICE` | (none) | Force camera device path |
| `JARVIS_TTS_VOICE` | `models/voices/en_GB-alan-medium.onnx` | Piper voice model path |
| `JARVIS_SERVE_HOST` | `0.0.0.0` | Server bind address |
| `JARVIS_SERVE_PORT` | `8000` | Server port |
| `JARVIS_CONTEXT_MAX_TURNS` | `4` | Max history turns in LLM context |

## Troubleshooting

- **Ollama OOM / cudaMalloc failed**: Run `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` before model load to reclaim buff/cache. Apply OOM-prevention settings with `sudo bash scripts/configure-ollama-systemd.sh`. The Python client also auto-recovers: on OOM it unloads the model, drops caches, and retries with smaller context.
- **Model only partially on GPU** (`ollama ps` shows CPU%): Drop caches (`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`), restart Ollama, then load model. Memory fragmentation from repeated context-size changes can cause spill. If persistent, reduce `OLLAMA_GPU_OVERHEAD` or `OLLAMA_NUM_CTX`. Close unnecessary desktop apps.
- **Slow responses (>10 s)**: Check `ollama ps` — model should be 100% GPU at 8192 ctx. If CPU% is >0, drop caches (`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`) and restart Ollama. For plain chat, ensure intent routing is sending no tools (queries without tool keywords should be 0.5–0.7 s).
- **Bluetooth mic not working**: Prefer HFP profile for the buds or use a USB microphone and keep A2DP for output.
- **Piper not found**: Ensure `piper-tts` is installed in the venv and the voice model exists at the configured path.
- **Ollama connection refused**: Start Ollama with `ollama serve` or check `systemctl status ollama`.
- **No camera**: Plug a USB UVC camera. Use `JARVIS_CAMERA_INDEX` or `JARVIS_CAMERA_DEVICE` to select the device.

## Testing

```bash
source venv/bin/activate

# Lint
ruff check .

# Unit tests
pytest tests/unit/

# E2E tests (requires hardware)
pytest tests/e2e/ -m e2e

# Quick smoke test
python main.py --dry-run
python main.py --one-shot "Say hello."
```
