# Jarvis plan verification

Verified against [.cursor/plans/jarvis_jetson_implementation_2e9bf9f7.plan.md](.cursor/plans/jarvis_jetson_implementation_2e9bf9f7.plan.md).

**Jarvis uses local Ollama only — no Docker.** Install with `scripts/install-ollama.sh`, start with `scripts/start-ollama.sh` (or systemd). API at `http://127.0.0.1:11434`.

## Ollama (local install – plan §2.4)

- **Install**: `bash scripts/install-ollama.sh` (official install script).
- **Start with GPU**: `bash scripts/start-ollama.sh` (sets `OLLAMA_NUM_GPU=1`). API at `http://127.0.0.1:11434`.
- **Pull model**: `ollama pull llama3.2:1b` (default in config) or `llama3.2:3b`. Config: `OLLAMA_MODEL=llama3.2:1b` in `config/settings.py`.
- **Systemd (GPU)**: `sudo scripts/configure-ollama-systemd.sh` then `sudo systemctl daemon-reload && sudo systemctl restart ollama` — ensures `OLLAMA_NUM_GPU=1` and `LimitMEMLOCK=infinity` for the service.
- **Before first GPU use (optional)**: `sudo scripts/prepare-ollama-gpu.sh` to free GPU/RAM (jetson_clocks, drop caches); then start Ollama.
- **Verified**: Ollama API at 127.0.0.1:11434 returns 200; `llama3.2:1b` present in `/api/tags`; systemd `ollama.service.d/gpu.conf` applied.

## YOLOE (vision – plan §2.5, Phase 3)

- **Export TensorRT engine**: With venv active and CUDA in PATH: `bash scripts/export_yolo_engine.sh`. Produces `models/yoloe26n.engine` (YOLOE-26N prompt-free).
- **Config**: `YOLOE_ENGINE_PATH` in `config/settings.py` points to `models/yoloe26n.engine`. Vision pipeline uses YOLOE-26N for object detection and MediaPipe for face/pose; scene description is fed into LLM context.

## Plan structure and code

- **Layout**: `main.py`, `config/`, `audio/`, `voice/`, `llm/`, `vision/`, `utils/`, `gui/`, `tests/`, `scripts/`, `requirements.txt`, `README.md` match plan §3.
- **Lint**: `ruff check .` — all checks passed.
- **Tests**: `pytest tests/` (unit + e2e). `pytest tests/ -m e2e` for E2E only.
- **Entry**: `python main.py --help` — options include --dry-run, --e2e, --no-vision, --one-shot, --test-audio, --gui.
- **Dry run**: `python main.py --dry-run` — config OK, ollama base URL and model reported.
- **One-shot**: `python main.py --one-shot "Say hello..."` — LLM (Ollama) → Piper TTS → played.
- **E2E**: `python main.py --e2e --no-vision` — full loop (wake word, STT, LLM, TTS). With vision: `python main.py --e2e` (requires camera and `models/yoloe26n.engine`; build with `bash scripts/export_yolo_engine.sh`).

## Summary

| Item                         | Status |
|------------------------------|--------|
| Ollama (local, no Docker)    | OK — API 127.0.0.1:11434, llama3.2:1b available |
| Ollama systemd GPU (sudo)    | OK — gpu.conf applied, service restarted |
| YOLO (export + vision path)  | OK — scripts/export_yolo_engine.sh → models/yolov8n.engine |
| Repo structure               | OK     |
| ruff                         | OK     |
| pytest                       | OK     |
| main.py --help / --dry-run   | OK     |
| main.py --one-shot           | OK (LLM → TTS) |
| main.py --e2e                | OK (full loop ± vision) |
| Phases 1–5                   | Complete (voice, LLM, vision, GUI, error handling) |

## Production readiness

- **Bootstrap**: `bash scripts/bootstrap_models.sh` to download openWakeWord (hey_jarvis), Faster-Whisper (small), Piper voice. Use `--with-yolo` to build `models/yolov8n.engine`.
- **Ollama**: `ollama pull llama3.2:1b`; ensure service runs (systemd or `bash scripts/start-ollama.sh`).
- **Validation**: `ruff check .`, `pytest tests/`, `pytest tests/ -m e2e`, `python main.py --dry-run`, `python main.py --one-shot "Say hello."`, `python main.py --e2e --no-vision`.
