# Jetson Orin Nano (8GB) – Models

## YOLOE-26N TensorRT (vision)

Jarvis uses **YOLOE-26N** (2026 Ultralytics YOLOE, prompt-free nano) for vision. Per [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) and [TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/), export is done **on-device** with CUDA. No workarounds.

1. **Prerequisites**: CUDA in PATH (`/etc/profile.d/cuda.sh`), PyTorch + torchvision for Jetson (see README “CUDA and PyTorch”), and `ultralytics` in the project venv.
2. **Export** (from project root, venv active):
   ```bash
   . /etc/profile.d/cuda.sh
   bash scripts/export_yolo_engine.sh
   ```
   The script loads `yoloe-26n-seg-pf.pt` (YOLOE class), exports to TensorRT with `format=engine device=0 half=True workspace=4`, and saves `models/yoloe26n.engine`.
3. **Inference**: `vision/detector_yolo.py` loads the `.engine` with `YOLO(engine_path)` and runs inference on GPU. The engine is required for the vision pipeline; `--e2e` without `--no-vision` will exit if `models/yoloe26n.engine` is missing.

References: [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/), [Ultralytics Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/), [TensorRT export](https://docs.ultralytics.com/integrations/tensorrt/).

---

## This project: local Ollama (GPU)

**Jarvis uses local Ollama** (no Docker). Install with `scripts/install-ollama.sh`, start with `scripts/start-ollama.sh` (GPU via `OLLAMA_NUM_GPU=1`).

- **Default in code**: `llama3.2:1b` (set in `config/settings.py`).
- **Pull**: `ollama pull llama3.2:1b` (or `llama3.2:3b` for better quality if RAM allows).
- **Alternatives**: `qwen2.5:1.5b` (if GPU memory is tight).

If the model fails to load with GPU OOM, free memory (e.g. `sudo scripts/prepare-ollama-gpu.sh`) and restart Ollama, or use a smaller model.

## Other models (Jetson AI Lab)

Per [Jetson AI Lab – Models](https://www.jetson-ai-lab.com/models), you can also use:

| Model            | RAM  | Ollama pull     | Notes (AI Lab)        |
|------------------|------|------------------|------------------------|
| Gemma 3 270M     | 1 GB | `gemma3:270m`    | Ultra-compact         |
| Gemma 3 1B       | 2 GB | `gemma3:1b`      | Good balance          |
| Llama 3.2 3B     | 4 GB | `llama3.2:3b`    | Suited for Orin Nano  |

## Override model

```bash
export OLLAMA_MODEL=llama3.2:1b   # or llama3.2:3b / qwen2.5:1.5b
python main.py --one-shot "What time is it?"
```

## References

- [Jetson AI Lab – Models](https://www.jetson-ai-lab.com/models)
- [Jetson AI Lab – Ollama](https://www.jetson-ai-lab.com/tutorials/ollama/)
- [Ollama library](https://ollama.com/library)
