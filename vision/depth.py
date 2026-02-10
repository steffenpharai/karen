"""DepthAnything V2 Small: monocular depth estimation for 3D scene understanding.

Loads a TensorRT FP16 engine exported on-device via ``scripts/export_depth_engine.sh``.
Falls back to PyTorch/ONNX if the engine is not available.  The depth map is
relative (0-1, near-far), not metric — suitable for 3D point cloud generation
and object distance ranking, not absolute measurements.

Memory: ~50 MB GPU for TensorRT FP16 engine (Small variant).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default input size for DepthAnything V2 Small
_DEPTH_INPUT_SIZE = (518, 518)  # H, W — model's native resolution


def load_depth_model(engine_path: str | Path):
    """Load DepthAnything V2 Small.

    Attempts TensorRT engine first, then falls back to torch hub.
    Returns model object or None.
    """
    path = Path(engine_path) if engine_path else None

    # Try TensorRT engine
    if path and path.exists() and path.suffix == ".engine":
        try:
            import pycuda.autoinit  # noqa: F401
            import tensorrt as trt

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                logger.warning("TensorRT engine deserialization returned None")
                return None
            context = engine.create_execution_context()
            logger.info("Depth engine loaded (TensorRT): %s", path)
            return {"type": "tensorrt", "engine": engine, "context": context}
        except ImportError:
            logger.info("TensorRT/PyCUDA not available, trying torch fallback")
        except Exception as e:
            logger.warning("TensorRT depth engine load failed: %s", e)

    # Try ONNX path
    if path and path.exists() and path.suffix == ".onnx":
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(str(path), providers=providers)
            logger.info("Depth model loaded (ONNX): %s", path)
            return {"type": "onnx", "session": session}
        except Exception as e:
            logger.warning("ONNX depth model load failed: %s", e)

    # Fallback: try torch hub (downloads ~25MB on first use)
    try:
        import torch

        model = torch.hub.load(
            "LiheYoung/depth-anything",
            "DepthAnything_V2_Small",
            pretrained=True,
            trust_repo=True,
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda().half()
        logger.info("Depth model loaded via torch hub (DepthAnything V2 Small)")
        return {"type": "torch", "model": model}
    except Exception as e:
        logger.warning("Depth model torch hub fallback failed: %s", e)

    return None


def _preprocess(frame, target_size: tuple[int, int] = _DEPTH_INPUT_SIZE) -> np.ndarray:
    """Resize and normalise BGR frame for depth model input.

    Returns CHW float32 array normalised to [0, 1] with ImageNet stats.
    """
    import cv2

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = img.astype(np.float32) / 255.0
    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW, add batch dim
    img = img.transpose(2, 0, 1)
    return img


def estimate_depth(model_info: dict, frame) -> np.ndarray | None:
    """Run depth estimation on a BGR frame.

    Returns HxW float32 depth map normalised to [0, 1] (0=near, 1=far),
    resized to the original frame dimensions.  Returns None on failure.
    """
    if model_info is None or frame is None:
        return None

    h, w = frame.shape[:2]
    model_type = model_info.get("type")

    try:
        preprocessed = _preprocess(frame)

        if model_type == "tensorrt":
            depth = _infer_tensorrt(model_info, preprocessed)
        elif model_type == "onnx":
            depth = _infer_onnx(model_info, preprocessed)
        elif model_type == "torch":
            depth = _infer_torch(model_info, preprocessed)
        else:
            return None

        if depth is None:
            return None

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        # Resize to original frame size
        import cv2

        depth = cv2.resize(depth.astype(np.float32), (w, h))
        return depth

    except Exception as e:
        logger.warning("Depth estimation failed: %s", e)
        return None


def _infer_tensorrt(model_info: dict, preprocessed: np.ndarray) -> np.ndarray | None:
    """Run TensorRT inference."""
    try:
        import pycuda.driver as cuda

        context = model_info["context"]
        engine = model_info["engine"]

        # Allocate buffers
        input_data = np.ascontiguousarray(preprocessed[np.newaxis], dtype=np.float32)
        d_input = cuda.mem_alloc(input_data.nbytes)

        # Get output shape from engine
        output_shape = context.get_tensor_shape(engine.get_tensor_name(1))
        output_data = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output_data.nbytes)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, input_data, stream)

        context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
        context.set_tensor_address(engine.get_tensor_name(1), int(d_output))
        context.execute_async_v3(stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()

        # Output is typically (1, H, W) or (1, 1, H, W)
        depth = output_data.squeeze()
        return depth
    except Exception as e:
        logger.warning("TensorRT depth inference failed: %s", e)
        return None


def _infer_onnx(model_info: dict, preprocessed: np.ndarray) -> np.ndarray | None:
    """Run ONNX Runtime inference."""
    try:
        session = model_info["session"]
        input_name = session.get_inputs()[0].name
        input_data = preprocessed[np.newaxis].astype(np.float32)
        outputs = session.run(None, {input_name: input_data})
        depth = outputs[0].squeeze()
        return depth
    except Exception as e:
        logger.warning("ONNX depth inference failed: %s", e)
        return None


def _infer_torch(model_info: dict, preprocessed: np.ndarray) -> np.ndarray | None:
    """Run PyTorch inference."""
    try:
        import torch

        model = model_info["model"]
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        tensor = torch.from_numpy(preprocessed[np.newaxis]).to(device=device, dtype=dtype)

        with torch.no_grad():
            output = model(tensor)

        if isinstance(output, dict):
            depth = output.get("depth", output.get("out"))
        elif isinstance(output, (list, tuple)):
            depth = output[0]
        else:
            depth = output

        return depth.squeeze().cpu().float().numpy()
    except Exception as e:
        logger.warning("Torch depth inference failed: %s", e)
        return None


# ── Utility functions ─────────────────────────────────────────────────


def depth_at_boxes(
    depth_map: np.ndarray | None,
    detections: list[dict],
) -> list[float | None]:
    """Compute median depth value within each detection bounding box.

    Parameters
    ----------
    depth_map : HxW float32 depth map (0=near, 1=far), or None
    detections : list of dicts with ``xyxy`` key

    Returns
    -------
    list of depth values (0-1) or None per detection
    """
    if depth_map is None:
        return [None] * len(detections)

    h, w = depth_map.shape[:2]
    results = []
    for det in detections:
        xyxy = det.get("xyxy")
        if xyxy is None or len(xyxy) != 4:
            results.append(None)
            continue
        x1 = max(0, int(xyxy[0]))
        y1 = max(0, int(xyxy[1]))
        x2 = min(w, int(xyxy[2]))
        y2 = min(h, int(xyxy[3]))
        if x2 <= x1 or y2 <= y1:
            results.append(None)
            continue
        roi = depth_map[y1:y2, x1:x2]
        results.append(float(np.median(roi)))
    return results


def generate_point_cloud(
    frame,
    depth_map: np.ndarray | None,
    sample_step: int = 8,
    max_points: int = 5000,
) -> list[dict]:
    """Generate a sparse 3D point cloud from frame + depth map.

    Each point has x, y, z (from depth), r, g, b.  Suitable for Three.js
    hologram rendering in the PWA.

    Parameters
    ----------
    frame : BGR numpy array
    depth_map : HxW float32 (0=near, 1=far)
    sample_step : pixel stride for sampling (higher = sparser, faster)
    max_points : cap on returned points

    Returns
    -------
    list of dicts: [{x, y, z, r, g, b}, ...]
    """
    if frame is None or depth_map is None:
        return []

    h, w = frame.shape[:2]
    dh, dw = depth_map.shape[:2]

    # Depth map may be different resolution; use frame coords
    import cv2

    if (dh, dw) != (h, w):
        depth_map = cv2.resize(depth_map, (w, h))

    points = []
    # Approximate camera intrinsics (assume ~60 deg horizontal FOV)
    fx = w / (2.0 * np.tan(np.radians(30)))
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    for y_px in range(0, h, sample_step):
        for x_px in range(0, w, sample_step):
            z = float(depth_map[y_px, x_px])
            if z < 0.01:
                continue  # skip near-zero depth
            # Convert depth (0-1 relative) to pseudo-metric (scale arbitrarily)
            z_m = z * 10.0  # 0-10m range
            x_m = (x_px - cx) * z_m / fx
            y_m = (y_px - cy) * z_m / fy
            b, g, r = frame[y_px, x_px]
            points.append({
                "x": round(x_m, 3),
                "y": round(-y_m, 3),  # flip Y for 3D convention
                "z": round(z_m, 3),
                "r": int(r), "g": int(g), "b": int(b),
            })
            if len(points) >= max_points:
                return points

    return points
