"""FastAPI application: health, REST API, MJPEG stream, and WebSocket endpoint.

Run via ``python main.py --serve`` (starts uvicorn + orchestrator in one process)
or standalone: ``python -m server`` for the HTTP/WS layer only.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from config import settings
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.bridge import bridge
from server.streaming import mjpeg_generator, mjpeg_raw_generator

logger = logging.getLogger(__name__)

# Path to the built PWA (pwa/build/)
_PWA_DIR = Path(settings.PROJECT_ROOT) / "pwa" / "build"


# ── Lifespan ──────────────────────────────────────────────────────────────

_vision_broadcast_task: asyncio.Task | None = None


async def _vision_broadcast_loop() -> None:
    """Background task: continuously run enriched vision and broadcast to PWA clients.

    Sends tracked objects, vitals, threat every VISION_BROADCAST_INTERVAL seconds.
    Sends full hologram (point cloud) every VISION_BROADCAST_DEPTH_EVERY cycles.
    Only runs when at least one WebSocket client is connected.
    """
    interval = getattr(settings, "VISION_BROADCAST_INTERVAL", 5)
    depth_every = getattr(settings, "VISION_BROADCAST_DEPTH_EVERY", 3)
    cycle = 0

    while True:
        try:
            await asyncio.sleep(interval)

            # Skip if no clients connected
            with bridge._clients_lock:
                n_clients = len(bridge._clients)
            if n_clients == 0:
                continue

            loop = asyncio.get_running_loop()

            # Run vision pipeline in executor (blocking I/O)
            try:
                from tools import vision_analyze_full

                data = await loop.run_in_executor(None, vision_analyze_full, None)
            except Exception as exc:
                logger.debug("Vision broadcast: pipeline error: %s", exc)
                continue

            # Always broadcast detections + tracked objects
            tracked = data.get("tracked", [])
            description = data.get("description", "")
            await bridge.broadcast({
                "type": "scan_result",
                "detections": data.get("detections", []),
                "tracked": tracked,
                "description": description,
            })

            # Always broadcast vitals if available
            vitals = data.get("vitals")
            if vitals is not None:
                await bridge.broadcast({
                    "type": "vitals",
                    "data": {
                        "fatigue": getattr(vitals, "fatigue_level", "unknown"),
                        "posture": getattr(vitals, "posture_label", "unknown"),
                        "heart_rate": getattr(vitals, "heart_rate_bpm", None),
                        "hr_confidence": getattr(vitals, "heart_rate_confidence", 0),
                        "alerts": getattr(vitals, "alerts", []),
                    },
                })

            # Always broadcast threat if available
            threat = data.get("threat")
            if threat is not None:
                await bridge.broadcast({
                    "type": "threat",
                    "data": {
                        "level": getattr(threat, "label", "clear"),
                        "score": getattr(threat, "level", 0) / 10.0,
                        "summary": getattr(threat, "recommendation", ""),
                    },
                })

            # Broadcast hologram (point cloud) every Nth cycle
            if cycle % depth_every == 0:
                point_cloud = data.get("point_cloud", [])
                await bridge.broadcast({
                    "type": "hologram",
                    "data": {
                        "point_cloud": point_cloud[:3000],
                        "tracked_objects": tracked,
                        "description": description,
                    },
                })

            cycle += 1

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("Vision broadcast loop error: %s", exc)
            await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set event loop on the bridge so threadsafe broadcasts work.

    Starts the continuous vision broadcast background task.
    On shutdown, release the shared camera so V4L2 doesn't leak the device.
    """
    global _vision_broadcast_task
    loop = asyncio.get_running_loop()
    bridge.set_loop(loop)
    _vision_broadcast_task = asyncio.create_task(_vision_broadcast_loop())
    logger.info("Jarvis server started on %s:%s", settings.JARVIS_SERVE_HOST, settings.JARVIS_SERVE_PORT)
    logger.info("Vision broadcast: every %ds, depth every %d cycles",
                getattr(settings, "VISION_BROADCAST_INTERVAL", 5),
                getattr(settings, "VISION_BROADCAST_DEPTH_EVERY", 3))
    yield
    logger.info("Jarvis server shutting down")
    if _vision_broadcast_task:
        _vision_broadcast_task.cancel()
        try:
            await _vision_broadcast_task
        except asyncio.CancelledError:
            pass
    try:
        from vision.shared import release_camera
        release_camera()
    except Exception:
        pass


app = FastAPI(
    title="Jarvis API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow PWA origin(s); permissive in dev (same LAN)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health / readiness ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Subsystem health: CUDA, camera, YOLOE engine."""
    from vision.shared import check_cuda, get_camera, get_yolo

    cuda_ok, cuda_msg = check_cuda()
    return {
        "status": "ok",
        "cuda": cuda_msg,
        "camera": get_camera() is not None,
        "yolo_engine": get_yolo()[0] is not None,
    }


@app.get("/api/status")
async def api_status():
    """Jetson summary: power, stats, thermal."""
    from tools import run_tool
    status = await asyncio.get_running_loop().run_in_executor(
        None, run_tool, "get_jetson_status", {}
    )
    return {"status": status}


# ── System stats ──────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    """Wrap utils.power for the HUD dashboard."""
    try:
        from utils.power import get_system_stats, get_thermal_warning
        stats = await asyncio.get_running_loop().run_in_executor(None, get_system_stats)
        thermal = await asyncio.get_running_loop().run_in_executor(None, get_thermal_warning)
        return {"stats": stats, "thermal": thermal}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Reminders CRUD ────────────────────────────────────────────────────────

@app.get("/api/reminders")
async def api_get_reminders():
    from utils.reminders import load_reminders
    data_dir = Path(settings.DATA_DIR)
    reminders = load_reminders(data_dir)
    return {"reminders": reminders}


@app.post("/api/reminders")
async def api_create_reminder(body: dict):
    from utils.reminders import add_reminder
    data_dir = Path(settings.DATA_DIR)
    text = body.get("text", "")
    time_str = body.get("time_str", body.get("time", ""))
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    add_reminder(data_dir, text, time_str)
    return {"ok": True, "text": text, "time_str": time_str}


@app.patch("/api/reminders/{index}")
async def api_toggle_reminder(index: int):
    from utils.reminders import toggle_reminder
    data_dir = Path(settings.DATA_DIR)
    try:
        done = toggle_reminder(data_dir, index)
        return {"ok": True, "index": index, "done": done}
    except IndexError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.delete("/api/reminders/{index}")
async def api_delete_reminder(index: int):
    from utils.reminders import delete_reminder
    data_dir = Path(settings.DATA_DIR)
    try:
        removed = delete_reminder(data_dir, index)
        return {"ok": True, "removed": removed}
    except IndexError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


# ── Depth map endpoint ────────────────────────────────────────────────────

@app.get("/api/depth")
async def api_depth():
    """Return the latest depth map as a 16-bit PNG (or JSON metadata if unavailable)."""
    try:
        from vision.shared import read_frame, run_depth_shared

        frame = await asyncio.get_running_loop().run_in_executor(None, read_frame)
        if frame is None:
            return JSONResponse({"error": "No frame available"}, status_code=503)

        depth_map = await asyncio.get_running_loop().run_in_executor(None, run_depth_shared, frame)
        if depth_map is None:
            return JSONResponse({"error": "Depth estimation unavailable (engine not loaded)"}, status_code=503)

        import cv2
        import numpy as np

        # Convert 0-1 float32 to 16-bit PNG
        depth_16 = (depth_map * 65535).astype(np.uint16)
        ok, buf = cv2.imencode(".png", depth_16)
        if not ok:
            return JSONResponse({"error": "Encoding failed"}, status_code=500)
        from fastapi.responses import Response
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Vitals endpoint ──────────────────────────────────────────────────────

@app.get("/api/vitals")
async def api_vitals():
    """Return latest vitals snapshot as JSON."""
    try:
        from vision.shared import get_vitals_analyzer

        analyzer = get_vitals_analyzer()
        if analyzer is None:
            return {"vitals": None, "message": "Vitals analyzer not initialized"}

        result = analyzer.last_result
        return {
            "vitals": {
                "fatigue_level": result.fatigue_level,
                "eye_aspect_ratio": result.eye_aspect_ratio,
                "blink_rate_per_min": result.blink_rate_per_min,
                "posture_score": result.posture_score,
                "posture_label": result.posture_label,
                "heart_rate_bpm": result.heart_rate_bpm,
                "heart_rate_confidence": result.heart_rate_confidence,
                "alerts": result.alerts,
            }
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Hologram endpoint ────────────────────────────────────────────────────

@app.get("/api/hologram")
async def api_hologram():
    """Trigger hologram generation and return point cloud + tracked objects.

    Always returns 200 with best-effort data; ``description`` explains
    any degraded state (no camera, no depth engine, etc.).
    """
    try:
        from tools import vision_analyze_full

        data = await asyncio.get_running_loop().run_in_executor(
            None, vision_analyze_full, None,
        )
        return {
            "point_cloud": data.get("point_cloud", [])[:3000],
            "tracked_objects": data.get("tracked", []),
            "description": data.get("description", ""),
        }
    except Exception as e:
        logger.warning("api_hologram error: %s", e)
        # Return empty but valid hologram payload so the frontend always works
        return {
            "point_cloud": [],
            "tracked_objects": [],
            "description": f"Hologram unavailable: {e}",
        }


# ── MJPEG camera stream ──────────────────────────────────────────────────

@app.get("/stream")
async def stream():
    """Live camera + YOLOE overlay as MJPEG (multipart/x-mixed-replace)."""
    return StreamingResponse(
        mjpeg_generator(fps=10),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/raw")
async def stream_raw():
    """Raw camera MJPEG without detection annotations (for HUD overlay).

    The client-side HUD draws all tracking/detection graphics on a canvas
    layer, so the base feed must be clean.
    """
    return StreamingResponse(
        mjpeg_raw_generator(fps=10),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    bridge.add_client(ws)
    try:
        while True:
            raw = await ws.receive_text()
            await bridge.handle_client_message(raw)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WS error: %s", e)
    finally:
        bridge.remove_client(ws)


# ── PWA static files (must be last) ──────────────────────────────────────

if _PWA_DIR.is_dir():
    # Mount the _app immutable assets directory for hashed JS/CSS
    _app_dir = _PWA_DIR / "_app"
    if _app_dir.is_dir():
        app.mount("/_app", StaticFiles(directory=str(_app_dir)), name="pwa-app")

    # Serve known static files at root (manifest, sw, icons, etc.)
    @app.get("/manifest.webmanifest")
    async def pwa_manifest():
        return FileResponse(_PWA_DIR / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/sw.js")
    async def pwa_sw():
        return FileResponse(_PWA_DIR / "sw.js", media_type="application/javascript")

    @app.get("/registerSW.js")
    async def pwa_register_sw():
        return FileResponse(_PWA_DIR / "registerSW.js", media_type="application/javascript")

    @app.get("/workbox-{rest_of_path:path}")
    async def pwa_workbox(rest_of_path: str):
        return FileResponse(_PWA_DIR / f"workbox-{rest_of_path}", media_type="application/javascript")

    @app.get("/favicon.svg")
    async def pwa_favicon():
        return FileResponse(_PWA_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/icon-192.png")
    async def pwa_icon_192():
        return FileResponse(_PWA_DIR / "icon-192.png", media_type="image/png")

    @app.get("/icon-512.png")
    async def pwa_icon_512():
        return FileResponse(_PWA_DIR / "icon-512.png", media_type="image/png")

    # SPA catch-all: serve index.html for any unmatched GET request
    @app.get("/{full_path:path}")
    async def pwa_spa_fallback(full_path: str):
        # If a real file exists in build/, serve it
        candidate = _PWA_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        # Otherwise serve index.html (SPA fallback)
        return FileResponse(_PWA_DIR / "index.html", media_type="text/html")
else:
    logger.warning("PWA build not found at %s – run 'npm run build' in pwa/", _PWA_DIR)
