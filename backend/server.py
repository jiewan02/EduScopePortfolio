"""
FastAPI server for the EduScope classroom engagement monitor.
Run from the webapp/ directory:
    uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import datetime
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import EngagementEngine

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="EduScope – Classroom Engagement Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).parent.parent / "frontend"
_LOG_DIR  = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

# ── Engine singleton ──────────────────────────────────────────────────────────
_engine:     EngagementEngine | None = None
_engine_lock = threading.Lock()
_executor    = ThreadPoolExecutor(max_workers=2)


def get_engine() -> EngagementEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = EngagementEngine()
    return _engine


@app.on_event("startup")
async def _preload():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, get_engine)


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return (_FRONTEND / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def api_status():
    return {"ready": _engine is not None, "version": "1.0.0"}


@app.post("/api/reset")
async def api_reset():
    if _engine:
        _engine.reset()
    return {"reset": True}


@app.get("/api/sessions")
async def api_sessions():
    sessions = []
    for path in sorted(_LOG_DIR.glob("session_*.jsonl")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        sessions.append({
            "file":   path.name,
            "frames": len(lines),
            "mtime":  datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        })
    return {"sessions": sessions}


@app.get("/api/sessions/{filename}")
async def api_session_detail(filename: str):
    path = _LOG_DIR / filename
    if not path.exists() or not filename.endswith(".jsonl"):
        return JSONResponse({"error": "not found"}, status_code=404)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines() if l]
    return {"filename": filename, "frames": len(lines), "data": lines}


# ── WebSocket stream ──────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    engine     = get_engine()
    session_id = uuid.uuid4().hex[:8]
    log_path   = _LOG_DIR / f"session_{session_id}.jsonl"

    print(f"[WS] Session {session_id} started. Logging → {log_path}")

    try:
        while True:
            # Client sends raw JPEG bytes
            jpeg_bytes = await websocket.receive_bytes()

            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Run engine in thread pool (heavy CPU work)
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor, engine.process_frame, frame.copy()
            )

            # Append to session log
            log_entry = {
                "ts":             datetime.datetime.now().isoformat(),
                "elapsed":        result["elapsed"],
                "avg_engagement": result["avg_engagement"],
                "persons":        result["persons"],
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Send result to client (JSON only — no image encoding)
            await websocket.send_json({
                "persons":        result["persons"],
                "objects":        result["objects"],
                "frame_size":     result["frame_size"],
                "fps":            result["fps"],
                "elapsed":        result["elapsed"],
                "avg_engagement": result["avg_engagement"],
                "num_persons":    result["num_persons"],
                "session_id":     session_id,
            })

    except WebSocketDisconnect:
        print(f"[WS] Session {session_id} disconnected.")
    except Exception as exc:
        print(f"[WS] Session {session_id} error: {exc}")
        try:
            await websocket.close()
        except Exception:
            pass
