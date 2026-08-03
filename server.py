import asyncio
import queue
import shutil
import threading
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

YOLO_MODEL = "yolov8n.pt"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Models (loaded once at startup, shared across requests)
# ---------------------------------------------------------------------------

MODELS = {}


def load_models():
    if MODELS:
        return MODELS

    MODELS["yolo"] = YOLO(YOLO_MODEL)

    MODELS["tokenizer"] = AutoTokenizer.from_pretrained(LLM_MODEL)

    dtype = torch.float16 if DEVICE != "cpu" else torch.float32

    MODELS["llm"] = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        dtype=dtype,
    ).to(DEVICE)

    return MODELS


# ---------------------------------------------------------------------------
# Analysis (blocking) - streams messages back through the `emit` callback
# ---------------------------------------------------------------------------


def analyze_video(video_path: str, emit):
    """Run YOLO + Qwen over a video, streaming progress via emit(dict)."""

    emit({"type": "log", "text": f"Using device: {DEVICE}"})
    emit({"type": "log", "text": "Loading models..."})

    models = load_models()
    yolo = models["yolo"]
    tokenizer = models["tokenizer"]
    llm = models["llm"]

    emit({"type": "log", "text": "Models ready."})

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_interval = max(int(fps), 1)  # one frame every second

    emit(
        {
            "type": "log",
            "text": f"Video opened: {fps:.1f} fps, {total_frames} frames "
            f"(~{total_frames / fps:.1f}s). Sampling 1 frame/sec.",
        }
    )
    emit({"type": "log", "text": "-" * 48})

    frame_number = 0
    frame_summaries = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_number % frame_interval != 0:
            frame_number += 1
            continue

        timestamp = frame_number / fps

        results = yolo(frame, verbose=False)[0]

        counts = Counter()
        for cls in results.boxes.cls:
            class_name = yolo.names[int(cls)]
            counts[class_name] += 1

        summary = f"{timestamp:.1f}s: "
        if len(counts) == 0:
            summary += "No objects detected."
        else:
            summary += ", ".join(f"{v} {k}" for k, v in counts.items())

        frame_summaries.append(summary)

        progress = (frame_number / total_frames) if total_frames else 0.0
        emit({"type": "log", "text": summary, "progress": progress})

        frame_number += 1

    cap.release()

    emit({"type": "log", "text": "-" * 48})
    emit({"type": "log", "text": "Detection complete. Asking the AI for a verdict...", "progress": 1.0})

    prompt = f"""
You are an AI driving safety assistant.

Below is a timeline of object detections from a dashcam.

Determine:

- Any potential safety concerns
- Whether anything unusual occurred
- General driving conditions

Timeline:

{chr(10).join(frame_summaries)}

Produce a concise report.
"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    # Stream the report token-by-token so the UI shows it being written live.
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    def _generate():
        with torch.inference_mode():
            llm.generate(**inputs, max_new_tokens=300, streamer=streamer)

    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    emit({"type": "verdict_start"})
    for chunk in streamer:
        if chunk:
            emit({"type": "verdict_chunk", "text": chunk})

    gen_thread.join()
    emit({"type": "done"})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the models so the first analysis isn't slow.
    threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(title="Driving Safety Assistant", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.post("/upload")
async def upload(file: UploadFile):
    job_id = uuid.uuid4().hex
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{job_id}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"job_id": dest.name}


@app.get("/video/{name}")
def video(name: str):
    path = UPLOAD_DIR / name
    if not path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(path)


@app.websocket("/ws/{name}")
async def ws_analyze(websocket: WebSocket, name: str):
    await websocket.accept()

    path = UPLOAD_DIR / name
    if not path.exists():
        await websocket.send_json({"type": "error", "text": "Video not found."})
        await websocket.close()
        return

    loop = asyncio.get_event_loop()
    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            analyze_video(str(path), q.put)
        except Exception as exc:  # surface errors to the terminal UI
            q.put({"type": "error", "text": f"Error: {exc}"})
        finally:
            q.put(None)  # sentinel

    threading.Thread(target=worker, daemon=True).start()

    try:
        while True:
            msg = await loop.run_in_executor(None, q.get)
            if msg is None:
                break
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
