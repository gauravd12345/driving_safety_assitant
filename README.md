
# Driving Safety Assistant

An AI-powered driving safety assistant that analyzes dashcam footage using computer vision and large language models to detect potential driving incidents and generate natural-language safety reports.

It ships with a web app: upload a video and watch the detections stream into a live "terminal" beside the player, then read the AI's final safety verdict.

## Run the web app

```bash
source .venv/bin/activate
python server.py
```

Then open http://127.0.0.1:8000 in your browser, drop in a video, and click **Analyze video**.

- **Left panel:** your uploaded video.
- **Right panel:** a live terminal streaming per-second object detections (via WebSocket).
- **Bottom:** the AI safety verdict once processing finishes.

The original CLI version still works too: `python main.py` (it analyzes `test.mp4` and prints to the terminal).

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install ultralytics transformers accelerate pillow opencv-python fastapi uvicorn python-multipart sentencepiece safetensors
```

### 3. Verify YOLO

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("YOLO loaded!")
```

The pretrained YOLO model will automatically download on the first run and will be cached locally.

### 4. Verify Qwen

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to("mps")

print("Qwen loaded!")
```

The model will automatically download on the first run and will be cached locally for future runs.
