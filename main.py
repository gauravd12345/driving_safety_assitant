import cv2
from collections import Counter

import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM


VIDEO_PATH = "/Users/gauravd/Desktop/driving_safety_assitant/test.mp4"

YOLO_MODEL = "yolov8n.pt"

LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():

    ############################################################
    # Load models
    ############################################################

    print("Loading YOLO...")
    yolo = YOLO(YOLO_MODEL)

    print("Loading Qwen...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
    ).to("mps")

    ############################################################
    # Open video
    ############################################################

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps)          # one frame every second

    frame_number = 0

    frame_summaries = []

    ############################################################
    # Process video
    ############################################################

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

            class_id = int(cls)

            class_name = yolo.names[class_id]

            counts[class_name] += 1

        summary = f"{timestamp:.1f}s: "

        if len(counts) == 0:
            summary += "No objects detected."
        else:
            summary += ", ".join(
                f"{v} {k}"
                for k, v in counts.items()
            )

        print(summary)

        frame_summaries.append(summary)

        frame_number += 1

    cap.release()

    ############################################################
    # Ask Qwen
    ############################################################

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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("mps")

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print("\n")
    print("=" * 60)
    print("AI REPORT")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()