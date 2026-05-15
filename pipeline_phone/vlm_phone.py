import base64
import json
import cv2
import numpy as np
import requests
import re

from utils.logger import get_logger

logger = get_logger("vlm_phone")

LOCAL_MODEL_ID = "gpt-oss-120b"
LOCAL_ENDPOINT = "https://excelsoft-llm.excelindia.com/v1/gpt-oss-120b/responses"
LOCAL_HEADERS  = {"Content-Type": "application/json"}

SYSTEM_PROMPT = """You are a strict proctoring assistant specializing in phone detection during exams.

Your job:
1. Validate if the detected object is genuinely a mobile phone
2. Assign severity based on position, size, and confidence

Validation rules:
- REJECT if confidence below 0.35
- ACCEPT if confidence above 0.6
- For confidence 0.35-0.6 use position and size to decide
- Small coverage (<1%) with low confidence → reject
- Phone at ear position (top corner) is valid even with low coverage

Severity rules:
- LOW: confidence 0.35-0.55, partially visible, or position unclear
- MEDIUM: confidence 0.55-0.75, clearly visible, held in hand
- HIGH: confidence above 0.75, clearly held or at ear position

Your reason MUST describe position and how phone is being held.
DO NOT mention aspect ratio, screen, brightness, or color.

Respond ONLY in this JSON format:
{
  "valid": true,
  "severity": "MEDIUM",
  "reason": "Phone clearly held in hand at bottom-center with high confidence"
}"""

def _describe_detection(frame: np.ndarray, detections: list[dict]) -> str:
    """
    Converts detections into text description for local model
    """
    h,w = frame.shape[:2]
    lines = []


    for i, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = d["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        area_pct = round((bw * bh) / (w * h) * 100, 1)

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        h_pos = "left" if cx < 0.33 else "center" if cx < 0.66 else "right"
        v_pos = "top"  if cy < 0.33 else "middle" if cy < 0.66 else "bottom"

        #partial visibility check
        touches_edge = x1 <= 5 or y1 <=5 or x2 >= w - 5 or y2 >= h-5
        print(f"touches_edge={touches_edge} bbox=({x1},{y1},{x2},{y2}) frame=({w},{h})")

        #aspect ratio filter - this is triggered only if fully visible
        if not touches_edge:
            ratio = round(bw / bh, 2) if bh > 0 else 0
            is_portrait  = 0.20 < ratio < 0.75
            is_landscape = 1.5  < ratio < 2.2
            if not (is_portrait or is_landscape):
                lines.append(f"Detection {i}: aspect ratio {ratio} inconsistent with phone — rejected.")
                continue

        #get the surrounding region (frame minus phone bos)
        #region of interest to only get the frame identified
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        near_top = cy < 0.4
        near_edge = cx < 0.15 or cx > 0.85
        if near_top and near_edge:
            hold_hint = "phone appears held to ear — narrow profile, top corner position"
        elif area_pct > 3 and v_pos in ["middle", "bottom"]:
            hold_hint = "phone appears held in hand"
        else:
            hold_hint = "phone position unclear"

        lines.append(
            f"Detection {i}: cell phone, YOLO confidence {d['confidence']}, "
            f"located at {v_pos}-{h_pos}, "
            f"covering {area_pct}% of frame, "
            f"{hold_hint}"
        )

    return "\n".join(lines)

def _call_local_model(prompt: str) -> str:
    payload = {
        "model": LOCAL_MODEL_ID,
        "instructions": SYSTEM_PROMPT,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.3,
        "max_output_tokens": 800,
        "reasoning": {"effort": "low"}
    }
    response = requests.post(
        LOCAL_ENDPOINT,
        headers=LOCAL_HEADERS,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    result = response.json()

    if "output_text" in result:
        return result["output_text"]
    if "output" in result:
        for item in result["output"]:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return c.get("text")

    raise ValueError("No output_text in response")


def validate_phone(frame, detections):
    if not detections:
        return None

    description = _describe_detection(frame, detections)
    prompt = f"Exam frame phone detection:\n{description}\n\nValidate and assess severity. Return only valid JSON."

    try:
        raw = _call_local_model(prompt)
        # clean up
        raw = re.sub(r"```json|```", "", raw).strip()
        # extract first { } block
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        parsed = json.loads(raw)
        logger.info(f"VLM result — valid: {parsed['valid']} severity: {parsed.get('severity')}")
        return parsed
    except Exception as e:
        logger.error(f"Local model error: {e}")
        return None
    

def detect_phone(frame) -> list[dict]:
    from pipeline.detector import detect
    detections = detect(frame, labels=["cell phone"])

    filtered = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            continue
        ratio = round(w / h, 2)
        print(f"Detection — conf: {d['confidence']} ratio: {ratio}")
        d["label"] = "phone"
        filtered.append(d)

    return filtered