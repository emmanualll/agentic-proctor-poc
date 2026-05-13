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

SYSTEM_PROMPT = """You are a proctoring assistant specializing in phone detection.
Given a text description of a phone detection in an exam frame, you must:
1. Validate if it is likely a real phone
2. Assess severity of the violation

Severity levels:
- LOW: phone partially visible, angled, small in frame, unclear if in use
- MEDIUM: phone held in hand but screen off or unclear
- HIGH: screen is ON, actively being used, or placed on desk for use. Screen ON overrides everything.

Respond ONLY in this JSON format, no extra text:
{
  "valid": true,
  "severity": "HIGH",
  "reason": "Phone screen is clearly on and candidate is looking at it"
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

        if area_pct > 3 and v_pos in ["middle", "bottom"]:
            hold_hint = "phone appears to be held in hand"
        else:
            hold_hint = "phone position unclear"

        roi = frame[y1:y2, x1:x2]
        brightness = round(float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()), 1)

        if d["confidence"] > 0.75 and brightness > 120:
            screen_hint = "screen is ON and phone is actively being used"
        elif hold_hint == "phone appears to be held in hand" and d["confidence"] > 0.60:
            screen_hint = "phone clearly held in hand, screen status unclear"
        else:
            screen_hint = "phone partially visible or screen OFF"

        lines.append(
            f"Detection {i}: cell phone, YOLO confidence {d['confidence']}, "
            f"located at {v_pos}-{h_pos}, "
            f"covering {area_pct}% of frame, "
            f"{hold_hint}, {screen_hint}"
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