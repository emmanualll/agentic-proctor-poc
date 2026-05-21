import json, re, requests, numpy as np
from utils.logger import get_logger
from config import LOCAL_MODEL_ID, LOCAL_ENDPOINT, LOCAL_HEADERS

logger = get_logger("vlm_earphone")


SYSTEM_PROMPT = """You are a strict proctoring assistant detecting audio devices during exams.

You receive detection metadata including label, confidence, and position in frame.

Rules:
- REJECT if no detection is near ear region (top-left or top-right of frame)
- REJECT if all confidences below 0.25
- ACCEPT if any detection above 0.35 is in near ear region
- ACCEPT if multiple detections above 0.30 regardless of position

Severity:
- LOW: detections present but position unclear or low confidence
- MEDIUM: clear detection, confidence 0.35-0.50
- HIGH: confidence above 0.50 OR detection explicitly near ear region

Your reason MUST mention the position and whether it appears worn or on a desk.

Respond ONLY in this JSON format:
{
  "valid": true,
  "severity": "HIGH",
  "reason": "Airpod detected near right ear region with confidence 0.547 — appears to be worn"
}"""

def _describe_detection(phrases, logits, boxes) -> str:
    lines = []
    boxes = boxes.tolist()
    for phrase, conf, box in zip(phrases, logits, boxes):
        cx, cy = float(box[0]), float(box[1])
        
        h_pos = "left" if cx < 0.35 else "right" if cx > 0.65 else "center"
        v_pos = "top" if cy < 0.35 else "bottom" if cy > 0.65 else "middle"
        
        near_ear = v_pos == "top" and h_pos in ["left", "right"]
        location = "near ear region" if near_ear else f"{v_pos}-{h_pos} of frame"
        
        lines.append(
            f"Detected: {phrase}, confidence: {round(float(conf), 3)}, position: {location}"
        )
    return "\n".join(lines)

def _call_local_model(prompt: str) -> str:
    payload = {
        "model": LOCAL_MODEL_ID,
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "temperature": 0.3,
        "max_output_tokens": 800,
        "reasoning": {"effort": "low"}
    }
    response = requests.post(LOCAL_ENDPOINT, headers=LOCAL_HEADERS, json=payload, timeout=120)
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

def validate_earphone(phrases, logits, boxes):
    if not phrases:
        return None
    description = _describe_detection(phrases, logits, boxes)
    prompt = f"Exam frame audio device detection:\n{description}\n\nValidate and assess severity. Return only valid JSON."
    try:
        raw = _call_local_model(prompt)
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None