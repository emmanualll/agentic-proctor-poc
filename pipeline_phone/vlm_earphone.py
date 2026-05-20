import json, re, requests, numpy as np
from utils.logger import get_logger
from config import LOCAL_MODEL_ID, LOCAL_ENDPOINT, LOCAL_HEADERS

logger = get_logger("vlm_earphone")


SYSTEM_PROMPT = """You are a strict proctoring assistant specializing in audio device detection during exams.

Your job:
1. Validate if the detected object is genuinely an earphone, airpod, earbud, or headphone
2. Assign severity based on confidence and position

Validation rules:
- ACCEPT if confidence above 0.30
- REJECT if confidence below 0.20
- Multiple detections = higher confidence

Severity rules:
- LOW: single detection, low confidence (0.25-0.35)
- MEDIUM: clear detection, confidence 0.35-0.5
- HIGH: multiple detections or confidence above 0.5

Respond ONLY in this JSON format:
{
  "valid": true,
  "severity": "MEDIUM",
  "reason": "Earbud detected near ear region with moderate confidence"
}"""

def _describe_detection(phrases, logits) -> str:
    lines = []
    for phrase, conf in zip(phrases, logits):
        lines.append(f"Detected: {phrase}, confidence: {round(float(conf), 3)}")
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

def validate_earphone(phrases, logits):
    if not phrases:
        return None
    description = _describe_detection(phrases, logits)
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