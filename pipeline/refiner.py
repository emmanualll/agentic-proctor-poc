import json
import base64
import cv2
import numpy as np
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
)
from utils.logger import get_logger

logger = get_logger("refiner")

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

CRITIC_PROMPT = """You are a proctoring detection critic.
You receive an annotated exam frame with numbered detections.
Your job:
1. Validate each detection — is it really that object?
2. Identify any missed suspicious objects not detected.
3. Suggest refined search terms for missed objects.

Respond ONLY in this JSON format:
{
  "validated": [
    {"number": 1, "label": "phone", "valid": true, "reason": "clearly a mobile phone"}
  ],
  "missed": ["smartwatch", "earphones"],
  "rerun_needed": true
}
"""

def _encode_frame(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def critic(
    annotated_frame: np.ndarray,
    detections: list[dict],
    is_final: bool = False
) -> dict:
    """
    Runs critic on annotated frame + detections.
    Returns {validated_detections, rerun_needed, missed_concepts}
    """
    encoded = _encode_frame(annotated_frame)

    detection_text = "\n".join(
        f"{i+1}. {d['label']} (confidence: {d['confidence']})"
        for i, d in enumerate(detections)
    ) or "No detections found."

    role_note = "This is the FINAL validation pass." if is_final else "This is the initial validation pass."

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{role_note}\n\nDetections:\n{detection_text}\n\nValidate and critique."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}"
                    }}
                ]
            }
        ],
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"Critic raw: {raw}")

    try:
        parsed = json.loads(raw)

        validated_numbers = {
            v["number"] for v in parsed["validated"] if v["valid"]
        }
        validated_detections = [
            d for i, d in enumerate(detections, start=1)
            if i in validated_numbers
        ]

        return {
            "validated_detections": validated_detections,
            "rerun_needed": parsed.get("rerun_needed", False) and not is_final,
            "missed_concepts": parsed.get("missed", []),
        }

    except Exception as e:
        logger.error(f"Critic parse error: {e} — passing all detections through")
        return {
            "validated_detections": detections,
            "rerun_needed": False,
            "missed_concepts": [],
        }


def refine_and_rerun(
    frame: np.ndarray,
    current_detections: list[dict],
    missed_concepts: list[str]
) -> list[dict]:
    """
    Reruns DINO with original + missed concepts merged.
    Returns merged deduplicated detections.
    """
    from pipeline.detector import detect, build_prompt
    from config import DETECTION_LABELS

    if not missed_concepts:
        return current_detections

    combined = list(set(DETECTION_LABELS + missed_concepts))
    logger.info(f"Rerunning DINO with expanded concepts: {combined}")

    new_detections = detect(frame, labels=combined)

    # merge — avoid duplicates by bbox overlap
    all_detections = current_detections.copy()
    for new in new_detections:
        if not _is_duplicate(new, current_detections):
            all_detections.append(new)

    return all_detections


def _is_duplicate(det: dict, existing: list[dict], iou_thresh=0.5) -> bool:
    for e in existing:
        if _iou(det["bbox"], e["bbox"]) > iou_thresh:
            return True
    return False


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)