import base64
import json
import cv2
import numpy as np
from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
)
from utils.logger import get_logger

logger = get_logger("vlm_validator")

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


def _encode_frame(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


SYSTEM_PROMPT = """You are a proctoring validation assistant.
You will receive an annotated exam frame with numbered bounding boxes and a list of detections.
Your job is to validate each detection and remove false positives.
Respond ONLY in this JSON format, no extra text:
{
  "validated": [
    {"number": 1, "label": "phone", "valid": true, "reason": "clearly a mobile phone"},
    {"number": 2, "label": "notebook", "valid": false, "reason": "it is a water bottle"}
  ]
}
"""


def validate(annotated_frame: np.ndarray, detections: list[dict]) -> list[dict]:
    """
    Sends annotated frame + detections to VLM.
    Returns only validated detections.
    """
    if not detections:
        return []

    encoded = _encode_frame(annotated_frame)

    detection_text = "\n".join(
        f"{i+1}. {d['label']} (confidence: {d['confidence']})"
        for i, d in enumerate(detections)
    )

    user_message = f"""Here are the detections from an exam frame:
{detection_text}

Validate each numbered detection in the image.
Is each one a real phone, notebook, or person?
Respond in the required JSON format only."""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}"
                    }}
                ]
            }
        ],
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"VLM raw response: {raw}")

    try:
        parsed = json.loads(raw)
        validated_numbers = {
            v["number"] for v in parsed["validated"] if v["valid"]
        }
        validated_detections = [
            d for i, d in enumerate(detections, start=1)
            if i in validated_numbers
        ]
        logger.info(f"VLM validated {len(validated_detections)}/{len(detections)} detections")
        return validated_detections

    except Exception as e:
        logger.error(f"VLM parse error: {e} — returning all detections")
        return detections