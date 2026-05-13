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

logger = get_logger("concept_detector")

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

SYSTEM_PROMPT = """You are a proctoring concept extractor.
Given an exam frame, extract what suspicious objects to look for.
You must always include: cell phone, person, paper notebook.
Add any other suspicious objects you see (earphones, smartwatch, etc).
Respond ONLY in this JSON format, no extra text:
{
  "concepts": ["cell phone", "person", "paper notebook", "earphones"]
}
"""

def _encode_frame(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def extract_concepts(frame: np.ndarray) -> list[str]:
    """
    Sends frame to VLM, returns dynamic list of concepts to detect.
    Falls back to default labels on failure.
    """
    from config import DETECTION_LABELS

    encoded = _encode_frame(frame)

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract suspicious objects to detect in this exam frame."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}"
                        }}
                    ]
                }
            ],
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        concepts = parsed["concepts"]
        logger.info(f"Concepts extracted: {concepts}")
        return concepts

    except Exception as e:
        logger.error(f"Concept extraction failed: {e} — using defaults")
        return DETECTION_LABELS