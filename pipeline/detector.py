
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.cuda.is_available = lambda: False
import cv2
import torch
import numpy as np
from PIL import Image

from groundingdino.util.inference import load_model as gdino_load, predict
from config import GDINO_CONFIG_PATH, GDINO_CHECKPOINT_PATH, GDINO_BOX_THRESH, GDINO_TEXT_THRESH, DETECTION_LABELS
from utils.logger import get_logger

logger = get_logger("detector")

LABEL_MAP = {
    "cell phone": "phone",
    "phone": "phone",
    "smartphone": "phone",
    "mobile phone": "phone",
    "tablet": "phone",
    "paper notebook": "notebook",
    "notebook": "notebook",
    "person": "person",
    "smartwatch": "smartwatch",
    "earphones": "earphones",
}
def normalize_label(phrase: str) -> str:
    phrase = phrase.lower().strip()
    for key, val in LABEL_MAP.items():
        if key in phrase:
            return val
    return phrase


_model = None
def load_model():
    global _model
    if _model is None:
        logger.info("Loading Grounding DINO...")
        _model = gdino_load(GDINO_CONFIG_PATH, GDINO_CHECKPOINT_PATH)
        _model = _model.to(torch.device("cpu"))
        logger.info("Model Loaded!")
    return _model

def build_prompt() -> str:
    return " . ".join(DETECTION_LABELS) + " ."

def detect(frame, labels:list[str] = None) -> list[dict]:
    model = load_model()

    if labels is None:
        from config import DETECTION_LABELS
        labels = DETECTION_LABELS

    prompt = " . ".join(labels) + " ."

    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w = frame.shape[:2]

    # GroundingDINO expects a normalized tensor on CPU
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil).to(torch.device("cpu"))

    prompt = build_prompt()

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=GDINO_BOX_THRESH,
        text_threshold=GDINO_TEXT_THRESH,
    )

    detections = []
    for box, logit, phrase in zip(boxes, logits, phrases):
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        detections.append({
            "label":      normalize_label(phrase),
            "confidence": round(logit.item(), 3),
            "bbox":       (x1, y1, x2, y2)
        })

    return detections