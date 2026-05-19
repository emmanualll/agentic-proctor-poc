from groundingdino.util.inference import load_model, load_image, predict
from utils.logger import get_logger
import numpy as np
import cv2, tempfile, os

logger = get_logger("detector_earphone")

GDINO_CONFIG     = "weights/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT = "weights/groundingdino_swint_ogc.pth"
CAPTION = "earphone . airpod . earbud . white earbud . headphone . over-ear headphone"

_model = None

def load_earphone_model():
    global _model
    if _model is None:
        _model = load_model(GDINO_CONFIG, GDINO_CHECKPOINT)
        logger.info("Grounding DINO earphone detector loaded.")

def detect_earphone(frame):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, frame)
    _, image = load_image(tmp.name)
    os.unlink(tmp.name)
    boxes, logits, phrases = predict(
        model=_model, image=image,
        caption=CAPTION,
        box_threshold=0.25, text_threshold=0.20
    )
    return phrases, logits