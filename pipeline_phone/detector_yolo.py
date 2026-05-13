import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE
from utils.logger import get_logger

logger = get_logger("detector_yolo")

_model = None

def load_yolo():
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLOv8s detector loaded.")
    return _model


def detect_phone(frame) -> list[dict]:
    """
    Returns list of phone detections with bbox + confidence.
    """
    model = load_yolo()
    results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)

    detections = []
    for box in results[0].boxes:
        label = results[0].names[int(box.cls)]
        if label != "cell phone":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(float(box.conf), 3)
        detections.append({
            "label":      "phone",
            "confidence": conf,
            "bbox":       (x1, y1, x2, y2)
        })
        logger.debug(f"Phone detected — conf: {conf} bbox: {x1,y1,x2,y2}")

    return detections