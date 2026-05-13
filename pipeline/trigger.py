import cv2
import numpy as np
from config import TRIGGER_MOTION_THRESHOLD
from utils.logger import get_logger
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_TARGET_CLASSES, MAX_ALLOWED_PERSONS
from utils.logger import get_logger

logger = get_logger("trigger")

_model = None

def load_trigger_model():
    global _model
    if _model is None:
        logger.info("Loading YOLOv8n trigger model...")
        _model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLOv8n ready.")
    return _model

def is_suspicious(frame) -> bool:
    """
    it returns true only if suspicious object is detected
    """
    model = load_trigger_model()

    results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)
    detected_labels = [
        results[0].names[int(cls)]
        for cls in results[0].boxes.cls
    ]

    person_count    = detected_labels.count("person")
    has_forbidden   = any(
        l in YOLO_TARGET_CLASSES and l != "person"
        for l in detected_labels
    )

    multiple_people = person_count > MAX_ALLOWED_PERSONS

    triggered = has_forbidden or multiple_people

    if triggered:
        logger.debug(f"Trigger fired — detected: {detected_labels}")

    return triggered