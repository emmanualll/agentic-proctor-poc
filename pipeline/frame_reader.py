import cv2
from config import FRAME_SAMPLE_INTERVAL
from utils.logger import get_logger

logger = get_logger("frame_reader")

def read_frames(source=0):
    """
    Generator yeilds frame_index and frame
    source: 0 for webcam or path to video file
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Stream ended.")
                break

            if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
                yield frame_idx, frame

            frame_idx += 1
    finally:
        cap.release()
        logger.info("Camera released.")