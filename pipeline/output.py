import cv2
import json
import os
import numpy as np
from datetime import datetime
from config import OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger("output")

VIOLATION_COLOR = (0, 0, 255)  # red


def compose_output(
    frame: np.ndarray,
    validated_detections: list[dict],
    violations: list[dict],
    status_text = None
) -> np.ndarray:
    """
    It draws the violation overlays on final annotated frame
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # red border if violations exist
    if violations:
        cv2.rectangle(output, (0, 0), (w - 1, h - 1), VIOLATION_COLOR, 8)

    # violation text panel at bottom
    panel_h = 30 * (len(violations) + 1)
    cv2.rectangle(output, (0, h - panel_h), (w, h), (0, 0, 0), -1)

    cv2.putText(
        output,
        f"VIOLATIONS: {len(violations)}",
        (10, h - panel_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        VIOLATION_COLOR if violations else (0, 255, 0),
        2
    )

    for i, v in enumerate(violations, start=1):
        cv2.putText(
            output,
            f"  {i}. {v['rule']}",
            (10, h - panel_h + 22 + (i * 28)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            VIOLATION_COLOR, 1
        )


    if status_text:
        cv2.putText(output, status_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        
    return output


def save_output(
    frame: np.ndarray,
    violations: list[dict],
    validated_detections: list[dict],
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save frame
    frame_path = os.path.join(OUTPUT_DIR, f"frame_{ts}.jpg")
    cv2.imwrite(frame_path, frame)

    # save report
    report = {
        "timestamp":   ts,
        "violations":  violations,
        "detections":  validated_detections,
    }
    report_path = os.path.join(OUTPUT_DIR, f"report_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved frame → {frame_path}")
    logger.info(f"Saved report → {report_path}")

    return frame_path, report_path