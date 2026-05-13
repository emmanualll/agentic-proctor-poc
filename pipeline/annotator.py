import cv2
import numpy as np

COLORS = {
    "person":   (0, 255, 0),
    "phone":    (0, 0, 255),
    "notebook": (255, 165, 0),
}
DEFAULT_COLOR = (200, 200, 200)


def annotate(frame, detections: list[dict]) -> np.ndarray:
    """
    Draws numbered boxes + arrows on frame.
    Returns annotated copy.
    """
    annotated = frame.copy()

    for i, det in enumerate(detections, start=1):
        label      = det["label"]
        confidence = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS.get(label, DEFAULT_COLOR)

        #bounding box used for numbering 
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # number circle
        cx, cy = (x1 + x2) // 2, y1 - 20
        cv2.circle(annotated, (cx, cy), 14, color, -1)
        cv2.putText(annotated, str(i), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # arrow from circle to box top
        cv2.arrowedLine(annotated, (cx, cy + 14), (cx, y1),
                        color, 2, tipLength=0.3)

        # label + confidence
        text = f"{label} {confidence:.2f}"
        cv2.putText(annotated, text, (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return annotated


def show_annotated(annotated_frame):
    cv2.imshow("Annotated", annotated_frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()