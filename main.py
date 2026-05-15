import cv2
import time
import threading
import os
import queue
from datetime import datetime

from config import PIPELINE_COOLDOWN_SECONDS, OUTPUT_DIR, VIOLATION_LOG_PATH, FRAME_SAMPLE_INTERVAL
from pipeline.frame_reader import read_frames
from pipeline.trigger import is_suspicious, load_trigger_model
from pipeline.annotator import annotate
from pipeline.output import compose_output, save_output
from pipeline_phone.detector_yolo import load_yolo, detect_phone
from pipeline_phone.vlm_phone import validate_phone
from utils.logger import get_logger

logger = get_logger("main")
os.makedirs(OUTPUT_DIR, exist_ok = True)

_last_run_time = 0
_last_suspicious_time = 0
frame_queue = queue. Queue(maxsize=3)
_phone_tracker = {"first_seen": None, "severity": "LOW"}

def log_violation(violation: dict, frame_path: str):
    os.makedirs(os.path.dirname(VIOLATION_LOG_PATH), exist_ok=True)
    with open(VIOLATION_LOG_PATH, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] severity={violation['severity']} reason={violation['reason']} | frame: {frame_path}\n"
        f.write(line)
    print(f"\n🚨 PHONE VIOLATION [{violation['severity']}]: {violation['reason']}")

def run_pipeline(frame):
    try:
        # 1. YOLO detection
        detections = detect_phone(frame)
        print(f"YOLO detections: {detections}")

        if not detections:
            print("No phone detected — skipping")
            reset_tracker()
            return

        # 2: annotate
        annotated = annotate(frame, detections)

        # 3: VLM validation + severity
        result = validate_phone(frame, detections)
        print(f"VLM result: {result}")

        if not result or not result.get("valid"):
            print("VLM invalidated detection — no violation")
            return

        # 4: screen-on detection
        screen_status = "SCREEN: UNCLEAR"
        screen_on     = False
        x1, y1, x2, y2 = detections[0]["bbox"]
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            phone_brightness = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
            full_brightness  = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            brightness_ratio = phone_brightness / full_brightness if full_brightness > 0 else 1

            if brightness_ratio > 0.88:
                screen_status = "SCREEN: ON"
                screen_on     = True
            elif brightness_ratio > 0.70:
                screen_status = "SCREEN: UNCLEAR"
            else:
                screen_status = "SCREEN: OFF"

            print(f"📱 {screen_status} (ratio={round(brightness_ratio,2)})")

        if screen_on and result["severity"] == "LOW":
            result["severity"] = "MEDIUM"
            result["reason"]  += " — screen appears ON"

        # 5: time based severity
        final_severity = get_time_based_severity(result["severity"])
        if final_severity != result["severity"]:
            reason = f"Candidate has been holding phone for extended period. Originally {result['severity']}: {result['reason']}"
            print(f"⏱️  SEVERITY ESCALATED: {result['severity']} → {final_severity}")
        else:
            reason = result["reason"]

        # 6: build violation
        violation = {
            "rule":       "Phone detected during exam.",
            "label":      "phone",
            "severity":   final_severity,
            "reason":     reason,
            "bbox":       detections[0]["bbox"],
            "confidence": detections[0]["confidence"],
        }

        # 7: compose + save
        final_frame = compose_output(annotated, [violation], [violation], status_text=screen_status)
        frame_path, _ = save_output(final_frame, [violation], detections)
        log_violation(violation, frame_path)

    except Exception as e:
        import traceback
        traceback.print_exc()

def pipeline_worker():
    global _last_run_time
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            print("WORKER GOT FRAME -- runnning pipeline....")
        except queue.Empty:
            continue
        try:
            run_pipeline(frame)
            print("Worker Done")
        except Exception as e:
            logger.error(f"Worker error : {e}")
        finally:
            _last_run_time = time.time()
            frame_queue.task_done()

def main(source=0):
    global _last_run_time, _last_suspicious_time

    load_yolo()
    load_trigger_model()
    logger.info("Phone proctoring pipeline ready. Press Q to quit.")

    worker = threading.Thread(target=pipeline_worker, daemon=True)
    worker.start()
    print(f"Worker started: {worker.is_alive()}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # display every frame — smooth feed
        display = frame.copy()
        cv2.putText(display, "MONITORING", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if not frame_queue.empty():
            cv2.putText(display, "ANALYZING...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.imshow("Proctor Feed", display)

        # process only every Nth frame
        if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
            now             = time.time()
            cooldown_passed = (now - _last_run_time) > PIPELINE_COOLDOWN_SECONDS
            suspicious      = is_suspicious(frame)

            if suspicious:
                _last_suspicious_time = now
                if _phone_tracker["first_seen"] is None:
                    _phone_tracker["first_seen"] = now
                if cooldown_passed and not frame_queue.full():
                    frame_queue.put(frame.copy())
                    _last_run_time = now
                    logger.info(f"Frame {frame_idx} queued.")

            if not suspicious:
                if now - _last_suspicious_time > 3:
                    reset_tracker()

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_time_based_severity(vlm_severity: str) -> str:
    """Escalate severity if phone stays in frame."""
    now = time.time()
    if _phone_tracker["first_seen"] is None:
        _phone_tracker["first_seen"] = now

    duration = now - _phone_tracker["first_seen"]

    if duration > 10:
        return "HIGH"
    elif duration > 5:
        return max_severity(vlm_severity, "MEDIUM")
    return vlm_severity


def max_severity(a: str, b: str) -> str:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def reset_tracker():
    _phone_tracker["first_seen"] = None

if __name__ == "__main__":
    main(source=0)

