import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("groundingdino").setLevel(logging.ERROR)

import cv2
import time
import threading
import os
import queue
from datetime import datetime

from config import PIPELINE_COOLDOWN_SECONDS, OUTPUT_DIR, VIOLATION_LOG_PATH, FRAME_SAMPLE_INTERVAL, EARPHONE_CHECK_INTERVAL
from pipeline.frame_reader import read_frames
from pipeline.trigger import is_suspicious, load_trigger_model, get_person_bbox
from pipeline.annotator import annotate
from pipeline.output import compose_output, save_output
from pipeline_phone.detector_yolo import load_yolo, detect_phone
from pipeline_phone.vlm_phone import validate_phone
from utils.logger import get_logger

from pipeline_phone.detector_earphone import load_earphone_model, detect_earphone
from pipeline_phone.vlm_earphone import validate_earphone

from utils.display import print_header, print_status, print_earphone_detections, print_phone_detections, print_llm_result, print_violation

logger = get_logger("main")
os.makedirs(OUTPUT_DIR, exist_ok = True)

_last_run_time = 0
_last_suspicious_time = 0
_earphone_history = []
earphone_queue = queue.Queue(maxsize=1)
frame_queue = queue. Queue(maxsize=3)
_phone_tracker = {"first_seen": None, "severity": "LOW"}

def log_violation(violation: dict, frame_path: str):
    os.makedirs(os.path.dirname(VIOLATION_LOG_PATH), exist_ok=True)
    with open(VIOLATION_LOG_PATH, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] severity={violation['severity']} reason={violation['reason']} | frame: {frame_path}\n"
        f.write(line)
    print_violation(violation)

    
def run_pipeline(frame):
    try:
        # 2. Phone detection
        detections = detect_phone(frame)
        print_phone_detections(detections)
        if not detections:
            print("No phone detected — skipping phone pipeline")
            reset_tracker()
            return

        # 3. Annotate
        annotated = annotate(frame, detections)

        # 4. VLM validation
        result = validate_phone(frame, detections)
        print_llm_result(result, label="phone")
        if not result or not result.get("valid"):
            print("VLM invalidated detection — no violation")
            return

        # 5. Screen-on detection
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

        # 6. Time-based severity
        final_severity = get_time_based_severity(result["severity"])
        if final_severity != result["severity"]:
            reason = f"Candidate has been holding phone for extended period. Originally {result['severity']}: {result['reason']}"
            print(f"⏱️  SEVERITY ESCALATED: {result['severity']} → {final_severity}")
        else:
            reason = result["reason"]

        # 7. Build violation
        violation = {
            "rule":       "Phone detected during exam.",
            "label":      "phone",
            "severity":   final_severity,
            "reason":     reason,
            "bbox":       detections[0]["bbox"],
            "confidence": detections[0]["confidence"],
        }

        # 8. Compose + save
        final_frame = compose_output(annotated, [violation], [violation], status_text=screen_status)
        frame_path, _ = save_output(final_frame, [violation], detections)
        log_violation(violation, frame_path)
        print_violation(violation)

    except Exception as e:
        import traceback
        traceback.print_exc()

def pipeline_worker():
    global _last_run_time
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            print_status("PHONE CHECK — analyzing...", "yellow")
        except queue.Empty:
            continue
        try:
            now = time.time()
            cooldown_passed = (now - _last_run_time) > PIPELINE_COOLDOWN_SECONDS
            suspicious = is_suspicious(frame)
            if suspicious:
                _last_suspicious_time = now
                if _phone_tracker["first_seen"] is None:
                    _phone_tracker["first_seen"] = now
                if cooldown_passed:
                    run_pipeline(frame)
            else:
                if now - _last_suspicious_time > 3:
                    reset_tracker()
        except Exception as e:
            logger.error(f"Worker error : {e}")
        finally:
            _last_run_time = time.time()
            frame_queue.task_done()

def main(source=0):
    global _last_run_time, _last_suspicious_time
    _last_earphone_check = time.time() + 3

    load_yolo()
    load_earphone_model()
    load_trigger_model()
    print_header()
    logger.info("Phone proctoring pipeline ready. Press Q to quit.")

    worker = threading.Thread(target=pipeline_worker, daemon=True)
    worker.start()
    print(f"Worker started: {worker.is_alive()}")

    earphone_thread = threading.Thread(target=earphone_worker, daemon=True)
    earphone_thread.start()
    print(f"Earphone worker started: {earphone_thread.is_alive()}")


    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - _last_earphone_check > EARPHONE_CHECK_INTERVAL:
            _last_earphone_check = now
            if not earphone_queue.full():
                earphone_queue.put(frame.copy())

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
            now = time.time()
            cooldown_passed = (now - _last_run_time) > PIPELINE_COOLDOWN_SECONDS
            if cooldown_passed and not frame_queue.full():
                frame_queue.put(frame.copy())

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

def get_escalated_earphone_severity(llm_severity: str) -> str:
    _earphone_history.append(llm_severity)
    if len(_earphone_history) > 5:
        _earphone_history.pop(0)
    non_none = [s for s in _earphone_history if s is not None]
    if len(non_none) >= 3:
        return "HIGH"
    return llm_severity

def earphone_worker():
    while True:

        try:
            frame = earphone_queue.get(timeout=1)

        except queue.Empty:
            continue

        try:

            phrases, logits, boxes = detect_earphone(frame)

            print_status("EARPHONE CHECK — running...", "yellow")
            print_earphone_detections(phrases, logits)

            person_bbox = get_person_bbox(frame)

            # =====================================================
            # RELATIVE EAR-REGION FILTER
            # =====================================================

            if person_bbox and phrases:

                h, w = frame.shape[:2]

                px1, py1, px2, py2 = person_bbox

                pw = px2 - px1
                ph = py2 - py1

                frame_coverage = ph / h

                # -------------------------------------------------
                # CLOSE CAMERA ADAPTATION
                # -------------------------------------------------

                if frame_coverage > 0.75:

                    head_y2 = py1 + int(ph * 0.75)

                    left_x2  = px1 + int(pw * 0.40)
                    right_x1 = px2 - int(pw * 0.40)

                    print_status("EARPHONE — close camera mode", "dim")
                else:

                    head_y2 = py1 + int(ph * 0.40)

                    left_x2  = px1 + int(pw * 0.30)
                    right_x1 = px2 - int(pw * 0.30)

                filtered_phrases = []
                filtered_logits  = []
                filtered_boxes   = []

                if hasattr(boxes, "tolist"):
                    boxes_iter = boxes.tolist()
                else:
                    boxes_iter = boxes

                for p, l, b in zip(phrases, logits, boxes_iter):

                    cx, cy = float(b[0]), float(b[1])

                    ex = int(cx * w)
                    ey = int(cy * h)

                    in_head_height = py1 <= ey <= head_y2
                    in_left_zone   = px1 <= ex <= left_x2
                    in_right_zone  = right_x1 <= ex <= px2

                    near_ear = (
                        in_head_height and
                        (in_left_zone or in_right_zone)
                    )

                    if near_ear:

                        filtered_phrases.append(p)
                        filtered_logits.append(l)
                        filtered_boxes.append(b)

                        print_status(f"EAR REGION PASS — {p} conf={float(l):.3f}", "green")

                    else:
                        print_status(f"EAR REGION REJECT — {p} conf={float(l):.3f}", "red")

                phrases = filtered_phrases
                logits  = filtered_logits
                boxes   = filtered_boxes

            # =====================================================
            # AREA FILTER
            # =====================================================

            area_phrases = []
            area_logits  = []
            area_boxes   = []

            for p, l, b in zip(phrases, logits, boxes):

                bw = float(b[2])
                bh = float(b[3])

                area = bw * bh

                if area < 0.026:

                    area_phrases.append(p)
                    area_logits.append(l)
                    area_boxes.append(b)

                else:
                    print_status(f"AREA REJECT — {p} conf={float(l):.3f}", "red")

            phrases = area_phrases
            logits  = area_logits
            boxes   = area_boxes

            # =====================================================
            # CONFIDENCE FILTER
            # =====================================================

            max_conf = max(
                [float(l) for l in logits],
                default=0
            )

            print_status(f"EARPHONE CONF FILTER — max={max_conf:.3f} threshold=0.40", "yellow")

            if max_conf < 0.40:

                print_status("EARPHONE — rejected before LLM", "red")

                continue

            # =====================================================
            # VLM VALIDATION
            # =====================================================

            print_status(f"EARPHONE — passing {len(phrases)} detections to LLM", "yellow")
            result = validate_earphone(
                phrases,
                logits,
                boxes,
                person_present=person_bbox is not None
            )

            if result and result.get("valid"):

                logger.warning(
                    f"EARPHONE VIOLATION — "
                    f"{result['severity']} — "
                    f"{result['reason']}"
                )

        except Exception as e:

            logger.error(
                f"Earphone worker error: {e}"
            )

if __name__ == "__main__":
    main(source=0)

