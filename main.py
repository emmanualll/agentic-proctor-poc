import cv2
import time
import threading
import os
import queue
from datetime import datetime

from config import PIPELINE_COOLDOWN_SECONDS, OUTPUT_DIR, VIOLATION_LOG_PATH
from pipeline.frame_reader import read_frames
from pipeline.trigger import is_suspicious, load_trigger_model
from pipeline.detector import load_model, detect
from pipeline.annotator import annotate
from pipeline.concept_detector import extract_concepts
from pipeline.refiner import critic, refine_and_rerun
from pipeline.rule_engine import apply_rules
from pipeline.output import compose_output, save_output
from utils.logger import get_logger

logger = get_logger("main")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_last_run_time = 0
frame_queue    = queue.Queue(maxsize=3)


def log_violation(violations: list[dict], frame_path: str):
    with open(VIOLATION_LOG_PATH, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for v in violations:
            line = f"[{ts}] {v['rule']} | frame: {frame_path}\n"
            f.write(line)
            print(f"\n🚨 VIOLATION: {v['rule']}")


def run_pipeline(frame):
    try:
        print("1 - extracting concepts")
        concepts   = extract_concepts(frame)
        print(f"2 - concepts: {concepts}")
        detections = detect(frame, labels=concepts)
        print(f"3 - detections: {detections}")

        if not detections:
            print("NO DETECTIONS — returning")
            return

        annotated = annotate(frame, detections)
        critique  = critic(annotated, detections, is_final=False)
        print(f"4 - critique: {critique}")

        if critique["rerun_needed"]:
            detections = refine_and_rerun(
                frame,
                critique["validated_detections"],
                critique["missed_concepts"]
            )
            annotated = annotate(frame, detections)

        final      = critic(annotated, detections, is_final=True)
        validated  = final["validated_detections"]
        violations = apply_rules(validated)
        print(f"5 - violations: {violations}")

        if violations:
            final_frame = compose_output(annotated, validated, violations)
            frame_path, _ = save_output(final_frame, violations, validated)
            log_violation(violations, frame_path)

    except Exception as e:
        import traceback
        traceback.print_exc()


def pipeline_worker():
    global _last_run_time
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            print("WORKER GOT FRAME — running pipeline...")
        except queue.Empty:
            continue
        try:
            run_pipeline(frame)
            print("WORKER DONE")
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            frame_queue.task_done()

def main(source=0):
    global _last_run_time

    load_model()
    load_trigger_model()
    logger.info("Pipeline ready. Press Q to quit.")

    # start worker thread
    worker = threading.Thread(target=pipeline_worker, daemon=True)
    worker.start()
    print(f"Worker started: {worker.is_alive()}")

    for idx, frame in read_frames(source):
        now             = time.time()
        cooldown_passed = (now - _last_run_time) > PIPELINE_COOLDOWN_SECONDS

        suspicious = is_suspicious(frame)

        if suspicious and cooldown_passed and not frame_queue.full():
            frame_queue.put(frame.copy())
            _last_run_time = now
            logger.info(f"Frame {idx} queued.")

            
        cv2.putText(frame, "MONITORING", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if not frame_queue.empty():
            cv2.putText(frame, "ANALYZING...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Proctor Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(source=0)