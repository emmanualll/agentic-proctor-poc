from config import MAX_ALLOWED_PERSONS
from utils.logger import get_logger

logger = get_logger("rule_engine")

VIOLATION_RULES = {
    "phone":    "Candidate has a phone visible.",
    "notebook": "Candidate has a notebook/cheat sheet visible.",
}


def apply_rules(detections: list[dict]) -> list[dict]:
    """
    Takes the violated detections and returns a list of violations. each violation will have rule confidence label bbox 
    """
    violations = []

    persons = [d for d in detections if d["label"] == "person"]
    others  = [d for d in detections if d["label"] != "person"]

    # Rule 1: forbidden objects
    for det in others:
        if det["label"] in VIOLATION_RULES:
            violations.append({
                "rule":       VIOLATION_RULES[det["label"]],
                "label":      det["label"],
                "bbox":       det["bbox"],
                "confidence": det["confidence"],
            })
            logger.warning(f"VIOLATION — {VIOLATION_RULES[det['label']]}")

    # Rule 2: multiple persons
    if len(persons) > MAX_ALLOWED_PERSONS:
        violations.append({
            "rule":       f"Multiple people detected ({len(persons)}).",
            "label":      "person",
            "bbox":       persons[0]["bbox"],
            "confidence": persons[0]["confidence"],
        })
        logger.warning(f"VIOLATION — Multiple people detected: {len(persons)}")

    if not violations:
        logger.info("No violations found.")

    return violations