# Agentic Object Detection Pipeline for AI Proctoring

An event-driven AI proctoring pipeline that combines motion triggering, open-vocabulary object detection, iterative refinement, rule-based reasoning, and evidence generation.

---

## Features

- Motion-triggered analysis
- GroundingDINO-based object detection
- Dynamic concept extraction
- Critic-based validation and rerun refinement
- Rule-based violation engine
- Real-time webcam monitoring
- Annotated evidence generation
- JSON report export
- Violation logging

---

## Pipeline Architecture

```text
Camera Feed
    ↓
Frame Reader
    ↓
Motion Trigger
    ↓
Concept Detector
    ↓
GroundingDINO Detection
    ↓
Critic Validation
    ↓
Refinement + Re-run
    ↓
Rule Engine
    ↓
Evidence + Reports
```

---

## Detection Classes

- Person
- Phone
- Notebook

---

## Tech Stack

- Python
- OpenCV
- PyTorch
- GroundingDINO
- NumPy
- Pillow

---

## Project Structure

```text
project/
│
├── pipeline/
│   ├── frame_reader.py
│   ├── trigger.py
│   ├── detector.py
│   ├── annotator.py
│   ├── concept_detector.py
│   ├── refiner.py
│   ├── rule_engine.py
│   └── output.py
│
├── utils/
│   └── logger.py
│
├── output/
├── weights/
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Setup

Clone the repository:

```bash
git clone <repo-url>
cd <repo-name>
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

Press `Q` to quit.

---

## Output

The pipeline generates:

- Annotated frames
- JSON reports
- Violation logs

Saved in:

```text
output/
```

---

## Current Limitations

- GroundingDINO may produce false positives
- CPU inference is relatively slow
- Detection quality depends heavily on prompts
- No temporal tracking yet

---

## Future Improvements

- YOLO-based production detector
- Temporal behavior tracking
- LLM-based reasoning layer
- Web dashboard
- Real-time alerts
- Multi-camera support

---

## Author

Emmanual Antony Clement