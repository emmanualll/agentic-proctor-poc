import os
from dotenv import load_dotenv

load_dotenv()


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "interns-gpt-4.1")


GDINO_CONFIG_PATH     = "weights/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"

#detection
DETECTION_LABELS  = ["cell phone", ]
GDINO_TEXT_THRESH = 0.35

# Trigger
TRIGGER_MOTION_THRESHOLD = 5000  
FRAME_SAMPLE_INTERVAL    = 15      # process every Nth frame

#Person Rule
MAX_ALLOWED_PERSONS = 1

#Output
OUTPUT_DIR = "output"

GDINO_DEVICE = "cpu"

PIPELINE_COOLDOWN_SECONDS = 5 
VIOLATION_LOG_PATH = "output/violations.log"

YOLO_MODEL_PATH = "weights/yolov8s.pt" 
YOLO_CONFIDENCE = 0.35
YOLO_TARGET_CLASSES = ["cell phone"]

LOG_LEVEL = "WARNING"