import os


DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoint"
PREDICTION_DIR = "pred"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

TRAIN_FILE = "train.jsonl"
VALID_FILE = "public.jsonl"

DATA_FILE = {
    "train": os.path.join(DATA_DIR, TRAIN_FILE),
    "valid": os.path.join(DATA_DIR, VALID_FILE),
}

TEXT_COL = "maintext"
SUMMARY_COL = "title"
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 64
