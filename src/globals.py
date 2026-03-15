import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent  # src/ directory
load_dotenv(BASE_DIR / ".env") 

# The audio dataset folder
DATASET_FOLDER = os.getenv("DATASET_FOLDER")

# The dataset metadata folder (original dataset metadata)
METADATA_FOLDER = os.path.join(DATASET_FOLDER, "fma_metadata")
TRACKS_PATH = os.path.join(METADATA_FOLDER, "tracks.csv")

# The dataset csv file folder (edited dataset metadata)
CSV_FOLDER = os.path.join(os.getenv("CSV_FOLDER"))

# Original song audio files
AUDIO_FOLDER = os.path.join(DATASET_FOLDER, "fma_large")

# Stem audio files (vocals and instrumental)
UVR_MODEL_PATH = os.path.join(os.getenv("MODEL_FOLDER"), "UVR")
STEMS_FOLDER = os.path.join(DATASET_FOLDER, "fma_large_stems")