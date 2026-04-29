"""Constants to be used across repo. This module should generally NOT import from other modules"""

import os
from enum import StrEnum, auto
from pathlib import Path

PATH_TO_DATA = Path("datasets/socialiq2/siq2/")
PATH_TO_AUGMENTED_DATA = Path("datasets/siq2long/")
RESULTS_DIR = "results/"
ORIGINAL_SPLITS_FILE = "original_split.json"

# GCS — set these in your .env file to enable cloud streaming
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_PREFIX = os.getenv("GCS_PREFIX", "siq2/video")


class Datasets(StrEnum):
    SIQ2 = "siq2"
    SIQ2LONG = "siq2long"

DATASET_TO_DIR = {
    Datasets.SIQ2: PATH_TO_DATA,
    Datasets.SIQ2LONG: PATH_TO_AUGMENTED_DATA
}

class DirPaths(StrEnum):
    TRANSCRIPT = auto()
    QA = auto()
    FRAMES = auto()
    AUDIO = auto()
    VIDEO = auto()

class SIQDatasetColumns(StrEnum):
    QID = "qid"
    VIDEO_ID = "vid_name"
    VIDEO_RAW = "video_raw"
    VIDEO = "video"
    QUESTION = "q"
    ANSWER_0 = "a0"
    ANSWER_1 = "a1"
    ANSWER_2 = "a2"
    ANSWER_3 = "a3"

class PipelineNames(StrEnum):
    LANGUAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
DEFAULT_MODEL_PROVIDER = "openai"
