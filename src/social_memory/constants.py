"""Constants to be used across repo. This module should generally NOT import from other modules"""

from enum import StrEnum, auto

PATH_TO_DATA = "datasets/socialiq2/siq2/"
PATH_TO_AUGMENTED_DATA = "datasets/validation_augmented/"
RESULTS_DIR = "results/"

class DirPaths(StrEnum):
    TRANSCRIPT = auto()
    QA = auto()
    FRAMES = auto()
    AUDIO = auto()
    VIDEO = auto()

class SocialIQDatasetColumns(StrEnum):
    QID = "qid"
    VIDEO_ID = "vid_name"
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



