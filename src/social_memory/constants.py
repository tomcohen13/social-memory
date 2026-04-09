"""Constants to be used across repo. This module should generally NOT import from other modules"""

from enum import StrEnum, auto

PATH_TO_DATA = "datasets/socialiq2/siq2/"
RESULTS_DIR = "results/"

class DirPaths(StrEnum):
    TRANSCRIPT = auto()
    QA = auto()
    FRAMES = auto()
    AUDIO = auto()

class PipelineNames(StrEnum):
    LANGUAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
DEFAULT_MODEL_PROVIDER = "openai"
