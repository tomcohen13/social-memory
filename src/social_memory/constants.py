"""Constants to be used across repo. This module should generally NOT import from other modules"""

from typing import Mapping
from enum import StrEnum, auto

PATH_TO_DATA = "datasets/socialiq2/siq2/"

class DirPaths(StrEnum):
    TRANSCRIPT = auto()
    QA = auto()
    FRAMES = auto()
    AUDIO = auto()


PROVIDER_TO_API_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", #"ANTHROPIC_API_KEY",
    "google": "OPENROUTER_API_KEY",  # Using OpenRouter for Google models
    "deepseek": "OPENROUTER_API_KEY",  # Using OpenRouter for Google models
    "qwen": "OPENROUTER_API_KEY",  # Using OpenRouter for Alibaba models
    "mistral": "OPENROUTER_API_KEY",  # Using OpenRouter for Mistral models
    "meta-llama": "OPENROUTER_API_KEY",  # Using OpenRouter for Meta models
}

MODELS_TO_TEST = {
    "anthropic": [],
    "openai": [],
    "qwen": [],
}