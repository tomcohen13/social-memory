"""CLI name -> concrete Pipeline class. Add new pipelines here only."""

from __future__ import annotations

from typing import Type

from social_memory.pipelines.base import BasePipeline
from social_memory.pipelines.audio_baseline import AudioPipeline
from social_memory.pipelines.lang_baseline import LanguagePipeline
from social_memory.pipelines.mcq_validation import MCQValidationPipeline
from social_memory.pipelines.video_baseline import VideoPipeline

PIPELINE_REGISTRY: dict[str, Type[BasePipeline]] = {
    "language": LanguagePipeline,
    "video": VideoPipeline,
    "audio": AudioPipeline,
    "mcq_validation": MCQValidationPipeline,
}
