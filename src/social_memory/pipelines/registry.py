"""CLI name -> concrete Pipeline class. Add new pipelines here only."""

from __future__ import annotations

from typing import Type

from social_memory.constants import PipelineNames
from social_memory.pipelines.base import BasePipeline
from social_memory.pipelines.audio_baseline import AudioPipeline
from social_memory.pipelines.text_baseline import LanguagePipeline
from social_memory.pipelines.sensory_deprivation import SDPipeline
from social_memory.pipelines.video_baseline import VideoPipeline
from social_memory.pipelines.video_clipped import VideoClippedPipeline

PIPELINE_REGISTRY: dict[str, Type[BasePipeline]] = {
    PipelineNames.LANGUAGE: LanguagePipeline,
    PipelineNames.VIDEO: VideoPipeline,
    PipelineNames.AUDIO: AudioPipeline,
    "sensory_deprivation": SDPipeline,
    "video_clipped": VideoClippedPipeline,
}
