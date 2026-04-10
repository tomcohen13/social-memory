"""CLI name -> concrete Pipeline class. Add new pipelines here only."""

from __future__ import annotations

from typing import Type

from social_memory.pipelines.base import Pipeline
from social_memory.pipelines.lang_baseline import LanguagePipeline

PIPELINE_REGISTRY: dict[str, Type[Pipeline]] = {
    "language": LanguagePipeline,
}
