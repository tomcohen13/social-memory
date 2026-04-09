"""LLM utils"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable

import pandas as pd
import webvtt
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from social_memory.constants import PATH_TO_DATA, PROVIDER_TO_API_KEY_MAP, DirPaths


def init_model(
    model: str,
    provider: str,
    api_key_provider: str | None = None,
    rate_limiter = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> BaseChatModel:
    """
    Initialize a chat model based on the provider and model name.

    Parameters:
        model: model name (e.g., "gpt-4o-mini")
        provider: provider name (e.g., "openai")
        api_key_provider: provider of the API key (e.g., "openai" or "openrouter")
        rate_limiter: rate limiter to use (default: None)
        temperature: temperature [0.0,1.0)
        max_retries: maximum number of retries (default: 3)
    """

    if rate_limiter is None:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.1,
            check_every_n_seconds=0.5,
            max_bucket_size=10,
        )

    base_model_configs = {
        "max_retries": max_retries,
        "rate_limiter": rate_limiter,
        "temperature": temperature,
    }

    # TODO: change to ChatOpenAI for all given OpenRouter rate limit >> Anthropic's
    if provider in ["openai", "anthropic"] and api_key_provider != "openrouter":
        return init_chat_model(
            model=model,
            model_provider=provider,
            **base_model_configs
        )

    else:
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1", # OpenRouter base URL
            api_key=os.getenv(PROVIDER_TO_API_KEY_MAP[provider]),
            model=f"{provider}/{model}",
            **base_model_configs,
        )


def load_qa_dataset(split: str) -> pd.DataFrame:
    """
    Load QA dataset from json file
    """
    path = os.path.join(PATH_TO_DATA, DirPaths.QA, f"qa_{split}.json")
    print(f"trying to read file: {path}...")
    qa = pd.read_json(path, lines=True)
    return qa

def _read_vtt_file(vtt_path: Path) -> str:
    return "\n".join(caption.text for caption in webvtt.read(str(vtt_path)))


def _load_single_transcript(vid: str, directory: Path) -> tuple[str, str]:
    path = directory / f"{vid}.vtt"
    if not path.is_file():
        return vid, ""
    return vid, _read_vtt_file(path)


def load_transcripts(
    video_ids: Iterable[str],
    max_workers: int | None = 4,
) -> dict[str, str]:
    """
    Loads .vtt transcript files for a list of video IDs.

    For each video ID, attempts to read the corresponding .vtt file from the Social-IQ transcript directory.
    Returns a dictionary mapping each video ID to its transcript text. If a transcript is missing,
    the value will be an empty string.

    Args:
        video_ids: An iterable of video IDs to look for .vtt files.
        max_workers: Maximum number of threads to use for reading files.

    Returns:
        A dictionary with video IDs as keys and transcript strings as values.
    """
    directory = Path(PATH_TO_DATA) / str(DirPaths.TRANSCRIPT)
    
    if not video_ids:
        return {}

    worker = partial(_load_single_transcript, directory=directory)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(worker, video_ids))
