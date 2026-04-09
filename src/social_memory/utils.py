"""LLM utils"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable

import pandas as pd
import webvtt

from social_memory.constants import PATH_TO_DATA, DirPaths


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


def compute_correctness(df):
    # adapted from Social-IQ repo
    mask = df['result'].notnull() & (df['answer_idx'] == df['result'])
    n_correct = mask.sum()
    n_total = df['result'].notnull().sum()
    if n_total == 0:
        return float('nan')
    return n_correct / n_total
