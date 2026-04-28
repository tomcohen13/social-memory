"""LLM utils"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
from typing import Iterable

import pandas as pd
import webvtt

from social_memory.constants import PATH_TO_DATA, DirPaths, SIQDatasetColumns

def load_qa_dataset(split: str, with_oracle: bool = True) -> pd.DataFrame:
    """
    Load QA dataset from json file
    """
    path = PATH_TO_DATA / DirPaths.QA / f"qa_{split}.json"
    print(f"trying to read file: {path}...")
    qa = pd.read_json(path, lines=True)
    if with_oracle:
        import json
        with open(PATH_TO_DATA / "trims.json", "r") as j:
            oracles = json.load(j)
            oracles = {vid: (start, start + 60.0) for vid, start in oracles.items()}
            qa["oracle"] = qa[SIQDatasetColumns.VIDEO_ID].map(oracles)
    return qa


def read_vtt_file(vtt_path: Path) -> str:
    return "\n".join(caption.text for caption in webvtt.read(str(vtt_path)))

# TODO: remove all load_Xs functions, we don't use them anymore.
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
    directory = PATH_TO_DATA / DirPaths.TRANSCRIPT
    
    if not video_ids:
        return {}

    worker = partial(_load_single_transcript, directory=directory)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(worker, video_ids))


def load_videos(
    video_ids: Iterable[str],
    max_workers: int | None = 4,
    with_audio: bool = True,
) -> dict[str, str]:
    """
    Loads .mp4 video files for a list of video IDs as base64-encoded strings.

    Returns a dictionary mapping each video ID to its base64-encoded video content.
    If a video file is missing, the value will be an empty string.

    Args:
        video_ids: An iterable of video IDs to load.
        max_workers: Maximum number of threads to use for reading files.

    Returns:
        A dictionary with video IDs as keys and base64 video strings as values.
    """
    directory = PATH_TO_DATA / DirPaths.VIDEO

    if not video_ids:
        return {}

    worker = partial(_load_single_video, directory=directory, with_audio=with_audio)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(worker, video_ids))


def load_audios(
    video_ids: Iterable[str],
    max_workers: int | None = 4,
) -> dict[str, tuple[str, str]]:
    """
    Finds audio files for a list of video IDs.

    Checks for mp3 then wav under the audio directory. Returns a dict mapping
    each video ID to a (file_path, mime_type) tuple. Missing files map to ("", "").

    Args:
        video_ids: An iterable of video IDs to load.
        max_workers: Maximum number of threads to use for reading files.

    Returns:
        A dictionary with video IDs as keys and (file_path, mime_type) tuples as values.
    """
    directory = PATH_TO_DATA / DirPaths.AUDIO

    if not video_ids:
        return {}

    worker = partial(_load_single_audio, directory=directory)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(worker, video_ids))


def load_audios(
    video_ids: Iterable[str],
    max_workers: int | None = 4,
) -> dict[str, tuple[str, str]]:
    """
    Finds audio files for a list of video IDs.

    Checks for mp3 then wav under the audio directory. Returns a dict mapping
    each video ID to a (file_path, mime_type) tuple. Missing files map to ("", "").

    Args:
        video_ids: An iterable of video IDs to load.
        max_workers: Maximum number of threads to use for reading files.

    Returns:
        A dictionary with video IDs as keys and (file_path, mime_type) tuples as values.
    """
    directory = PATH_TO_DATA / DirPaths.AUDIO

    if not video_ids:
        return {}

    worker = partial(_load_single_audio, directory=directory)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(worker, video_ids))


def compute_correctness(df: pd.DataFrame) -> float:
    """Compute fraction of valid predictions that match the gold answer index.

    Args:
        df: DataFrame with columns ``result`` (model predictions) and
            ``answer_idx`` (gold labels). Either column may contain ``pd.NA``
            or non-numeric values.

    Returns:
        Accuracy as a float in [0, 1], or ``pd.NA`` if no valid predictions exist.
    """
    # Coerce to numeric so int answer_idx vs object `result` (pd.NA + ints) does not hit
    # NAType in == (TypeError: boolean value of NA is ambiguous).
    predicted = pd.to_numeric(df["result"], errors="coerce")
    gold = pd.to_numeric(df["answer_idx"], errors="coerce")
    valid = predicted.notna()
    mask = valid & (gold == predicted)
    n_correct = int(mask.sum())
    n_total = int(valid.sum())
    if n_total == 0:
        return pd.NA
    return n_correct / n_total


def get_duration(filename: str) -> float:
    """Get duration of video file in seconds using ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filename]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


