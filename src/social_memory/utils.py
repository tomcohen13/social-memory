"""LLM utils"""

from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
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


def _load_single_video(vid: str, directory: Path, with_audio: bool = True) -> tuple[str, str]:
    """
    Load a single .mp4 video file as a base64-encoded string.

    Args:
        vid: Video ID (filename without extension).
        directory: Directory containing the .mp4 files.
        with_audio: If True, includes audio. If False, returns video only (audio removed).

    Returns:
        Tuple of (video ID, base64-encoded video string). If the file is missing, the value is an empty string.
    """
    path = directory / f"{vid}.mp4"
    if not path.is_file():
        return vid, ""

    if with_audio:
        with open(path, "rb") as f:
            return vid, base64.b64encode(f.read()).decode("utf-8")

    # Return muted (no audio) video
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(path),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-movflags", "+frag_keyframe+empty_moov",
            "-f", "mp4",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return vid, base64.b64encode(result.stdout).decode("utf-8")


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
    directory = Path(PATH_TO_DATA) / str(DirPaths.VIDEO)

    if not video_ids:
        return {}

    worker = partial(_load_single_video, directory=directory, with_audio=with_audio)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(worker, video_ids))


def compute_correctness(df):
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
