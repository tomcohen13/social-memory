"""LLM utils"""

from __future__ import annotations

import pandas as pd
import subprocess
import webvtt

from collections import UserDict
from pathlib import Path
from typing import List


from social_memory.constants import (
    DATASET_TO_DIR,
    DirPaths,
    SIQDatasetColumns,
)

def load_dataset(dataset_name: str, split: str) -> pd.DataFrame:

    split_qids = load_qa_dataset(dataset_name, split, with_oracle=False)["qid"].unique()
    df = pd.read_csv(DATASET_TO_DIR.get(dataset_name) / "dataset.csv").drop_duplicates("qid")
    df["num_chunks"] = df["chunk_ids"].apply(lambda x: len(eval(x)))
    df["oracle_idx"] = df["oracle_idx"].apply(int)
    return df[df["qid"].isin(split_qids)].reset_index(drop=True)

def load_qa_dataset(dataset: str, split: str, with_oracle: bool = True) -> pd.DataFrame:
    """
    Load QA dataset from json file
    """
    path = DATASET_TO_DIR.get(dataset) / DirPaths.QA / f"qa_{split}.json"
    print(f"trying to read file: {path}...")
    qa = pd.read_json(path, lines=True)
    # if with_oracle:
    #     import json
    #     with open(PATH_TO_DATA / "trims.json", "r") as j:
    #         oracles = json.load(j)
    #         oracles = {vid: (start, start + 60.0) for vid, start in oracles.items()}
    #         qa["oracle"] = qa[SIQDatasetColumns.VIDEO_ID].map(oracles)
    return qa


def read_vtt_file(vtt_path: Path) -> str:
    return "\n".join(caption.text for caption in webvtt.read(str(vtt_path)))


def group_inputs_by_video_id(inputs: List[dict]) -> List[dict]:
    """
    Group inputs by video id.


    Example:
    >> inputs = [{'vid_name': 1, 'q': 'why?'}, {'vid_name': 1, 'q': 'what?'}, {'vid_name': 2, 'q': 'how?'}]
    >> group_inputs_by_video_id(inputs)
    [
        {'vid_name': 1, 'questions': ['why?', 'what?']},
        {'vid_name': 2, 'q': 'how?'}
    ]
    """
    inputs_df = pd.DataFrame.from_records(inputs)
    grouped_df = inputs_df.groupby(SIQDatasetColumns.VIDEO_ID).agg(
        {"vid_name": "first", "q": list, "qid": list, "oracle_idx": "first", "num_chunks": "first"}
    )
    grouped_df = grouped_df.rename(columns={"q": "questions"})
    grouped_inputs = grouped_df.to_dict(orient="records")
    return grouped_inputs


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
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or "").strip() or "ffprobe failed with no stderr"
        raise RuntimeError(f"ffprobe exited {result.returncode}: {err}")
    raw = (result.stdout or "").strip()
    if not raw:
        raise RuntimeError("ffprobe returned empty duration")
    return float(raw)

def check_device() -> str:
    import torch
    """Checks if MPS is available and returns the device string"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class LRUCache(UserDict):
    def __init__(self, max_size=10):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        # If the key is new and cache is full, remove the first inserted item
        if key not in self.data and len(self.data) >= self.max_size:
            first_key = next(iter(self.data))
            del self.data[first_key]
        
        # Move key to the end by deleting and re-inserting if it already exists
        if key in self.data:
            del self.data[key]
            
        self.data[key] = value

    def __getitem__(self, key):
        # Move accessed item to the end (making it most recently used)
        value = self.data[key]
        del self.data[key]
        self.data[key] = value
        return value
