"""
Prepare contrastive training data from the SIQ2Long dataset.

For each question, creates:
  - 1 positive:        (question, oracle_chunk)
  - 2 hard negatives:  (question, non-oracle chunks from same video), sampled randomly
  - 4 easy negatives:  (question, chunks from 4 different other videos), sampled randomly

Outputs (relative to repo root):
  datasets/siq2long/train_contrastive.jsonl  — one record per question, used by training
  datasets/siq2long/train_contrastive.csv    — flattened rows with a 'role' column, for inspection tasks.. 
"""

import ast
import csv
import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
N_HARD = 2
N_EASY = 4

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "datasets" / "siq2long"
OUTPUT_DIR = DATA_DIR

def chunk_path(vid_name: str, chunk_idx: int) -> str:
    return f"{vid_name}_chunk_{chunk_idx:03d}.mp4"

def main() -> None:
    random.seed(SEED)

    df = pd.read_csv(DATA_DIR / "dataset.csv")
    df["chunk_ids"] = df["chunk_ids"].apply(ast.literal_eval)

    # answer_idx is NaN for test-split questions — keep as-is, handle per row below

    # Drop questions with <= 7 words —  893 questions
    n_before = len(df)
    df = df[df["q"].str.split().str.len() > 7].reset_index(drop=True)
    print(f"Dropped {n_before - len(df):,} short questions (<= 7 words), {len(df):,} remaining")

    # Drop questions with timestamps (e.g. "at 0:25") - 173 questions
    n_before = len(df)
    df = df[~df["q"].str.contains(r"\d+:\d+", regex=True)].reset_index(drop=True)
    print(f"Dropped {n_before - len(df):,} timestamp questions, {len(df):,} remaining")

    chunks_by_video: dict[str, list[int]] = (
        df.drop_duplicates("vid_name")
        .set_index("vid_name")["chunk_ids"]
        .to_dict()
    )
    all_videos = list(chunks_by_video.keys())

    records = []

    for _, row in df.iterrows():
        vid = row["vid_name"]
        oracle_idx = int(row["oracle_idx"])
        chunk_ids: list[int] = row["chunk_ids"]

        # Positive
        positive = chunk_path(vid, oracle_idx)

        # Hard negatives: randomly sample
        hard_pool = [c for c in chunk_ids if c != oracle_idx]
        hard_neg_paths = [chunk_path(vid, c) for c in random.choices(hard_pool, k=N_HARD)]

        # Easy negatives
        other_videos = [v for v in all_videos if v != vid]
        sampled_videos = random.sample(other_videos, k=N_EASY)
        easy_neg_paths = [
            chunk_path(v, random.choice(chunks_by_video[v])) for v in sampled_videos
        ]

        options = [row["a0"], row["a1"], row["a2"], row["a3"]]
        answer_idx = int(row["answer_idx"]) if pd.notna(row["answer_idx"]) else None
        records.append({
            "qid": row["qid"],
            "vid_name": vid,
            "question": row["q"],
            "options": options,
            "answer_idx": answer_idx,
            "answer": options[answer_idx] if answer_idx is not None else None,
            "positive": positive,
            "hard_negatives": hard_neg_paths,
            "easy_negatives": easy_neg_paths,
        })

    # JSONL — one record per question
    jsonl_path = OUTPUT_DIR / "train_contrastive.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records):,} records → {jsonl_path}")

    # CSV — flattened rows with role column, probably for better inspection
    csv_path = OUTPUT_DIR / "train_contrastive.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "vid_name", "question", "answer_idx", "chunk_path", "role"])
        for rec in records:
            base = [rec["qid"], rec["vid_name"], rec["question"], rec["answer_idx"]]
            writer.writerow([*base, rec["positive"], "positive"])
            for path in rec["hard_negatives"]:
                writer.writerow([*base, path, "hard_negative"])
            for path in rec["easy_negatives"]:
                writer.writerow([*base, path, "easy_negative"])
    print(f"Wrote flattened CSV  → {csv_path}")


if __name__ == "__main__":
    main()
