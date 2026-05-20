"""
Prepare contrastive data splits from the SIQ2Long dataset.

For each question, creates:
  - 1 positive:        (question, oracle_chunk)
  - hard negatives:    (question, non-oracle chunks from same video)
  - 4 easy negatives:  (question, chunks from 4 different other videos), sampled randomly

Outputs (relative to repo root):
  datasets/siq2long/train_contrastive.jsonl / .csv
  datasets/siq2long/val_contrastive.jsonl   / .csv
  datasets/siq2long/test_contrastive.jsonl  / .csv
"""

import ast
import csv
import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
N_EASY = 4

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "datasets" / "siq2long"
OUTPUT_DIR = DATA_DIR

SPLITS = ["train", "val", "test"]


def chunk_path(vid_name: str, chunk_idx: int) -> str:
    return f"{vid_name}_chunk_{chunk_idx:03d}.mp4"


def load_split_qids(split: str) -> set[str]:
    qids = set()
    with open(DATA_DIR / "qa" / f"qa_{split}.json") as f:
        for line in f:
            line = line.strip()
            if line:
                qids.add(json.loads(line)["qid"])
    return qids


def build_records(df: pd.DataFrame, all_videos: list[str], chunks_by_video: dict[str, list[int]]) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        vid = row["vid_name"]
        oracle_idx = int(row["oracle_idx"])
        chunk_ids: list[int] = row["chunk_ids"]

        hard_neg_paths = [chunk_path(vid, c) for c in chunk_ids if c != oracle_idx]

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
            "oracle_idx": oracle_idx,
            "oracle_path": chunk_path(vid, oracle_idx),
            "hard_negatives": hard_neg_paths,
            "easy_negatives": easy_neg_paths,
        })
    return records


def write_outputs(records: list[dict], split: str) -> None:
    jsonl_path = OUTPUT_DIR / f"{split}_contrastive.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records):,} records → {jsonl_path}")

    csv_path = OUTPUT_DIR / f"{split}_contrastive.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "vid_name", "question", "answer_idx", "chunk_path", "role"])
        for rec in records:
            base = [rec["qid"], rec["vid_name"], rec["question"], rec["answer_idx"]]
            writer.writerow([*base, rec["oracle_path"], "positive"])
            for path in rec["hard_negatives"]:
                writer.writerow([*base, path, "hard_negative"])
            for path in rec["easy_negatives"]:
                writer.writerow([*base, path, "easy_negative"])
    print(f"Wrote flattened CSV  → {csv_path}")


def process_split(split: str, df_full: pd.DataFrame) -> None:
    print(f"\n=== {split} ===")
    qids = load_split_qids(split)
    df = df_full[df_full["qid"].isin(qids)].reset_index(drop=True)
    print(f"Kept {len(df):,} / {len(df_full):,} questions that appear in qa_{split}")

    n_before = len(df)
    df = df[df["q"].str.split().str.len() > 8].reset_index(drop=True)
    print(f"Dropped {n_before - len(df):,} short questions (<= 8 words), {len(df):,} remaining")

    n_before = len(df)
    df = df[~df["q"].str.contains(r"\d+:\d+", regex=True)].reset_index(drop=True)
    print(f"Dropped {n_before - len(df):,} timestamp questions, {len(df):,} remaining")

    chunks_by_video: dict[str, list[int]] = (
        df.drop_duplicates("vid_name")
        .set_index("vid_name")["chunk_ids"]
        .to_dict()
    )
    all_videos = list(chunks_by_video.keys())

    records = build_records(df, all_videos, chunks_by_video)
    write_outputs(records, split)


def main() -> None:
    random.seed(SEED)

    df_full = pd.read_csv(DATA_DIR / "dataset.csv")
    df_full["chunk_ids"] = df_full["chunk_ids"].apply(ast.literal_eval)

    for split in SPLITS:
        process_split(split, df_full)


if __name__ == "__main__":
    main()
