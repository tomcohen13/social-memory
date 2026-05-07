"""Build a unified dataset table from SIQ2 QA files + video chunk definitions.

Output schema (one JSON object per line):
  qid         — question ID
  vid_name    — video ID
  q           — question text
  a0..a3      — answer choices
  answer_idx  — index of correct answer
  chunk_ids   — list of chunk indices for this video (null if no chunks defined)
  oracle_idx  — index of the chunk containing the oracle clip (null if no chunks defined)

Usage:
    python scripts/create_dataset.py
    python scripts/create_dataset.py --output datasets/siq2long/dataset.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_QA_DIR     = _REPO_ROOT / "datasets" / "socialiq2" / "siq2" / "qa"
_CHUNKS_FILE = _REPO_ROOT / "datasets" / "siq2long" / "video_chunks.json"
_ORACLES_FILE = _REPO_ROOT / "datasets" / "siq2long" / "oracles.json"
_DEFAULT_OUT = _REPO_ROOT / "datasets" / "siq2long" / "dataset.jsonl"

_QA_FILES = ["qa_train.json", "qa_val.json", "qa_test.json"]

def _load_qa() -> list[dict]:
    rows = []
    for fname in _QA_FILES:
        path = _QA_DIR / fname
        if not path.exists():
            print(f"  Warning: {fname} not found, skipping.")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build(output: Path) -> None:
    print("Loading QA files...")
    qa_rows = _load_qa()
    print(f"  {len(qa_rows)} questions loaded")

    print("Loading chunk definitions...")
    with open(_CHUNKS_FILE) as f:
        chunks_map = json.load(f)
    print(f"  {len(chunks_map)} videos with chunk definitions")

    print("Loading oracles...")
    with open(_ORACLES_FILE) as f:
        oracles = json.load(f)
    print(f"  {len(oracles)} oracle entries")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_chunks = 0

    records = []
    for row in qa_rows:
        vid = row["vid_name"]
        entry = chunks_map.get(vid)

        if not entry:
            skipped_chunks += 1
            continue

        records.append({
            "qid":        row["qid"],
            "vid_name":   vid,
            "q":          row["q"],
            "a0":         row["a0"],
            "a1":         row["a1"],
            "a2":         row["a2"],
            "a3":         row["a3"],
            "answer_idx": row.get("answer_idx"),
            "chunk_ids":  list(range(len(entry["chunks"]))),
            "oracle_idx": entry["oracle_idx"],
        })
        written += 1

    # JSONL
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # chunk_ids stored as JSON string since CSV is flat
    csv_path = output.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "chunk_ids": json.dumps(record["chunk_ids"])})

    print(f"\nWrote {written} records to:")
    print(f"  {output}")
    print(f"  {csv_path}")
    print(f"  Skipped (no chunk definition): {skipped_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified SIQ2-long dataset")
    parser.add_argument("--output", default=str(_DEFAULT_OUT),
                        help=f"Output JSONL path (default: {_DEFAULT_OUT})")
    args = parser.parse_args()

    build(Path(args.output))
