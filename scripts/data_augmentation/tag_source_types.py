"""Tag a QA split file with source_type from original_split.json.

Reads datasets/socialiq2/siq2/qa/qa_{split}.json and adds a `source_type`
column ("youtube", "movie", or "car") derived from original_split.json.
Saves the result to qa/qa_augmented_{split}.json.

Usage:
    python scripts/tag_source_types.py [--split val]
"""

import argparse
import json
import pandas as pd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR  = _REPO_ROOT / "datasets" / "socialiq2" / "siq2"

SUBSET_LABEL = {
    "youtubeclips": "youtube",
    "movieclips":   "movie",
    "car":          "car",
}


def run(split: str = "val") -> None:
    with open(_DATA_DIR / "original_split.json") as f:
        original_split = json.load(f)

    vid_to_source: dict[str, str] = {}
    for subset_name, splits in original_split["subsets"].items():
        label = SUBSET_LABEL[subset_name]
        for vid_ids in splits.values():
            for vid_id in vid_ids:
                vid_to_source[vid_id] = label

    qa_path = _DATA_DIR / "qa" / f"qa_{split}.json"
    qa = pd.read_json(qa_path, lines=True)
    qa["source_type"] = qa["vid_name"].map(vid_to_source)

    print(qa["source_type"].value_counts().to_string())
    print(f"\nUntagged: {qa['source_type'].isna().sum()}")

    out_path = _DATA_DIR / "qa" / f"qa_augmented_{split}.json"
    qa.to_json(out_path, orient="records", lines=True)
    print(f"Saved {len(qa)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag QA files with source_type")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Which split to tag (default: val)")
    args = parser.parse_args()
    run(split=args.split)
