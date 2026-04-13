"""Add full-video timestamps and durations to siq2_augmented/qa_augmented.json.

For each row in siq2_augmented/qa_augmented.json:
  - Maps the 60-second clip timestamp (`ts`) to the corresponding range in the
    full video using datasets/socialiq2/siq2/trims.json.
  - Rewrites any "at MM:SS" references in the question text to the correct
    full-video time.
  - Fetches the total video duration from YouTube via yt-dlp.

Results are saved back to siq2_augmented/qa_augmented.json with new columns:
  full_video_ts, full_video_duration, augmented_q (timestamp-shifted).

Usage:
    python scripts/add_video_metadata.py
"""

import json
import re
import pandas as pd
from pathlib import Path
from yt_dlp import YoutubeDL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR  = _REPO_ROOT / "datasets" / "socialiq2" / "siq2"
_AUG_DIR   = _REPO_ROOT / "datasets" / "socialiq2" / "siq2_augmented"


def get_total_duration(vid_id: str) -> int | None:
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid_id}", download=False)
        return info.get("duration") if info else None


def clip_ts_to_full_video(vid_name: str, ts_str: str, trims: dict) -> tuple[str, float | None]:
    """Map a clip-relative timestamp range to the full-video range."""
    clip_end = float(ts_str.split("-")[1])
    trim_start = trims.get(vid_name)
    if trim_start is None:
        return ts_str, None
    trim_end = trim_start + clip_end
    return f"{trim_start:.2f}-{trim_end:.2f}", float(trim_start)


def shift_question_timestamps(q: str, trim_start: float) -> str:
    """Rewrite 'at MM:SS' references in question text to full-video times."""
    def _shift(match: re.Match) -> str:
        raw = match.group(1)
        parts = raw.split(":")
        secs = int(parts[0]) * 60 + int(parts[1])
        new_secs = secs + trim_start
        new_m = int(new_secs) // 60
        new_s = int(new_secs) % 60
        return f"at {new_m}:{new_s:02d}"

    return re.sub(r"at (\d+:\d{2})", _shift, q)


def run() -> None:
    trims = json.loads((_DATA_DIR / "trims.json").read_text())
    qa_path = _AUG_DIR / "qa_augmented.json"
    qa = pd.read_json(qa_path, lines=True)

    unique_vids = qa["vid_name"].unique().tolist()
    print(f"Fetching durations for {len(unique_vids)} videos...")
    vid_total_duration = {vid: get_total_duration(vid) for vid in unique_vids}

    full_ts_list, aug_qs, total_durs = [], [], []
    for _, row in qa.iterrows():
        full_ts, trim_start = clip_ts_to_full_video(row["vid_name"], row["ts"], trims)
        full_ts_list.append(full_ts)
        aug_qs.append(
            shift_question_timestamps(row["q"], trim_start) if trim_start is not None else row["q"]
        )
        total_durs.append(vid_total_duration.get(row["vid_name"]))

    qa["augmented_q"]         = aug_qs
    qa["full_video_ts"]       = full_ts_list
    qa["full_video_duration"] = total_durs

    col_order = [
        "qid", "vid_name", "source_type",
        "q", "augmented_q",
        "ts", "full_video_ts", "full_video_duration",
        "ans_corr", "answer_idx", "idx_types",
        "a0", "a1", "a2", "a3",
        "result", "is_correct",
    ]
    qa = qa[[c for c in col_order if c in qa.columns]]

    changed = (qa["q"] != qa["augmented_q"]).sum()
    print(f"Questions with timestamp refs rewritten: {changed}")

    qa.to_json(qa_path, orient="records", lines=True)
    print(f"Saved -> {qa_path}")


if __name__ == "__main__":
    run()
