"""Select all-correct videos from inference results and download them.

Steps:
  1. Merge inference results into qa_augmented_val.json as `is_correct`.
  2. Identify videos where every question was answered correctly.
  3. Filter movie clips to those longer than --min_movie_duration seconds.
  4. Download up to --max_youtube YouTube and --max_movie movie clips.
  5. Save the selected QA rows to siq2_augmented/qa_augmented.json.

Requires GOOGLE_API_KEY to NOT be set (just yt-dlp, no Gemini needed here).

Usage:
    python scripts/select_and_download.py \\
        --results results/video-audio/google_genai:gemini-2.5-pro/val/results.jsonl \\
        [--min_movie_duration 120] \\
        [--max_youtube 10] \\
        [--max_movie 10]
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from yt_dlp import YoutubeDL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR  = _REPO_ROOT / "datasets" / "socialiq2" / "siq2"
_AUG_DIR   = _REPO_ROOT / "datasets" / "socialiq2" / "siq2_augmented"

# youtube_utils lives in the dataset directory
sys.path.insert(0, str(_DATA_DIR))
import youtube_utils  # noqa: E402


def get_duration(vid_id: str) -> int | None:
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid_id}", download=False)
        return info.get("duration") if info else None


def run(
    results_path: str,
    min_movie_duration: int = 120,
    max_youtube: int = 10,
    max_movie: int = 10,
) -> None:
    # --- Merge is_correct into qa_augmented_val ---
    results = pd.read_json(results_path, lines=True)
    qa_path = _DATA_DIR / "qa" / "qa_augmented_val.json"
    qa = pd.read_json(qa_path, lines=True)
    qa = qa.merge(results[["qid", "result"]], on="qid", how="left")
    qa["is_correct"] = qa["result"] == qa["answer_idx"]

    print(f"is_correct distribution:\n{qa['is_correct'].value_counts().to_string()}")
    print(f"Unmatched (no result): {qa['result'].isna().sum()}")

    out_qa = _DATA_DIR / "qa" / "qa_augmented.json"
    qa.to_json(out_qa, orient="records", lines=True)
    print(f"Updated {out_qa} with is_correct")

    # --- Find videos where ALL questions were answered correctly ---
    all_correct_vids = (
        qa.groupby("vid_name")["is_correct"]
        .all()
        .reset_index()
        .query("is_correct")["vid_name"]
    )
    perfect_vids = (
        qa[qa["vid_name"].isin(all_correct_vids)]
        [["vid_name", "source_type"]]
        .drop_duplicates("vid_name")
    )
    print(f"\nVideos with all questions correct: {len(perfect_vids)}")

    # --- Filter movie clips by duration ---
    movie_candidates = perfect_vids[perfect_vids["source_type"] == "movie"]["vid_name"].tolist()
    print(f"Fetching durations for {len(movie_candidates)} movie candidates...")
    movie_durations = {vid: get_duration(vid) for vid in movie_candidates}
    long_movies = [vid for vid, dur in movie_durations.items() if dur and dur > min_movie_duration]
    print(f"Movies > {min_movie_duration}s: {len(long_movies)} / {len(movie_candidates)}")

    youtube_picks = perfect_vids[perfect_vids["source_type"] == "youtube"].head(max_youtube)
    movie_picks   = perfect_vids[perfect_vids["vid_name"].isin(long_movies)].head(max_movie)
    selected_vids = youtube_picks["vid_name"].tolist() + movie_picks["vid_name"].tolist()
    print(f"Selected {len(selected_vids)} videos ({len(youtube_picks)} YouTube, {len(movie_picks)} movie)")

    # --- Save QA subset ---
    _AUG_DIR.mkdir(parents=True, exist_ok=True)
    video_dir = _AUG_DIR / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    selected_qa = qa[qa["vid_name"].isin(selected_vids)]
    qa_json_path = _AUG_DIR / "qa_augmented.json"
    selected_qa.to_json(qa_json_path, orient="records", lines=True)
    print(f"Saved {len(selected_qa)} QA rows -> {qa_json_path}")

    # --- Download videos ---
    failed = []
    for vid_id in selected_vids:
        out_path = video_dir / f"{vid_id}.mp4"
        if out_path.exists():
            print(f"[skip] {vid_id} already downloaded")
            continue
        print(f"[downloading] {vid_id} ...")
        result = youtube_utils.download_video(vid_id, str(video_dir), no_download=False)
        if result is None:
            print(f"  FAILED: {vid_id}")
            failed.append(vid_id)
        else:
            print(f"  OK -> {result}")

    print(f"\nDone. {len(selected_vids) - len(failed)}/{len(selected_vids)} downloaded.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select all-correct videos from inference results and download them"
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to inference results JSONL "
                             "(e.g. results/video-audio/google_genai:gemini-2.5-pro/val/results.jsonl)")
    parser.add_argument("--min_movie_duration", type=int, default=120,
                        help="Minimum duration (seconds) for movie clips (default: 120)")
    parser.add_argument("--max_youtube", type=int, default=10,
                        help="Max YouTube videos to select (default: 10)")
    parser.add_argument("--max_movie", type=int, default=10,
                        help="Max movie videos to select (default: 10)")
    args = parser.parse_args()

    run(
        results_path=args.results,
        min_movie_duration=args.min_movie_duration,
        max_youtube=args.max_youtube,
        max_movie=args.max_movie,
    )
