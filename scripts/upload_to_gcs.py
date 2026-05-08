"""Download the full SIQ2 dataset and upload full videos to a GCS bucket.

Each video is:
  1. Downloaded from YouTube to a temp dir via yt-dlp.
  2. Uploaded as-is to gs://<bucket>/<prefix>/<vid_id>.mp4.
  3. Cleaned up locally.

Already-uploaded blobs are skipped, so the script is fully resumable.
Failed IDs are saved to --failed_ids_file for easy retrying.

Usage:
    # Full run
    python scripts/upload_to_gcs.py --bucket my-bucket

    # Retry only failed IDs from previous run
    python scripts/upload_to_gcs.py --bucket my-bucket --video_ids_file failed_ids.json \\
        --max_workers 1 --cookies_from_browser chrome
"""

from __future__ import annotations

import argparse
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from yt_dlp import YoutubeDL

from social_memory.constants import GCS_TRANSCRIPT_PREFIX, ORIGINAL_SPLITS_FILE
from social_memory.gcs import blob_exists, transcript_blob_name, upload_file, video_blob_name

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR  = _REPO_ROOT / "datasets" / "socialiq2" / "siq2"


# ---------------------------------------------------------------------------
# Download / upload
# ---------------------------------------------------------------------------

def _download_transcript(vid_id: str, cache_path: str, cookies_from_browser: str | None) -> Path | None:
    """Download auto-generated VTT subtitle without downloading the video. Returns path or None."""
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,       # manual subtitles
        'writeautomaticsub': True,    # auto-generated subtitles
        'subtitlesformat': 'vtt',
        'subtitleslangs': ['en.*', 'en'],
        'outtmpl': f'{cache_path}/%(id)s',
        'sleep_interval': 1,
        'quiet': True,
        'ignoreerrors': True,
    }
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(vid_id, download=True, ie_key='Youtube')
        # Accept any .vtt file created (lang code varies for manual subs)
        vtts = sorted(Path(cache_path).glob(f'{vid_id}*.vtt'))
        return vtts[0] if vtts else None
    except Exception:
        return None


def _download_video(vid_id: str, cache_path: str, cookies_from_browser: str | None) -> str | None:

    ydl_opts = {
        'format': 'best[height<=360][ext=mp4]',
        'outtmpl': f'{cache_path}/%(id)s.%(ext)s',
        'retries': 3,
        'ignoreerrors': True,
        'source_address': '0.0.0.0',
        'quiet': True,
    }
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(vid_id, download=True, ie_key='Youtube')
        out = f'{cache_path}/{vid_id}.mp4'
        return out if Path(out).exists() else None
    except Exception:
        return None


def _process_video(
    vid_id: str,
    bucket_name: str,
    prefix: str,
    transcript_prefix: str,
    cookies_from_browser: str | None,
    transcripts_only: bool,
) -> str:
    """Download and upload one video + transcript. Returns vid_id on success, '' on failure."""

    t_blob = transcript_blob_name(vid_id, transcript_prefix)

    with tempfile.TemporaryDirectory() as tmp:
        if not transcripts_only:
            blob = video_blob_name(vid_id, prefix)
            if not blob_exists(bucket_name, blob):
                full_video = _download_video(vid_id, tmp, cookies_from_browser)
                if full_video is None:
                    return ""
                upload_file(bucket_name, Path(full_video), blob)

        if transcripts_only or not blob_exists(bucket_name, t_blob):
            vtt = _download_transcript(vid_id, tmp, cookies_from_browser)
            if vtt is not None:
                upload_file(bucket_name, vtt, t_blob)

    return vid_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_vid_ids_from_split(data_dir: Path, splits: list[str]) -> list[str]:
    """Load video IDs from the original_split.json for the specified splits."""
    with open(data_dir / ORIGINAL_SPLITS_FILE) as f:
        split_data = json.load(f)
    vid_ids: list[str] = []
    for subset in split_data["subsets"].values():
        for split_name, ids in subset.items():
            if split_name in splits:
                vid_ids.extend(ids)
    return list(dict.fromkeys(vid_ids)) 


def run(
    bucket_name: str,
    prefix: str,
    transcript_prefix: str,
    vid_ids: list[str],
    max_workers: int,
    cookies_from_browser: str | None,
    failed_ids_file: str,
    transcripts_only: bool,
) -> None:
    mode = "transcripts only (replacing existing)" if transcripts_only else "video + transcript"
    print(f"Mode: {mode} | videos: {len(vid_ids)} | bucket: gs://{bucket_name}/")

    succeeded, failed = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_video, vid_id, bucket_name, prefix, transcript_prefix,
                cookies_from_browser, transcripts_only,
            ): vid_id
            for vid_id in vid_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            vid_id = futures[future]
            try:
                result = future.result()
                (succeeded if result else failed).append(vid_id)
            except Exception as exc:
                print(f"\n[{vid_id}] error: {exc}")
                failed.append(vid_id)

    print(f"\nDone: {len(succeeded)} succeeded, {len(failed)} failed.")

    if failed:
        Path(failed_ids_file).write_text(json.dumps(failed, indent=2))
        print(f"Failed IDs saved to {failed_ids_file}")
        retry_flags = "--transcripts_only " if transcripts_only else ""
        print(f"\nTo retry:\n  python scripts/upload_to_gcs.py --bucket {bucket_name} "
              f"{retry_flags}--video_ids_file {failed_ids_file} --max_workers 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload SIQ2 dataset to a GCS bucket")
    parser.add_argument("--bucket",               required=True,
                        help="GCS bucket name")
    parser.add_argument("--prefix",               default="siq2/video",
                        help="GCS video prefix (default: siq2/video)")
    parser.add_argument("--transcript_prefix",    default=GCS_TRANSCRIPT_PREFIX,
                        help=f"GCS transcript prefix (default: {GCS_TRANSCRIPT_PREFIX})")
    parser.add_argument("--transcripts_only",     action="store_true",
                        help="Skip videos; only download and upload transcripts "
                             "(overwrites existing blobs — use to fix empty VTTs)")

    # Source of video IDs: either a file of specific IDs, or the full split
    id_source = parser.add_mutually_exclusive_group()
    id_source.add_argument("--video_ids_file",    default=None,
                           help="JSON file with a list of video IDs to process "
                                "(e.g. failed_ids.json from a previous run)")
    id_source.add_argument("--splits",            nargs="+", default=["train", "val", "test"],
                           help="Which splits to include (default: all). "
                                "Ignored if --video_ids_file is set.")

    parser.add_argument("--max_workers",           type=int, default=1,
                        help="Parallel download/upload workers (default: 1)")
    parser.add_argument("--data_dir",             default=str(_DATA_DIR),
                        help="Path to the siq2 data directory")
    parser.add_argument("--cookies_from_browser", default=None,
                        help="Pass cookies from a browser to bypass bot detection "
                             "(e.g. chrome, firefox, safari)")
    parser.add_argument("--failed_ids_file",      default="failed_ids.json",
                        help="Where to save failed IDs (default: failed_ids.json)")
    args = parser.parse_args()

    if args.video_ids_file:
        vid_ids = json.loads(Path(args.video_ids_file).read_text())
        print(f"Loaded {len(vid_ids)} IDs from {args.video_ids_file}")
    else:
        vid_ids = _load_vid_ids_from_split(Path(args.data_dir), args.splits)

    run(
        bucket_name=args.bucket,
        prefix=args.prefix,
        transcript_prefix=args.transcript_prefix,
        vid_ids=vid_ids,
        max_workers=args.max_workers,
        cookies_from_browser=args.cookies_from_browser,
        failed_ids_file=args.failed_ids_file,
        transcripts_only=args.transcripts_only,
    )
