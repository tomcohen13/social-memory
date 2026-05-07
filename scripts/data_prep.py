"""Download videos from GCS, chunk them using video_chunks.json, and clean up.

Each video is:
  1. Downloaded from gs://<bucket>/<prefix>/<vid_id>.mp4 to a temp dir.
  2. Split into chunks using the [start, end] entries in video_chunks.json.
  3. The full video is deleted; only the chunks are kept.
  4. Chunks are cleaned up on exit unless --keep is set.

Already-downloaded chunks are skipped, so the script is fully resumable.
Failed IDs are saved to --failed_ids_file for easy retrying.

Usage:
    # All splits (reads bucket/prefix from .env)
    python scripts/download_from_gcs.py --splits train val test

    # Retry only failed IDs
    python scripts/download_from_gcs.py --video_ids_file failed_downloads.json

    # Keep chunks after script exits
    python scripts/download_from_gcs.py --splits train --keep --output_dir /tmp/chunks
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import webvtt

from social_memory.gcs import video_blob_name
from social_memory.constants import GCS_BUCKET, GCS_PREFIX, ORIGINAL_SPLITS_FILE, PATH_TO_AUGMENTED_DATA, PATH_TO_DATA

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "datasets" / "socialiq2" / "siq2"
_CHUNKS_FILE = _REPO_ROOT / PATH_TO_AUGMENTED_DATA / "video_chunks.json"
_TRANSCRIPT_DIR = _REPO_ROOT / PATH_TO_DATA / "transcript"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _cut_chunk(src: Path, start: float, end: float, dest: Path) -> None:
    """Cut [start, end] from src into dest using ffmpeg (-c copy, no re-encode)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(src),
            "-c", "copy",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _slice_transcript(vid_id: str, start: float, end: float) -> str:
    """Return caption text from the existing VTT for the [start, end] window."""
    vtt_path = _TRANSCRIPT_DIR / f"{vid_id}.vtt"
    if not vtt_path.exists():
        return ""
    captions = [
        c for c in webvtt.read(str(vtt_path))
        if c.start_in_seconds >= start and c.start_in_seconds < end
    ]
    return "\n".join(c.text for c in captions)


def _chunk_video(vid_id: str, video_path: Path, chunks: list[list[float]], output_dir: Path) -> list[Path]:
    """Split a video into chunks and write matching transcript slices. Returns chunk video paths."""
    created = []
    for i, (start, end) in enumerate(chunks):
        dest = output_dir / f"{vid_id}_chunk_{i:03d}.mp4"
        transcript_dest = output_dir / f"{vid_id}_chunk_{i:03d}.txt"

        if not dest.exists():
            _cut_chunk(video_path, start, end, dest)

        if not transcript_dest.exists():
            transcript_dest.write_text(_slice_transcript(vid_id, start, end), encoding="utf-8")

        created.append(dest)
    return created


# ---------------------------------------------------------------------------
# Download + chunk
# ---------------------------------------------------------------------------

def _download_blob(bucket_name: str, blob_name: str, dest_path: Path) -> bool:
    """Download a single GCS blob to dest_path. Returns True on success."""
    from google.cloud import storage

    if dest_path.exists():
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            return False
        blob.download_to_filename(str(dest_path))
        return True
    except Exception:
        return False


def _process_video(
    vid_id: str,
    bucket_name: str,
    prefix: str,
    output_dir: Path,
    chunks_map: dict,
) -> str:
    """Download one video, chunk it, delete the full video. Returns vid_id on success, '' on failure."""
    chunks = chunks_map.get(vid_id, {}).get("chunks")
    if not chunks:
        return ""  # no chunk definition for this ID

    # Check if all chunks already exist — skip download entirely
    all_chunk_paths = [output_dir / f"{vid_id}_chunk_{i:03d}.mp4" for i in range(len(chunks))]
    if all(p.exists() for p in all_chunk_paths):
        return vid_id

    blob_name = video_blob_name(vid_id, prefix)
    tmp_video = output_dir / f"_tmp_{vid_id}.mp4"

    try:
        if not _download_blob(bucket_name, blob_name, tmp_video):
            return ""
        _chunk_video(vid_id, tmp_video, chunks, output_dir)
    finally:
        tmp_video.unlink(missing_ok=True)  # always clean up full video

    return vid_id


# ---------------------------------------------------------------------------
# Prefix-listing mode (no vid IDs)
# ---------------------------------------------------------------------------

def _list_blobs(bucket_name: str, prefix: str) -> list[str]:
    from google.cloud import storage
    client = storage.Client()
    return [b.name for b in client.list_blobs(bucket_name, prefix=prefix)]


def _download_blob_to_relative(bucket_name: str, blob_name: str, prefix: str, output_dir: Path) -> bool:
    rel = Path(blob_name).relative_to(prefix.rstrip("/"))
    return _download_blob(bucket_name, blob_name, output_dir / rel)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vid_ids_from_split(data_dir: Path, splits: list[str]) -> list[str]:
    with open(data_dir / ORIGINAL_SPLITS_FILE) as f:
        split_data = json.load(f)
    vid_ids: list[str] = []
    for subset in split_data["subsets"].values():
        for split_name, ids in subset.items():
            if split_name in splits:
                vid_ids.extend(ids)
    return list(dict.fromkeys(vid_ids))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    bucket_name: str,
    prefix: str,
    output_dir: Path,
    vid_ids: list[str] | None,
    chunks_map: dict,
    max_workers: int,
    keep: bool,
    failed_ids_file: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if vid_ids is not None:
        print(f"Videos to process: {len(vid_ids)} | source: gs://{bucket_name}/{prefix}/")
        print(f"Output: {output_dir}")

        succeeded, failed = [], []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_video, vid_id, bucket_name, prefix, output_dir, chunks_map): vid_id
                for vid_id in vid_ids
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading + chunking"):
                vid_id = futures[future]
                try:
                    result = future.result()
                    (succeeded if result else failed).append(vid_id)
                except Exception as exc:
                    print(f"\n[{vid_id}] error: {exc}")
                    failed.append(vid_id)

        print(f"\nDone: {len(succeeded)} processed, {len(failed)} failed.")

        if failed:
            Path(failed_ids_file).write_text(json.dumps(failed, indent=2))
            print(f"Failed IDs saved to {failed_ids_file}")
            print(f"\nTo retry:\n  python scripts/download_from_gcs.py "
                  f"--video_ids_file {failed_ids_file} --max_workers 1")

    else:
        # Prefix mode — no chunking, just raw download
        print(f"Listing blobs at gs://{bucket_name}/{prefix}/ ...")
        blob_names = _list_blobs(bucket_name, prefix)
        print(f"Found {len(blob_names)} blobs | output: {output_dir}")

        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_blob_to_relative, bucket_name, blob_name, prefix, output_dir): blob_name
                for blob_name in blob_names
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                blob_name = futures[future]
                try:
                    if not future.result():
                        failed.append(blob_name)
                except Exception as exc:
                    print(f"\n[{blob_name}] error: {exc}")
                    failed.append(blob_name)

        print(f"\nDone: {len(blob_names) - len(failed)} downloaded, {len(failed)} failed.")

    if not keep:
        print(f"Cleaning up {output_dir} ...")
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        print(f"Chunks kept at: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GCS videos, chunk them, clean up full videos")
    parser.add_argument("--bucket", default=GCS_BUCKET,
                        help="GCS bucket name (default: $GCS_BUCKET from .env)")
    parser.add_argument("--prefix", default=GCS_PREFIX,
                        help="GCS path prefix (default: $GCS_PREFIX from .env)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save chunks (default: system temp dir)")
    parser.add_argument("--chunks_file", default=str(_CHUNKS_FILE),
                        help=f"Path to video_chunks.json (default: {_CHUNKS_FILE})")
    parser.add_argument("--keep", action="store_true",
                        help="Keep chunk files after script exits")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--failed_ids_file", default="failed_downloads.json",
                        help="Where to save failed IDs (default: failed_downloads.json)")

    id_source = parser.add_mutually_exclusive_group()
    id_source.add_argument("--video_ids_file", default=None,
                           help="JSON file with a list of video IDs to process")
    id_source.add_argument("--splits", nargs="+", default=None,
                           help="Dataset splits to process (e.g. train val test)")
    parser.add_argument("--data_dir", default=str(_DATA_DIR),
                        help="Path to the siq2 data directory (used with --splits)")

    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set GCS_BUCKET in .env)")

    # Load chunk definitions
    with open(args.chunks_file) as f:
        chunks_map = json.load(f)
    print(f"Loaded chunk definitions for {len(chunks_map)} videos from {args.chunks_file}")

    # Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
        cleanup_tmp = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="gcs_chunks_"))
        cleanup_tmp = True

    # Resolve video IDs
    vid_ids: list[str] | None = None
    if args.video_ids_file:
        vid_ids = json.loads(Path(args.video_ids_file).read_text())
        print(f"Loaded {len(vid_ids)} IDs from {args.video_ids_file}")
    elif args.splits:
        vid_ids = _load_vid_ids_from_split(Path(args.data_dir), args.splits)
        print(f"Loaded {len(vid_ids)} IDs from splits: {args.splits}")

    try:
        run(
            bucket_name=args.bucket,
            prefix=args.prefix,
            output_dir=output_dir,
            vid_ids=vid_ids,
            chunks_map=chunks_map,
            max_workers=args.max_workers,
            keep=args.keep,
            failed_ids_file=args.failed_ids_file,
        )
    finally:
        if cleanup_tmp and not args.keep and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
