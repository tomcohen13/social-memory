"""CLI driver for `social_memory.precompute`.

Examples
--------
Smoke test on the demo split with InternVideo2 (limits to first 2 videos)::

    uv run python scripts/precompute_features.py \\
        --encoder internvideo2 --splits demo --limit-videos 2

Full pass for adapter training (chunks + text for train/val)::

    uv run python scripts/precompute_features.py \\
        --encoder internvideo2 --splits train val

Outputs land in ``features/<encoder>/{chunks,text}/`` and are resumable —
already-encoded videos and splits are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
# `social_memory.encoder.RemoteInternVideoEncoder` does
# `from scripts.modal_internvideo import ...`, so the repo root must be on
# sys.path for that import to resolve.
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from social_memory.constants import Datasets
from social_memory.precompute import (
    chunk_features_exist,
    encode_text_for_split,
    encode_video_chunks,
    list_chunks_for_video,
    save_chunk_features,
)
from social_memory.utils import load_qa_dataset


def build_encoder(name: str):
    if name == "internvideo2":
        from social_memory.encoder import RemoteInternVideoEncoder

        return RemoteInternVideoEncoder()
    if name == "xclip":
        from social_memory.encoder import XCLIPEncoder
        from social_memory.utils import check_device

        enc = XCLIPEncoder()
        enc.to(check_device())
        return enc
    raise ValueError(f"unknown encoder {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", choices=["internvideo2", "xclip"], required=True)
    p.add_argument("--dataset", default=str(Datasets.SIQ2LONG))
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: features/<encoder>/",
    )
    p.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="encode at most N videos (smoke test)",
    )
    p.add_argument("--skip-text", action="store_true")
    p.add_argument("--skip-chunks", action="store_true")
    args = p.parse_args()

    out_dir = args.output_dir or _REPO_ROOT / "features" / args.encoder
    chunk_dir = out_dir / "chunks"
    text_dir = out_dir / "text"

    print(f"building encoder: {args.encoder}")
    encoder = build_encoder(args.encoder)

    split_dfs = {s: load_qa_dataset(args.dataset, s) for s in args.splits}
    all_videos = sorted(
        {v for df in split_dfs.values() for v in df["vid_name"].unique().tolist()}
    )
    if args.limit_videos:
        all_videos = all_videos[: args.limit_videos]
    print(f"target: {len(all_videos)} videos across splits {args.splits}")

    if not args.skip_chunks:
        for vid_name in all_videos:
            try:
                chunks = list_chunks_for_video(vid_name)
            except Exception as e:
                print(f"  [error] listing GCS chunks for {vid_name}: {e}")
                continue
            if not chunks:
                print(f"  [warn] no chunks on GCS for {vid_name}")
                continue
            if chunk_features_exist(chunk_dir, vid_name, len(chunks)):
                print(f"  [skip] {vid_name} ({len(chunks)} chunks already cached)")
                continue
            try:
                arrays = encode_video_chunks(encoder, vid_name, chunks=chunks)
            except Exception as e:
                # Don't abort the whole run on a single bad video.
                print(f"  [error] encoding {vid_name}: {e}")
                continue
            path = save_chunk_features(chunk_dir, vid_name, arrays)
            print(
                f"  [done] {vid_name} → {path.name} "
                f"({arrays['video_emb'].shape[0]} chunks, "
                f"dim={arrays['video_emb'].shape[1]})"
            )

    if not args.skip_text:
        text_dir.mkdir(parents=True, exist_ok=True)
        for split, df in split_dfs.items():
            target = text_dir / f"{split}.npz"
            if target.exists():
                with np.load(target, allow_pickle=True) as f:
                    if len(f["qids"]) == len(df):
                        print(f"  [skip] text/{split}.npz")
                        continue
            arrays = encode_text_for_split(encoder, df)
            np.savez(target, **arrays)
            print(
                f"  [done] text/{split}.npz "
                f"({len(arrays['qids'])} rows, dim={arrays['q_emb'].shape[1]})"
            )


if __name__ == "__main__":
    main()
