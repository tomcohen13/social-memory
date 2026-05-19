"""Precompute frozen-encoder features for chunks, questions, and answers.

The question-centric adapter trains on top of frozen InternVideo2 / X-CLIP.
Re-encoding the full corpus every epoch is wasteful (and, for InternVideo2,
costly — each call is a Modal RPC). This module runs the encoders once over
the dataset and dumps numpy archives the training loop can mmap.

Output layout::

    features/<encoder>/chunks/<vid_name>.npz
        video_emb       (C, D) float32 — InternVideo2 aligned video features
        transcript_emb  (C, D) float32 — text-tower features of the chunk VTT
        has_transcript  (C,)   bool    — False when VTT was missing/empty
        chunk_idx       (C,)   int32   — numeric stem of each chunk on GCS
    features/<encoder>/text/<split>.npz
        qids            (N,)         object  — string ids, aligned with q/a
        q_emb           (N, D)       float32 — question embeddings
        a_emb           (N, 4, D)    float32 — answer-candidate embeddings
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from social_memory.constants import (
    GCS_BUCKET,
    GCS_CHUNKS_PREFIX,
)
from social_memory.encoder import sample_frames
from social_memory.gcs import download_to_temp, list_blobs
from social_memory.utils import read_vtt_file


class FrozenEncoder(Protocol):
    """Minimal interface both XCLIPEncoder and RemoteInternVideoEncoder satisfy."""

    num_frames: int

    def __call__(self, frames: list[np.ndarray], transcript: str) -> dict: ...
    def encode_text(self, text: str) -> torch.Tensor: ...


_CHUNK_STEM = re.compile(r"^(\d+)$")


def list_chunks_for_video(vid_name: str) -> list[tuple[int, str, str | None]]:
    """Return `(chunk_idx, mp4_blob, vtt_blob)` for every chunk of `vid_name`.

    Sorted by `chunk_idx`. `vtt_blob` is None when no transcript companion exists.
    Non-numeric chunk stems are skipped.
    """
    mp4s: dict[int, str] = {}
    vtts: dict[int, str] = {}
    for blob in list_blobs(GCS_BUCKET, f"{GCS_CHUNKS_PREFIX}/{vid_name}/"):
        path = blob.name
        m = _CHUNK_STEM.match(Path(path).stem)
        if not m:
            continue
        idx = int(m.group(1))
        ext = Path(path).suffix
        if ext == ".mp4":
            mp4s[idx] = path
        elif ext == ".vtt":
            vtts[idx] = path
    return [(i, mp4s[i], vtts.get(i)) for i in sorted(mp4s)]


def encode_video_chunks(
    encoder: FrozenEncoder,
    vid_name: str,
    chunks: list[tuple[int, str, str | None]] | None = None,
) -> dict[str, np.ndarray]:
    """Encode every chunk of one video into video+transcript embedding arrays."""
    if chunks is None:
        chunks = list_chunks_for_video(vid_name)
    if not chunks:
        raise ValueError(f"no chunks found on GCS for {vid_name}")

    video_embs: list[np.ndarray] = []
    txt_embs: list[np.ndarray] = []
    has_t: list[bool] = []
    idxs: list[int] = []

    # No tqdm here: carriage-return progress bars come through Modal's log
    # capture as partial frames. The outer per-video summary line is enough.
    for chunk_idx, mp4_blob, vtt_blob in chunks:
        transcript = ""
        if vtt_blob is not None:
            with download_to_temp(GCS_BUCKET, vtt_blob) as vtt_path:
                if vtt_path is not None:
                    transcript = read_vtt_file(vtt_path).strip()

        with download_to_temp(GCS_BUCKET, mp4_blob) as mp4_path:
            if mp4_path is None:
                raise FileNotFoundError(f"missing GCS object {mp4_blob}")
            frames = sample_frames(mp4_path, num_frames=encoder.num_frames)

        # BERT tokenizes "" to just [CLS][SEP]; pass a space to be defensive
        # while still being able to flag has_transcript=False downstream.
        out = encoder(frames, transcript if transcript else " ")
        video_embs.append(_to_np1d(out["video_embeddings"]))
        txt_embs.append(_to_np1d(out["text_embeddings"]))
        has_t.append(bool(transcript))
        idxs.append(chunk_idx)

    return {
        "video_emb": np.stack(video_embs).astype(np.float32),
        "transcript_emb": np.stack(txt_embs).astype(np.float32),
        "has_transcript": np.asarray(has_t, dtype=bool),
        "chunk_idx": np.asarray(idxs, dtype=np.int32),
    }


def encode_text_for_split(
    encoder: FrozenEncoder,
    qa_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Encode the question and four answer candidates for every row in `qa_df`.

    The dataframe must contain columns ``qid``, ``q``, ``a0``..``a3``.
    """
    qids: list[str] = []
    q_emb_list: list[np.ndarray] = []
    a_emb_list: list[np.ndarray] = []

    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="text"):
        qids.append(str(row["qid"]))
        q_emb_list.append(_to_np1d(encoder.encode_text(row["q"])))
        a_emb_list.append(
            np.stack([_to_np1d(encoder.encode_text(row[f"a{i}"])) for i in range(4)])
        )

    return {
        "qids": np.asarray(qids, dtype=object),
        "q_emb": np.stack(q_emb_list).astype(np.float32),
        "a_emb": np.stack(a_emb_list).astype(np.float32),
    }


def save_chunk_features(out_dir: Path, vid_name: str, arrays: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{vid_name}.npz"
    np.savez(path, **arrays)
    return path


def chunk_features_exist(out_dir: Path, vid_name: str, expected_count: int) -> bool:
    """Treat the cache as valid only if the row count matches what's on GCS."""
    path = out_dir / f"{vid_name}.npz"
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as f:
        return int(f["video_emb"].shape[0]) == expected_count


def _to_np1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError(f"expected 1D embedding, got shape {x.shape}")
    return x


def precompute_features_run(
    encoder,
    out_root,
    split_dfs: dict,
    *,
    limit_videos: int | None = None,
    skip_text: bool = False,
    skip_chunks: bool = False,
    max_chunks_per_video: int = 24,
    commit_callback=None,
    commit_every: int = 25,
) -> None:
    """Driver: encode every video chunk + per-split QA text, write `.npz`s.

    Shared by the local CLI and the Modal `precompute_features` wrapper so
    both code paths execute the same logic. The Modal wrapper passes
    `commit_callback=features_vol.commit` so intermediate cache writes
    become durable mid-run.

    Args:
        encoder: an `XCLIPEncoder` (or any `FrozenEncoder`) already on GPU.
        out_root: dir to write `chunks/<vid>.npz` and `text/<split>.npz`.
        split_dfs: `{split_name: qa_dataframe}` — same shape that
            `load_qa_dataset(SIQ2LONG, split)` returns.
        commit_callback: invoked every `commit_every` chunked videos and
            once after text encoding. Modal volume commit hook.
    """
    from pathlib import Path

    out_root = Path(out_root)
    chunk_dir = out_root / "chunks"
    text_dir = out_root / "text"

    all_videos = sorted(
        {v for df in split_dfs.values() for v in df["vid_name"].unique().tolist()}
    )
    if limit_videos:
        all_videos = all_videos[:limit_videos]
    print(
        f"target: {len(all_videos)} videos across splits "
        f"{list(split_dfs)}"
    )

    if not skip_chunks:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for i, vid_name in enumerate(all_videos):
            try:
                chunks = list_chunks_for_video(vid_name)
            except Exception as e:
                print(f"  [{i+1}/{len(all_videos)}] [error] list {vid_name}: {e}")
                continue
            if not chunks:
                print(f"  [{i+1}/{len(all_videos)}] [warn] no chunks for {vid_name}")
                continue
            if len(chunks) > max_chunks_per_video:
                print(
                    f"  [{i+1}/{len(all_videos)}] [skip] {vid_name}: "
                    f"{len(chunks)} chunks > max {max_chunks_per_video}"
                )
                continue
            if chunk_features_exist(chunk_dir, vid_name, len(chunks)):
                print(f"  [{i+1}/{len(all_videos)}] [skip] {vid_name} cached")
                continue
            try:
                arrays = encode_video_chunks(encoder, vid_name, chunks=chunks)
            except Exception as e:
                print(f"  [{i+1}/{len(all_videos)}] [error] encode {vid_name}: {e}")
                continue
            path = save_chunk_features(chunk_dir, vid_name, arrays)
            print(
                f"  [{i+1}/{len(all_videos)}] [done] {vid_name} → {path.name} "
                f"({arrays['video_emb'].shape[0]} chunks, dim={arrays['video_emb'].shape[1]})"
            )
            if commit_callback is not None and (i + 1) % commit_every == 0:
                commit_callback()

    if not skip_text:
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

    if commit_callback is not None:
        commit_callback()
    print("precompute complete")
