"""Dataset + adapter wrapper that train on *precomputed* frozen-encoder features.

The on-the-fly pipeline (SIQ2LongDataset → XCLIPAdapter) re-encodes every
chunk through X-CLIP at every step, which dominates training time. Once
`scripts/precompute_features.py` (or the Modal `precompute_features`
function) has written

    features/<encoder>/chunks/<vid_name>.npz
    features/<encoder>/text/<split>.npz

we can train the adapter directly on those tensors with no backbone in the
hot path. This module provides:

- `FeatureCachedDataset` — emits per-video batches shaped like SIQ2LongDataset
  but with `video_emb`/`transcript_emb`/`q_emb` arrays instead of frames /
  strings.
- `XCLIPFeatureAdapter` — the same trainable heads as `XCLIPAdapter` from
  `social_memory.adapters.xclipadapter`, but consuming precomputed
  embeddings. Drop-in replacement: exposes `encode_chunks` and
  `encode_questions` so `train.compute_batch_loss_features` can use it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from social_memory.constants import (
    PATH_TO_AUGMENTED_DATA,
    SIQDatasetColumns,
)


class FeatureCachedDataset(Dataset):
    """One row = all questions for one video, with precomputed embeddings.

    Args:
        split: name of the contrastive jsonl under PATH_TO_AUGMENTED_DATA
            (e.g. "train" → datasets/siq2long/train_contrastive.jsonl).
        features_root: directory containing `chunks/` and `text/` subdirs
            (e.g. /features/xclip on Modal, or features/xclip locally).
        text_split: name used for the text npz file. Defaults to `split`.
        max_chunks_per_video: drop videos whose precomputed chunk count
            exceeds this (matches SIQ2LongDataset behavior).
    """

    def __init__(
        self,
        split: str,
        features_root: str | Path,
        text_split: str | None = None,
        max_chunks_per_video: int = 24,
    ):
        self.split = split
        self.features_root = Path(features_root)
        self.chunk_dir = self.features_root / "chunks"
        text_path = self.features_root / "text" / f"{text_split or split}.npz"
        if not text_path.exists():
            raise FileNotFoundError(f"missing text features: {text_path}")
        with np.load(text_path, allow_pickle=True) as f:
            self.qids = np.asarray(f["qids"], dtype=object)
            self.q_emb = np.asarray(f["q_emb"], dtype=np.float32)
        self.qid_to_idx = {str(q): i for i, q in enumerate(self.qids)}
        self.dim = int(self.q_emb.shape[1])

        df = pd.read_json(
            PATH_TO_AUGMENTED_DATA / f"{split}_contrastive.jsonl",
            lines=True,
        ).reset_index(drop=True)

        def _ok(vid_name: str) -> bool:
            p = self.chunk_dir / f"{vid_name}.npz"
            if not p.exists():
                return False
            with np.load(p) as f:
                n = int(f["video_emb"].shape[0])
            return 0 < n <= max_chunks_per_video

        valid = df[SIQDatasetColumns.VIDEO_ID].apply(_ok)
        n_dropped = int((~valid).sum())
        if n_dropped:
            print(
                f"FeatureCachedDataset: dropping {n_dropped} rows with "
                f"missing/oversize chunk features"
            )
        df = df[valid].reset_index(drop=True)

        # Same group-by-video shape as SIQ2LongDataset(group_by_video=True).
        self.groups = (
            df.groupby(SIQDatasetColumns.VIDEO_ID)
            .agg(
                {
                    "qid": list,
                    "question": list,
                    "oracle_idx": "first",
                    "hard_negatives": "first",
                }
            )
            .reset_index()
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict | None:
        row = self.groups.iloc[idx]
        vid = row[SIQDatasetColumns.VIDEO_ID]
        try:
            with np.load(self.chunk_dir / f"{vid}.npz") as f:
                video_emb = np.asarray(f["video_emb"], dtype=np.float32)
                transcript_emb = np.asarray(f["transcript_emb"], dtype=np.float32)
                chunk_idx = np.asarray(f["chunk_idx"], dtype=np.int32)
        except Exception as e:
            print(f"FeatureCachedDataset: skip {vid}: {type(e).__name__}: {e}")
            return None

        chunks = {
            int(ci): {
                "chunk_idx": int(ci),
                "video_emb": video_emb[j],
                "transcript_emb": transcript_emb[j],
            }
            for j, ci in enumerate(chunk_idx)
        }

        # Look up question embeddings by qid; drop questions we lack features for.
        q_embs: list[np.ndarray] = []
        kept_qids: list[str] = []
        kept_questions: list[str] = []
        for qid, q in zip(row["qid"], row["question"]):
            j = self.qid_to_idx.get(str(qid))
            if j is None:
                continue
            q_embs.append(self.q_emb[j])
            kept_qids.append(str(qid))
            kept_questions.append(q)
        if not q_embs:
            return None

        return {
            "vid_name": vid,
            "qid": kept_qids,
            "question": kept_questions,
            "q_emb": np.stack(q_embs).astype(np.float32),
            "oracle_idx": int(row["oracle_idx"]),
            "chunks": chunks,
        }


def feature_collate_fn(batch: list[dict | None]) -> dict:
    """Same shape contract as train.collate_fn — fields become per-video lists."""
    out: dict = defaultdict(list)
    for item in batch:
        if item is None:
            continue
        for k, v in item.items():
            out[k].append(v)
    return dict(out)


class _MLPHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class XCLIPFeatureAdapter(nn.Module):
    """Trainable X-CLIP adapter that consumes precomputed embeddings.

    Mirrors `social_memory.adapters.xclipadapter.XCLIPAdapter` (mean-fuse
    video + transcript, separate MLPs for chunk and question towers,
    L2-normalized outputs), but takes precomputed `video_emb` /
    `transcript_emb` / `q_emb` as inputs instead of frames / strings.

    Inputs are expected to be already L2-normalized by the frozen encoder.
    """

    def __init__(
        self,
        feat_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.chunk_adapter = _MLPHead(feat_dim, hidden_dim, output_dim)
        self.question_adapter = _MLPHead(feat_dim, hidden_dim, output_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32, non_blocking=True)
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

    def encode_chunks(
        self,
        video_embs: list,
        transcript_embs: list,
    ) -> torch.Tensor:
        """video_embs, transcript_embs: lists of 1-D (D,) arrays/tensors."""
        v = torch.stack([self._to_tensor(x) for x in video_embs], dim=0)
        t = torch.stack([self._to_tensor(x) for x in transcript_embs], dim=0)
        fused = (v + t) / 2.0
        out = self.chunk_adapter(fused)
        return F.normalize(out, dim=-1)

    def encode_questions(self, q_embs) -> torch.Tensor:
        """q_embs: (N, D) array/tensor."""
        x = self._to_tensor(q_embs)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        out = self.question_adapter(x)
        return F.normalize(out, dim=-1)
