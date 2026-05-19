"""Unit tests for `social_memory.precompute`.

Mocks GCS, the encoder, and frame sampling so tests run offline.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from social_memory import precompute as P


# ── _to_np1d ──────────────────────────────────────────────────────────────────

def test_to_np1d_from_torch_2d_singleton():
    t = torch.tensor([[1.0, 2.0, 3.0]])
    out = P._to_np1d(t)
    assert out.shape == (3,)
    assert out.dtype == np.float32 or out.dtype == np.float64


def test_to_np1d_from_numpy_1d():
    arr = np.array([1.0, 2.0, 3.0])
    out = P._to_np1d(arr)
    assert out.shape == (3,)


def test_to_np1d_rejects_higher_rank():
    with pytest.raises(ValueError):
        P._to_np1d(np.zeros((2, 3)))


# ── list_chunks_for_video ─────────────────────────────────────────────────────

def _fake_blob(name: str):
    return SimpleNamespace(name=name)


def test_list_chunks_pairs_mp4_and_vtt(monkeypatch):
    blobs = [
        _fake_blob("siq2/chunks/vidA/0.mp4"),
        _fake_blob("siq2/chunks/vidA/0.vtt"),
        _fake_blob("siq2/chunks/vidA/2.mp4"),
        _fake_blob("siq2/chunks/vidA/2.vtt"),
        _fake_blob("siq2/chunks/vidA/1.mp4"),
        # no vtt for chunk 1 — should still appear, with vtt=None
        _fake_blob("siq2/chunks/vidA/README.txt"),  # non-numeric stem, skipped
    ]
    monkeypatch.setattr(P, "list_blobs", lambda bucket, prefix: iter(blobs))

    result = P.list_chunks_for_video("vidA")

    assert [r[0] for r in result] == [0, 1, 2]
    assert result[0][1].endswith("0.mp4") and result[0][2].endswith("0.vtt")
    assert result[1][1].endswith("1.mp4") and result[1][2] is None
    assert result[2][1].endswith("2.mp4") and result[2][2].endswith("2.vtt")


# ── save / exists round trip ──────────────────────────────────────────────────

def test_save_and_exists_roundtrip(tmp_path: Path):
    arrays = {
        "video_emb": np.zeros((3, 8), dtype=np.float32),
        "transcript_emb": np.zeros((3, 8), dtype=np.float32),
        "has_transcript": np.array([True, False, True]),
        "chunk_idx": np.array([0, 1, 2], dtype=np.int32),
    }
    P.save_chunk_features(tmp_path, "vidA", arrays)
    assert P.chunk_features_exist(tmp_path, "vidA", expected_count=3)
    # Count mismatch must invalidate the cache.
    assert not P.chunk_features_exist(tmp_path, "vidA", expected_count=4)
    # Missing video must report False.
    assert not P.chunk_features_exist(tmp_path, "vidB", expected_count=1)


# ── encode_video_chunks (fully mocked) ────────────────────────────────────────

class _FakeEncoder:
    """Stand-in for XCLIPEncoder / RemoteInternVideoEncoder.

    Returns deterministic embeddings so the tests can assert ordering.
    """

    num_frames = 4
    D = 6

    def __call__(self, frames, transcript: str):
        # Encode the first pixel of the first frame so each chunk's vector is unique.
        seed = int(frames[0].flat[0])
        v = np.full(self.D, float(seed), dtype=np.float32)
        # Use transcript length so empty vs non-empty is distinguishable.
        t = np.full(self.D, float(len(transcript)), dtype=np.float32)
        return {
            "video_embeddings": torch.from_numpy(v[None, :]),
            "text_embeddings": torch.from_numpy(t[None, :]),
            "fused_embeddings": torch.from_numpy(((v + t) / 2)[None, :]),
        }

    def encode_text(self, text: str) -> torch.Tensor:
        return torch.full((1, self.D), float(len(text)))


@contextmanager
def _yield_path(path: Path):
    yield path


def test_encode_video_chunks_orders_and_flags_transcripts(monkeypatch, tmp_path: Path):
    # Two chunks: one with a transcript, one without.
    chunks = [
        (0, "vidA/0.mp4", "vidA/0.vtt"),
        (1, "vidA/1.mp4", None),
    ]
    # Distinct first pixel per chunk so we can assert ordering survives.
    fake_frames_by_blob = {
        "vidA/0.mp4": [np.full((2, 2, 3), 5, dtype=np.uint8)] * 4,
        "vidA/1.mp4": [np.full((2, 2, 3), 9, dtype=np.uint8)] * 4,
    }

    def fake_download_to_temp(bucket, name):
        # Caller uses the path; vtt is read via read_vtt_file, mp4 via sample_frames.
        # Returning the blob name itself as the path is fine because both are mocked.
        return _yield_path(Path(name))

    monkeypatch.setattr(P, "download_to_temp", fake_download_to_temp)
    monkeypatch.setattr(
        P,
        "read_vtt_file",
        lambda path: "hello world" if str(path).endswith("0.vtt") else "",
    )
    monkeypatch.setattr(
        P, "sample_frames", lambda path, num_frames: fake_frames_by_blob[str(path)]
    )

    out = P.encode_video_chunks(_FakeEncoder(), "vidA", chunks=chunks)

    assert out["video_emb"].shape == (2, _FakeEncoder.D)
    assert out["chunk_idx"].tolist() == [0, 1]
    # First chunk had pixel 5, second had pixel 9.
    assert out["video_emb"][0, 0] == 5.0
    assert out["video_emb"][1, 0] == 9.0
    # Transcript flag reflects whether the VTT had content.
    assert out["has_transcript"].tolist() == [True, False]
    # transcript_emb[0] was encoded from "hello world" (11 chars); [1] from " " (1 char).
    assert out["transcript_emb"][0, 0] == 11.0
    assert out["transcript_emb"][1, 0] == 1.0


def test_encode_video_chunks_raises_on_empty(monkeypatch):
    monkeypatch.setattr(P, "list_blobs", lambda bucket, prefix: iter([]))
    with pytest.raises(ValueError, match="no chunks"):
        P.encode_video_chunks(_FakeEncoder(), "missing")


# ── encode_text_for_split ─────────────────────────────────────────────────────

def test_encode_text_for_split_shapes_and_alignment():
    enc = _FakeEncoder()
    df = pd.DataFrame(
        [
            {"qid": "q0", "q": "aa", "a0": "x", "a1": "yy", "a2": "zzz", "a3": "wwww"},
            {"qid": "q1", "q": "bbb", "a0": "p", "a1": "qq", "a2": "rrr", "a3": "ssss"},
        ]
    )

    out = P.encode_text_for_split(enc, df)

    assert out["qids"].tolist() == ["q0", "q1"]
    assert out["q_emb"].shape == (2, enc.D)
    assert out["a_emb"].shape == (2, 4, enc.D)
    # Question embeddings encode string length under the fake encoder.
    assert out["q_emb"][0, 0] == 2.0  # len("aa")
    assert out["q_emb"][1, 0] == 3.0  # len("bbb")
    # Answer matrix carries lengths 1,2,3,4 / 1,2,3,4.
    assert out["a_emb"][0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out["a_emb"][1, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
