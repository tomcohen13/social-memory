"""Unit tests for the trainable QCentric adapter."""

from __future__ import annotations

import math

import pytest
import torch

from social_memory.adapters.qcentric import (
    ChunkAdapter,
    QCentricAdapter,
    QuestionAdapter,
)


# ── shapes & invariants ───────────────────────────────────────────────────────

def test_question_adapter_shape_and_norm():
    adapter = QuestionAdapter(d_in=512, d_out=256)
    q = adapter(torch.randn(7, 512))
    assert q.shape == (7, 256)
    # L2 norm must be 1 along the embedding axis.
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_chunk_adapter_shape_and_norm():
    adapter = ChunkAdapter(d_video=1408, d_text=512, d_out=256)
    c = adapter(torch.randn(4, 1408), torch.randn(4, 512))
    assert c.shape == (4, 256)
    norms = c.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_qcentric_forward_returns_q_c_and_scale():
    m = QCentricAdapter(d_video=1408, d_text=512, d_out=256)
    out = m(
        q_emb=torch.randn(3, 512),
        video_emb=torch.randn(3, 1408),
        transcript_emb=torch.randn(3, 512),
    )
    assert set(out.keys()) == {"q", "c", "logit_scale"}
    assert out["q"].shape == (3, 256)
    assert out["c"].shape == (3, 256)
    assert out["logit_scale"].ndim == 0
    # exp(log(1/0.07)) ≈ 14.29 at init.
    assert math.isclose(out["logit_scale"].item(), 1.0 / 0.07, rel_tol=1e-4)


def test_temperature_is_clamped():
    m = QCentricAdapter()
    # Force log_scale absurdly high; temperature() must clamp at 100.
    with torch.no_grad():
        m.logit_scale.fill_(20.0)  # exp(20) ≈ 4.85e8
    assert m.temperature().item() == pytest.approx(100.0)


# ── multimodal fusion is actually multimodal ─────────────────────────────────

def test_chunk_output_depends_on_both_modalities():
    """If output is invariant to either modality, the 'fusion' isn't fusion."""
    adapter = ChunkAdapter(d_video=8, d_text=8, d_out=8)
    v1 = torch.randn(1, 8)
    t1 = torch.randn(1, 8)
    t2 = torch.randn(1, 8)
    v2 = torch.randn(1, 8)

    c_v1_t1 = adapter(v1, t1)
    c_v1_t2 = adapter(v1, t2)  # different transcript, same video
    c_v2_t1 = adapter(v2, t1)  # different video, same transcript

    assert not torch.allclose(c_v1_t1, c_v1_t2, atol=1e-4), "output ignores transcript"
    assert not torch.allclose(c_v1_t1, c_v2_t1, atol=1e-4), "output ignores video"


def test_has_transcript_mask_changes_output():
    """The has_transcript mask must actually suppress the text stream."""
    adapter = ChunkAdapter(d_video=8, d_text=8, d_out=8)
    v = torch.randn(2, 8)
    t = torch.randn(2, 8)
    mask_on = torch.tensor([1.0, 1.0])
    mask_off = torch.tensor([0.0, 0.0])

    c_on = adapter(v, t, has_transcript=mask_on)
    c_off = adapter(v, t, has_transcript=mask_off)

    assert not torch.allclose(c_on, c_off, atol=1e-4)


# ── gradient flow ─────────────────────────────────────────────────────────────

def test_backward_pass_populates_grads():
    m = QCentricAdapter(d_video=16, d_text=16, d_out=8, d_hidden=32)
    out = m(
        q_emb=torch.randn(5, 16),
        video_emb=torch.randn(5, 16),
        transcript_emb=torch.randn(5, 16),
    )
    # Toy InfoNCE-shaped loss to exercise both towers and logit_scale.
    logits = out["logit_scale"] * out["q"] @ out["c"].T
    labels = torch.arange(5)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # Every trainable parameter must receive gradient.
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no grad reached: {missing}"
    # logit_scale specifically must move with the loss.
    assert m.logit_scale.grad.abs().item() > 0
