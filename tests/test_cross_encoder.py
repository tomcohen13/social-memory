"""Unit tests for the CrossEncoderReranker (architecture C)."""

from __future__ import annotations

import torch

from social_memory.adapters.cross_encoder import CrossEncoderReranker


def _make_model(d_video=16, d_text=16, d_hidden=32):
    return CrossEncoderReranker(
        d_video=d_video,
        d_text=d_text,
        d_hidden=d_hidden,
        num_heads=2,
        num_layers=2,
    )


def test_returns_one_score_per_pair():
    m = _make_model()
    score = m(
        q_tokens=torch.randn(5, 7, 16),
        video_tokens=torch.randn(5, 10, 16),
        transcript_tokens=torch.randn(5, 6, 16),
    )
    assert score.shape == (5,)


def test_score_depends_on_all_three_streams():
    """If swapping any stream doesn't move the score, attention isn't crossing."""
    torch.manual_seed(0)
    m = _make_model().eval()
    q = torch.randn(1, 4, 16)
    v = torch.randn(1, 6, 16)
    t = torch.randn(1, 3, 16)
    base = m(q, v, t)
    s_q = m(torch.randn(1, 4, 16), v, t)
    s_v = m(q, torch.randn(1, 6, 16), t)
    s_t = m(q, v, torch.randn(1, 3, 16))
    assert not torch.isclose(base, s_q, atol=1e-4)
    assert not torch.isclose(base, s_v, atol=1e-4)
    assert not torch.isclose(base, s_t, atol=1e-4)


def test_pad_mask_changes_score():
    m = _make_model().eval()
    torch.manual_seed(0)
    q = torch.randn(1, 4, 16)
    v = torch.randn(1, 6, 16)
    t = torch.randn(1, 3, 16)
    mask_all = torch.ones(1, 6, dtype=torch.bool)
    mask_half = torch.tensor([[True, True, True, False, False, False]])
    s_all = m(q, v, t, video_mask=mask_all)
    s_half = m(q, v, t, video_mask=mask_half)
    assert not torch.isclose(s_all, s_half, atol=1e-4)


def test_backward_through_pairwise_loss():
    """Simulate a hinge / margin-style rerank loss; check gradients flow."""
    m = _make_model()
    q = torch.randn(3, 4, 16)
    pos_v = torch.randn(3, 6, 16)
    pos_t = torch.randn(3, 3, 16)
    neg_v = torch.randn(3, 6, 16)
    neg_t = torch.randn(3, 3, 16)

    pos = m(q, pos_v, pos_t)
    neg = m(q, neg_v, neg_t)
    loss = torch.relu(1.0 - (pos - neg)).mean()
    loss.backward()

    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no grad reached: {missing}"
