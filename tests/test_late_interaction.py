"""Unit tests for the LateInteractionAdapter (architecture B)."""

from __future__ import annotations

import torch

from social_memory.adapters.late_interaction import LateInteractionAdapter


def _make_model(d_video=16, d_text=16, d_out=8, num_chunk_tokens=4):
    return LateInteractionAdapter(
        d_video=d_video,
        d_text=d_text,
        d_out=d_out,
        d_hidden=32,
        num_chunk_tokens=num_chunk_tokens,
        num_heads=2,
        num_qformer_layers=1,
    )


def test_encode_chunk_returns_K_unit_tokens():
    m = _make_model(num_chunk_tokens=5)
    v_tokens = torch.randn(3, 10, 16)  # 3 chunks, 10 video tokens each
    t_tokens = torch.randn(3, 6, 16)
    c = m.encode_chunk(v_tokens, t_tokens)
    assert c.shape == (3, 5, 8)
    norms = c.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_encode_question_returns_unit_tokens_per_input_token():
    m = _make_model()
    q_tokens = torch.randn(2, 7, 16)
    q = m.encode_question(q_tokens)
    assert q.shape == (2, 7, 8)
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_handles_degenerate_pooled_inputs():
    """A single pooled vector should plug in as a length-1 sequence."""
    m = _make_model()
    v_pooled = torch.randn(2, 1, 16)
    t_pooled = torch.randn(2, 1, 16)
    q_pooled = torch.randn(2, 1, 16)
    out = m(q_pooled, v_pooled, t_pooled)
    assert out["q"].shape == (2, 1, 8)
    assert out["c"].shape[0] == 2  # K is preserved regardless of input length


def test_maxsim_aligned_pairs():
    """MaxSim with a known optimum: identical token in q and c → score ≈ 1."""
    # Hand-build L2-normalized tokens so the test is deterministic.
    q = torch.tensor([[[1.0, 0.0, 0.0]]])  # (1, 1, 3)
    c = torch.tensor([
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],   # matches q exactly
         [0.0, 0.0, 1.0]]
    ])  # (1, 3, 3)
    score = LateInteractionAdapter.maxsim(q, c)
    assert torch.allclose(score, torch.tensor([1.0]), atol=1e-6)


def test_maxsim_matrix_shape_for_infonce():
    """Cross-batch MaxSim returns a (B_q, B_c) matrix usable as InfoNCE logits."""
    q = torch.randn(4, 2, 8)
    c = torch.randn(5, 6, 8)
    q = q / q.norm(dim=-1, keepdim=True)
    c = c / c.norm(dim=-1, keepdim=True)
    mat = LateInteractionAdapter.maxsim_matrix(q, c)
    assert mat.shape == (4, 5)


def test_video_pad_mask_changes_chunk_tokens():
    """Padding tokens must be excluded from cross-attention; output should shift."""
    m = _make_model()
    torch.manual_seed(0)
    v_tokens = torch.randn(1, 4, 16)
    t_tokens = torch.randn(1, 2, 16)

    mask_all = torch.ones(1, 4, dtype=torch.bool)
    mask_half = torch.tensor([[True, True, False, False]])

    c_all = m.encode_chunk(v_tokens, t_tokens, video_mask=mask_all)
    c_half = m.encode_chunk(v_tokens, t_tokens, video_mask=mask_half)
    assert not torch.allclose(c_all, c_half, atol=1e-4)


def test_backward_through_infonce_loss():
    m = _make_model()
    q = m.encode_question(torch.randn(4, 3, 16))
    c = m.encode_chunk(torch.randn(4, 8, 16), torch.randn(4, 5, 16))
    logits = m.temperature() * LateInteractionAdapter.maxsim_matrix(q, c)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(4))
    loss.backward()

    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no grad reached: {missing}"
