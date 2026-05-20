import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_src := _REPO_ROOT / "src").is_dir():
    sys.path.insert(0, str(_src))

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch

from social_memory.adapters.xclipadapter import XCLIPAdapter


FEAT_DIM = 512
HIDDEN_DIM = 64
OUTPUT_DIM = 32


def _make_fake_backbone(feat_dim=FEAT_DIM):
    """XCLIPEncoder stub — returns normalized random tensors of the right shape."""
    backbone = MagicMock()
    backbone.model.config.projection_dim = feat_dim
    backbone.encode_video.side_effect = (
        lambda frames_list: F.normalize(torch.randn(len(frames_list), feat_dim), dim=-1)
    )
    backbone.encode_text.side_effect = (
        lambda texts: F.normalize(torch.randn(len(texts), feat_dim), dim=-1)
    )
    return backbone


def _make_frames(n_chunks, n_frames=4):
    """Tiny placeholder frames; backbone is mocked so actual content is irrelevant."""
    return [
        [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n_frames)]
        for _ in range(n_chunks)
    ]


@pytest.fixture
def adapter():
    """Real XCLIPAdapter with a mocked backbone (no model weights loaded)."""
    with patch("social_memory.adapters.xclipadapter.XCLIPEncoder") as MockEncoderClass:
        MockEncoderClass.return_value = _make_fake_backbone()
        model = XCLIPAdapter(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
    return model


class TestEncodeChunks:
    def test_output_shape(self, adapter):
        embs = adapter.encode_chunks(_make_frames(5), ["transcript"] * 5)
        assert embs.shape == (5, OUTPUT_DIM)

    def test_output_shape_single_chunk(self, adapter):
        embs = adapter.encode_chunks(_make_frames(1), ["transcript"])
        assert embs.shape == (1, OUTPUT_DIM)

    def test_output_is_l2_normalized(self, adapter):
        embs = adapter.encode_chunks(_make_frames(4), ["transcript"] * 4)
        norms = embs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_backbone_called_once_not_per_chunk(self, adapter):
        # Batching: one encode_video + one encode_text call regardless of n_chunks.
        adapter.encode_chunks(_make_frames(6), ["transcript"] * 6)
        assert adapter.backbone.encode_video.call_count == 1
        assert adapter.backbone.encode_text.call_count == 1

    def test_frames_and_transcripts_forwarded_together(self, adapter):
        frames = _make_frames(3)
        adapter.encode_chunks(frames, ["a", "b", "c"])
        call_args = adapter.backbone.encode_video.call_args[0][0]
        assert len(call_args) == 3


class TestEncodeQuestions:
    def test_output_shape(self, adapter):
        questions = ["What happened?", "Who is there?", "Where are they?"]
        embs = adapter.encode_questions(questions)
        assert embs.shape == (len(questions), OUTPUT_DIM)

    def test_output_shape_single_question(self, adapter):
        embs = adapter.encode_questions(["What?"])
        assert embs.shape == (1, OUTPUT_DIM)

    def test_output_is_l2_normalized(self, adapter):
        embs = adapter.encode_questions(["Why?", "How?", "When?"])
        norms = embs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_backbone_called_once_not_per_question(self, adapter):
        adapter.encode_questions(["a", "b", "c", "d"])
        assert adapter.backbone.encode_text.call_count == 1

    def test_all_questions_forwarded_together(self, adapter):
        questions = ["a", "b", "c"]
        adapter.encode_questions(questions)
        forwarded = adapter.backbone.encode_text.call_args[0][0]
        assert forwarded == questions


class TestSimilarity:
    def test_dot_product_in_valid_range(self, adapter):
        chunk_embs = adapter.encode_chunks(_make_frames(5), ["t"] * 5)
        question_embs = adapter.encode_questions(["q1", "q2"])
        sims = question_embs @ chunk_embs.T  # (2, 5)
        assert sims.shape == (2, 5)
        assert sims.min() >= -1.0 - 1e-5
        assert sims.max() <= 1.0 + 1e-5


class TestGradients:
    def test_adapter_params_receive_gradients(self, adapter):
        adapter.train()
        chunk_embs = adapter.encode_chunks(_make_frames(3), ["t"] * 3)
        question_embs = adapter.encode_questions(["What?"])
        loss = -(question_embs @ chunk_embs.T).mean()
        loss.backward()
        for name, param in adapter.named_parameters():
            assert param.grad is not None, f"{name} has no gradient after backward"
            assert param.grad.abs().sum() > 0, f"{name} gradient is all zeros"
