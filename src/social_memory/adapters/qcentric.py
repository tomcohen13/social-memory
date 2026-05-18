"""Trainable adapter that warps frozen InternVideo2 / X-CLIP features into a
question-answering space.

This module is *only* the trainable component. Backbones (vision tower, text
tower) are assumed frozen and run upstream — we consume their L2-normalized
output vectors directly. The module exposes two towers that map into a shared
d-dim space:

    QuestionAdapter:  q_emb (frozen text)              -> q  in R^d
    ChunkAdapter:     video_emb, transcript_emb        -> c  in R^d   (multimodal)

The chunk-side fusion is genuinely multimodal — not a `(v + t) / 2` shortcut.
Each modality is projected separately, a per-dimension gate weights the two
streams, and a Hadamard interaction `v ⊙ t` is concatenated into the MLP so
the network can score *co-occurrence*, not just average descriptions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    """2-layer MLP with LayerNorm-in-the-middle and GELU."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuestionAdapter(nn.Module):
    """Question tower: small MLP on top of a frozen text embedding."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.adapter = _MLP(d_in, d_hidden, d_out, dropout)

    def forward(self, q_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.adapter(q_emb), dim=-1)


class ChunkAdapter(nn.Module):
    """Chunk tower: multimodal fusion of frozen video + transcript embeddings.

    Fusion path::

        v' = LN(W_v · video_emb)
        t' = LN(W_t · transcript_emb)              # zeroed if has_transcript=False
        g  = sigmoid(W_g · [v'; t'])               # per-dim gate
        v_g, t_g = g ⊙ v', (1 - g) ⊙ t'
        m  = v_g ⊙ t_g                              # Hadamard interaction
        c  = MLP([v_g; t_g; m]) -> L2 normalize

    Why this instead of `(v + t) / 2`:
      - separate, LayerNormed projections calibrate each modality
      - the gate lets the network suppress the text path when transcript is
        missing or unreliable (silent chunk, ASR failure)
      - the Hadamard term lets downstream layers attend to co-occurrence
        ("the face shows surprise *while* the speaker says X"), which a mean
        cannot represent
    """

    def __init__(
        self,
        d_video: int,
        d_text: int,
        d_out: int,
        d_hidden: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj_v = nn.Sequential(nn.Linear(d_video, d_hidden), nn.LayerNorm(d_hidden))
        self.proj_t = nn.Sequential(nn.Linear(d_text, d_hidden), nn.LayerNorm(d_hidden))
        self.gate = nn.Linear(2 * d_hidden, d_hidden)
        self.fuse = _MLP(3 * d_hidden, d_hidden, d_out, dropout)

    def forward(
        self,
        video_emb: torch.Tensor,
        transcript_emb: torch.Tensor,
        has_transcript: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v = self.proj_v(video_emb)
        t = self.proj_t(transcript_emb)
        if has_transcript is not None:
            # Hard mask: kill the transcript stream for chunks without one so
            # the gate sees a clean zero rather than BERT's [CLS]-of-empty noise.
            t = t * has_transcript.to(t.dtype).unsqueeze(-1)
        g = torch.sigmoid(self.gate(torch.cat([v, t], dim=-1)))
        v_g = g * v
        t_g = (1.0 - g) * t
        m = v_g * t_g
        c = self.fuse(torch.cat([v_g, t_g, m], dim=-1))
        return F.normalize(c, dim=-1)


class QCentricAdapter(nn.Module):
    """Two-tower adapter with a learned temperature for InfoNCE training.

    Inputs are the frozen-encoder outputs:
        q_emb           (B, d_text)   normalized text-tower embedding of the question
        video_emb       (B, d_video)  normalized video-tower embedding of the chunk
        transcript_emb  (B, d_text)   normalized text-tower embedding of the chunk transcript
        has_transcript  (B,) bool/0-1 optional mask; if omitted, transcript stream stays live

    Outputs (from `forward`)::

        {
            "q":            (B, d_out)  L2-normalized question embedding
            "c":            (B, d_out)  L2-normalized multimodal chunk embedding
            "logit_scale":  scalar      exp(log_temp), clamped at 100
        }

    The training loop computes `scale * q @ c.T` and feeds it to InfoNCE.
    """

    def __init__(
        self,
        d_video: int = 512,
        d_text: int = 512,
        d_out: int = 256,
        d_hidden: int = 512,
        dropout: float = 0.0,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.q_adapter = QuestionAdapter(d_text, d_out, d_hidden, dropout)
        self.c_adapter = ChunkAdapter(d_video, d_text, d_out, d_hidden, dropout)
        # Log-parameterized temperature, same as CLIP. Keeps the gradient
        # well-scaled and guarantees the effective τ stays positive.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_question(self, q_emb: torch.Tensor) -> torch.Tensor:
        return self.q_adapter(q_emb)

    def encode_chunk(
        self,
        video_emb: torch.Tensor,
        transcript_emb: torch.Tensor,
        has_transcript: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.c_adapter(video_emb, transcript_emb, has_transcript)

    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def forward(
        self,
        q_emb: torch.Tensor,
        video_emb: torch.Tensor,
        transcript_emb: torch.Tensor,
        has_transcript: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "q": self.encode_question(q_emb),
            "c": self.encode_chunk(video_emb, transcript_emb, has_transcript),
            "logit_scale": self.temperature(),
        }
