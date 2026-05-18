"""Cross-encoder reranker.

Question and chunk tokens are concatenated into one sequence and run through
a small Transformer encoder with self-attention across all positions. A
learnable [CLS] token pools the joint representation; a linear head produces
a relevance scalar. Modality/segment type embeddings tell the transformer
which token came from which stream.

This module is NOT indexable — you must re-run it for every (question,
chunk) pair. Use it as Stage-2 over the top-K returned by an indexable
retriever (QCentricAdapter or LateInteractionAdapter). The ceiling is much
higher than either retriever alone because the network can co-attend over
question tokens and chunk tokens at every layer — that's the only place
"the face shows X *while* the speaker says Y" can be represented end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_TYPE_QUESTION = 0
_TYPE_VIDEO = 1
_TYPE_TRANSCRIPT = 2


class CrossEncoderReranker(nn.Module):
    """Joint-encode [CLS; q; v; t] tokens, score the [CLS] output."""

    def __init__(
        self,
        d_video: int,
        d_text: int,
        d_hidden: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj_q = nn.Linear(d_text, d_hidden)
        self.proj_v = nn.Linear(d_video, d_hidden)
        self.proj_t = nn.Linear(d_text, d_hidden)
        self.type_emb = nn.Embedding(3, d_hidden)
        self.cls = nn.Parameter(torch.randn(1, 1, d_hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=num_heads,
            dim_feedforward=4 * d_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_hidden)
        self.head = nn.Linear(d_hidden, 1)

    def forward(
        self,
        q_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        transcript_tokens: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        transcript_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one relevance score per (q, chunk) row in the batch.

        Token shapes are (B, T, D). Masks are (B, T), 1 for keep, 0 for pad.
        """
        B = q_tokens.shape[0]

        q = self.proj_q(q_tokens) + self.type_emb.weight[_TYPE_QUESTION]
        v = self.proj_v(video_tokens) + self.type_emb.weight[_TYPE_VIDEO]
        t = self.proj_t(transcript_tokens) + self.type_emb.weight[_TYPE_TRANSCRIPT]
        cls = self.cls.expand(B, -1, -1)

        x = torch.cat([cls, q, v, t], dim=1)
        pad_mask = _build_pad_mask(B, q, v, t, q_mask, video_mask, transcript_mask)

        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def _build_pad_mask(
    batch: int,
    q: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    q_mask: torch.Tensor | None,
    video_mask: torch.Tensor | None,
    transcript_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if q_mask is None and video_mask is None and transcript_mask is None:
        return None
    device = q.device
    cls_pad = torch.zeros(batch, 1, dtype=torch.bool, device=device)
    q_pad = _pad_or_zero(q_mask, batch, q.shape[1], device)
    v_pad = _pad_or_zero(video_mask, batch, v.shape[1], device)
    t_pad = _pad_or_zero(transcript_mask, batch, t.shape[1], device)
    return torch.cat([cls_pad, q_pad, v_pad, t_pad], dim=1)


def _pad_or_zero(
    mask: torch.Tensor | None, batch: int, length: int, device: torch.device
) -> torch.Tensor:
    if mask is None:
        return torch.zeros(batch, length, dtype=torch.bool, device=device)
    return ~mask.bool()
