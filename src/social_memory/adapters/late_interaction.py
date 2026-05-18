"""Late-interaction multi-vector adapter (ColBERT-style for video QA).

Each chunk is encoded into a **set of K tokens** via a small Q-Former that
cross-attends K learnable queries over the concatenated frozen video and
transcript token sequences. The question is projected token-wise into the
same space. Scoring is MaxSim — for each question token, take the max dot
product over the chunk's K tokens, then average across question tokens.

Why this exists (vs. the single-vector mid-fusion adapter):

- A single vector per chunk forces the model to summarize the chunk along
  one axis. Multiple "social aspects" (affect, speech, gesture, scene) get
  averaged into one direction.
- K chunk tokens let different tokens specialize. A question about emotion
  routes to the affect token; a question about "what did X say" routes to
  the speech token. Cross-modal fusion still happens, but inside each token.
- Late interaction (MaxSim) is more expressive than dot product but still
  indexable per chunk token, so first-stage retrieval stays cheap.

Inputs are token sequences. If you only have a pooled vector, pass it as
shape (B, 1, D) — the Q-Former handles arbitrary key length.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _QFormerBlock(nn.Module):
    """Pre-norm cross-attention block: learnable queries attend over context."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attended, _ = self.cross_attn(
            self.norm_q(queries),
            self.norm_ctx(context),
            self.norm_ctx(context),
            key_padding_mask=context_pad_mask,
            need_weights=False,
        )
        queries = queries + attended
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class _QFormer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_queries: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [_QFormerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        context: torch.Tensor,
        context_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = context.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, context, context_pad_mask)
        return q


class LateInteractionAdapter(nn.Module):
    """Question side: token-wise projection. Chunk side: Q-Former → K tokens."""

    def __init__(
        self,
        d_video: int,
        d_text: int,
        d_out: int = 128,
        d_hidden: int = 512,
        num_chunk_tokens: int = 8,
        num_heads: int = 4,
        num_qformer_layers: int = 2,
        dropout: float = 0.0,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.proj_v = nn.Linear(d_video, d_hidden)
        self.proj_t = nn.Linear(d_text, d_hidden)
        # Modality embeddings (0 = video, 1 = transcript) so the Q-Former can
        # tell which stream each context token came from.
        self.modality_emb = nn.Embedding(2, d_hidden)

        self.qformer = _QFormer(
            d_model=d_hidden,
            num_queries=num_chunk_tokens,
            num_heads=num_heads,
            num_layers=num_qformer_layers,
            dropout=dropout,
        )
        self.chunk_out = nn.Sequential(
            nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_out)
        )

        self.q_adapter = nn.Sequential(
            nn.Linear(d_text, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_question(self, q_tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_adapter(q_tokens), dim=-1)

    def encode_chunk(
        self,
        video_tokens: torch.Tensor,
        transcript_tokens: torch.Tensor,
        video_mask: torch.Tensor | None = None,
        transcript_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v = self.proj_v(video_tokens) + self.modality_emb.weight[0]
        t = self.proj_t(transcript_tokens) + self.modality_emb.weight[1]
        ctx = torch.cat([v, t], dim=1)

        # nn.MultiheadAttention treats True as "padding, ignore".
        ctx_pad = _build_pad_mask(
            (v.shape[0], v.shape[1]),
            (t.shape[0], t.shape[1]),
            video_mask,
            transcript_mask,
            device=v.device,
        )

        c_tokens = self.qformer(ctx, context_pad_mask=ctx_pad)
        return F.normalize(self.chunk_out(c_tokens), dim=-1)

    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    @staticmethod
    def maxsim(
        q: torch.Tensor,
        c: torch.Tensor,
        q_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aligned MaxSim: score(q_i, c_i) for batched pairs.

        q: (B, M, d), c: (B, K, d) — both L2-normalized.
        Returns: (B,) scores.
        """
        sims = torch.einsum("bmd,bkd->bmk", q, c)
        per_q = sims.max(dim=-1).values
        if q_mask is None:
            return per_q.mean(dim=-1)
        mask = q_mask.to(per_q.dtype)
        return (per_q * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    @staticmethod
    def maxsim_matrix(
        q: torch.Tensor,
        c: torch.Tensor,
        q_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-batch MaxSim for InfoNCE: returns (B_q, B_c) similarity matrix.

        q: (B_q, M, d), c: (B_c, K, d) — both L2-normalized.
        """
        sims = torch.einsum("imd,jkd->ijmk", q, c)
        per_q = sims.max(dim=-1).values
        if q_mask is None:
            return per_q.mean(dim=-1)
        mask = q_mask.to(per_q.dtype).unsqueeze(1)
        denom = mask.sum(dim=-1).clamp(min=1)
        return (per_q * mask).sum(dim=-1) / denom

    def forward(
        self,
        q_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        transcript_tokens: torch.Tensor,
        video_mask: torch.Tensor | None = None,
        transcript_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "q": self.encode_question(q_tokens),
            "c": self.encode_chunk(
                video_tokens, transcript_tokens, video_mask, transcript_mask
            ),
            "logit_scale": self.temperature(),
        }


def _build_pad_mask(
    v_shape: tuple[int, int],
    t_shape: tuple[int, int],
    video_mask: torch.Tensor | None,
    transcript_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Construct a key_padding_mask over [v; t]. None when nothing is masked."""
    if video_mask is None and transcript_mask is None:
        return None
    Bv, Tv = v_shape
    Bt, Tt = t_shape
    v_pad = (
        ~video_mask.bool()
        if video_mask is not None
        else torch.zeros(Bv, Tv, dtype=torch.bool, device=device)
    )
    t_pad = (
        ~transcript_mask.bool()
        if transcript_mask is not None
        else torch.zeros(Bt, Tt, dtype=torch.bool, device=device)
    )
    return torch.cat([v_pad, t_pad], dim=1)
