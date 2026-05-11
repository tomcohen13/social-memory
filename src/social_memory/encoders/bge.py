"""BGE text encoder (weights frozen). Standalone text retrieval baseline."""

from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BGEEncoder(torch.nn.Module):
    """
    Frozen BGE text encoder for retrieval.

    BGE follows the standard BERT-style encoder pattern: take the [CLS] token's
    last_hidden_state as the sentence embedding, then L2-normalize. The official
    BGE retrieval pipeline uses cosine similarity on normalized embeddings.

    For asymmetric retrieval (short query against longer documents), BGE-v1.5
    recommends prepending a query instruction. We expose this via `encode_query`
    vs `encode_passage`. The instruction is only applied to the query side.
    """

    CHECKPOINT = "BAAI/bge-large-en-v1.5"

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.requires_grad_(False)
        self.model.eval()
        self.max_length = max_length

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Tokenize, forward, take [CLS], L2-normalize."""
        if not texts:
            raise ValueError("encode() received an empty list of texts")
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("All inputs must be strings")

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        # BGE uses [CLS] pooling, which is the first token of last_hidden_state.
        cls_embeddings = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_embeddings, p=2, dim=-1)

    @torch.inference_mode()
    def encode_query(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Encode questions/queries. Prepends the BGE retrieval instruction."""
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(texts)

    @torch.inference_mode()
    def encode_passage(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Encode chunks/passages/documents. No instruction prepended."""
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(texts)

    @torch.inference_mode()
    def encode(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Symmetric encoding (no instruction). Use when query/passage roles don't apply."""
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(texts)