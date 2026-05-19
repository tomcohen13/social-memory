import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from social_memory.encoders.xclip import XCLIPEncoder


class XCLIPAdapter(nn.Module):
    """
    Frozen X-CLIP backbone + small trainable MLP adapters on both modalities.
    
    Chunks are encoded by fusing X-CLIP's video and text embeddings (mean),
    then passed through the chunk adapter. Questions are encoded by X-CLIP's
    text tower and passed through the question adapter. Output is L2-normalized
    so downstream cosine similarity is just a dot product.
    """
    
    def __init__(self, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.backbone = XCLIPEncoder()  # already freezes internally
        feat_dim = self.backbone.model.config.projection_dim  # 512 for base
        
        self.chunk_adapter = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.question_adapter = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_chunks(
        self,
        frames_list: list[list[np.ndarray]],
        transcripts: list[str],
    ) -> torch.Tensor:
        """Encode a list of chunks. Returns (n_chunks, output_dim).

        Two backbone passes for all chunks instead of 2*N — one batched video call
        and one batched text call.
        """
        video_embs = self.backbone.encode_video(frames_list)    # (n_chunks, D) — no grad
        text_embs = self.backbone.encode_text(transcripts)      # (n_chunks, D) — no grad
        x = (video_embs + text_embs) / 2
        x = self.chunk_adapter(x)                               # gradients flow here
        return F.normalize(x, dim=-1)

    def encode_questions(self, questions: list[str]) -> torch.Tensor:
        """Encode a list of question strings. Returns (n_questions, output_dim)."""
        x = self.backbone.encode_text(questions)    # single batched call — no grad
        x = self.question_adapter(x)                # gradients flow here
        return F.normalize(x, dim=-1)