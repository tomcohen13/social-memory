"""X-CLIP text + video encoders (weights frozen). Frame sampling via PyAV."""


import av
import numpy as np
import torch
import torch.nn.functional as F
from transformers import XCLIPModel, XCLIPProcessor
from collections import Counter
from pathlib import Path
from typing import TypedDict


class XCLIPEncoderOutput(TypedDict):
    text_embeddings: torch.Tensor
    video_embeddings: torch.Tensor
    fused_embeddings: torch.Tensor


class XCLIPEncoder(torch.nn.Module):
    
    CHECKPOINT = "microsoft/xclip-base-patch16-16-frames"
    
    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__()
        # Slow tokenizer avoids subtle mismatches vs the fast Rust tokenizer on some inputs.
        self.processor = XCLIPProcessor.from_pretrained(checkpoint, use_fast=False)
        self.model = XCLIPModel.from_pretrained(checkpoint)
        self.model.requires_grad_(False)
        self.num_frames = self.model.config.vision_config.num_frames

    @torch.inference_mode()
    def forward(self, frames: list[np.ndarray], transcript: str) -> XCLIPEncoderOutput:
        device = self.model.device
        text_inputs = self.processor.tokenizer(
            [transcript],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        video_inputs = self.processor.image_processor([frames], return_tensors="pt").to(device)

        text_out = self.model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_emb = F.normalize(text_out.pooler_output, dim=-1)
        
        video_out = self.model.get_video_features(pixel_values=video_inputs["pixel_values"])
        video_emb = F.normalize(video_out.pooler_output, dim=-1)
        
        fused_mean = (text_emb + video_emb) / 2

        return {
            "text_embeddings": text_emb,
            "video_embeddings": video_emb,
            "fused_embeddings": fused_mean,
        }
    
    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of strings. Returns (n_texts, embed_size)."""
        device = self.model.device
        text_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_out = self.model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        return F.normalize(text_out.pooler_output, dim=-1)

    @torch.inference_mode()
    def encode_video(self, frames: list[list[np.ndarray]]) -> torch.Tensor:
        """Encode a batch of video chunks. Returns (n_chunks, embed_size)."""
        device = self.model.device
        video_inputs = self.processor.image_processor(frames, return_tensors="pt").to(device)
        video_out = self.model.get_video_features(pixel_values=video_inputs["pixel_values"])
        return F.normalize(video_out.pooler_output, dim=-1)


def _decoded_frame_count(video_path: Path) -> int:
    with av.open(str(video_path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _stream_total_frames(stream: av.video.stream.VideoStream, video_path: Path) -> int:
    if stream.frames and stream.frames > 0:
        return int(stream.frames)
    duration_s: float | None = None
    if stream.duration is not None and stream.time_base is not None:
        duration_s = float(stream.duration * stream.time_base)
    if duration_s is not None and stream.average_rate is not None:
        est = int(duration_s * float(stream.average_rate))
        if est > 0:
            return est
    return _decoded_frame_count(video_path)


def sample_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """
    Decode `num_frames` evenly-spaced RGB frames from an mp4 as HWC uint8 arrays.

    Uses container metadata when reliable; otherwise counts by decoding once.
    If the stream ends before enough frames are collected (bad metadata), pads
    by repeating the last decoded frame so the list length is always `num_frames`.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total = _stream_total_frames(stream, video_path)

    if total <= 0:
        raise ValueError(f"No decodable video frames in {video_path}")

    targets = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    want = Counter(targets)
    frames: list[np.ndarray] = []

    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            k = want.get(i, 0)
            if k:
                arr = frame.to_ndarray(format="rgb24")
                for _ in range(k):
                    frames.append(arr)
                    if len(frames) == num_frames:
                        return frames

    if len(frames) < num_frames:
        if not frames:
            raise ValueError(f"No frames decoded from {video_path} (metadata total={total})")
        pad = frames[-1]
        frames.extend([pad] * (num_frames - len(frames)))
    return frames
