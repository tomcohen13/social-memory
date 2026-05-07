# the two encoder (text & video) xclip and then videointern2

from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from transformers import XCLIPModel, XCLIPProcessor

from social_memory.utils import read_vtt_file

CHECKPOINT = "microsoft/xclip-base-patch16-16-frames"
DATA_ROOT = Path("datasets/socialiq2/siq2")


class XCLIPEncoder(torch.nn.Module):
    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__()
        self.processor = XCLIPProcessor.from_pretrained(checkpoint, use_fast=False)
        self.model = XCLIPModel.from_pretrained(checkpoint)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.num_frames = self.model.config.vision_config.num_frames

    def forward(
        self, frames: list[np.ndarray], transcript: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs = self.processor.tokenizer(
            [transcript], return_tensors="pt", padding=True, truncation=True,
        )
        video_inputs = self.processor.image_processor([frames], return_tensors="pt")
        text_out = self.model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        video_out = self.model.get_video_features(pixel_values=video_inputs["pixel_values"])
        text_emb = F.normalize(text_out.pooler_output, dim=-1)
        video_emb = F.normalize(video_out.pooler_output, dim=-1)
        return text_emb, video_emb


def sample_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """Decode `num_frames` evenly-spaced RGB frames from an mp4 as HWC uint8 arrays."""
    container = av.open(str(video_path))
    total = container.streams.video[0].frames
    indices = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    frames: list[np.ndarray] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break
    container.close()
    return frames


encoder = XCLIPEncoder()

# go into the datasets socialiq2 - siq2 videos as demo videos
video_id = "0GQ8pgQJShg"
frames = sample_frames(DATA_ROOT / "video" / f"{video_id}.mp4", encoder.num_frames)
transcript = read_vtt_file(DATA_ROOT / "transcript" / f"{video_id}.vtt")

with torch.no_grad():
    text_emb, video_emb = encoder(frames, transcript)

fused_concat = torch.cat([text_emb, video_emb], dim=-1)   # (1, 1024)
fused_mean = (text_emb + video_emb) / 2                   # (1, 512)
similarity = (text_emb * video_emb).sum(-1)               # (1,)

print(f"video_id       : {video_id}")
print(f"transcript[:80]: {transcript[:80]!r}")
print(f"text_emb       : {tuple(text_emb.shape)}")
print(f"video_emb      : {tuple(video_emb.shape)}")
print(f"fused_concat   : {tuple(fused_concat.shape)}")
print(f"fused_mean     : {tuple(fused_mean.shape)}")
print(f"cos similarity : {similarity.item():.4f}")