"""X-CLIP text + video encoders (weights frozen). Frame sampling via PyAV."""

from pathlib import Path
from typing import TypedDict

import av
import numpy as np
import torch
import torch.nn.functional as F
from transformers import XCLIPModel, XCLIPProcessor

CHECKPOINT = "microsoft/xclip-base-patch16-16-frames"


class XCLIPEncoderOutput(TypedDict):
    text_embeddings: torch.Tensor
    video_embeddings: torch.Tensor
    fused_embeddings: torch.Tensor


class XCLIPEncoder(torch.nn.Module):
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
        video_out = self.model.get_video_features(pixel_values=video_inputs["pixel_values"])
        text_emb = F.normalize(text_out, dim=-1)
        video_emb = F.normalize(video_out, dim=-1)
        fused_mean = F.normalize((text_emb + video_emb) / 2, dim=-1)

        return {
            "text_embeddings": text_emb,
            "video_embeddings": video_emb,
            "fused_embeddings": fused_mean,
        }
    
    def encode_text(self, text: str) -> torch.Tensor:

        device = self.model.device
        text_inputs = self.processor.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_out = self.model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_emb = F.normalize(text_out, dim=-1)
        return text_emb


class RemoteInternVideoEncoder:
    """InternVideo2 Stage2-1B served on Modal. Mirrors XCLIPEncoder's interface.

    Requires scripts/modal_internvideo.py to be deployable (modal token + HF secret
    + a one-time `modal run scripts/modal_internvideo.py::download_weights`).
    """

    def __init__(self) -> None:
        from scripts.modal_internvideo import InternVideo2Stage2

        self._cls = InternVideo2Stage2()
        self.num_frames = int(self._cls.num_frames_required.remote())

    def __call__(self, frames: list[np.ndarray], transcript: str) -> XCLIPEncoderOutput:
        arr = np.stack(frames)
        out = self._cls.encode.remote(arr, transcript)
        return {
            "text_embeddings": torch.from_numpy(out["text_embeddings"]),
            "video_embeddings": torch.from_numpy(out["video_embeddings"]),
            "fused_embeddings": torch.from_numpy(out["fused_embeddings"]),
        }

    def encode_text(self, text: str) -> torch.Tensor:
        return torch.from_numpy(self._cls.encode_text.remote(text))


def _duration_in_pts(stream: av.video.stream.VideoStream, container: av.container.InputContainer) -> int:
    if stream.duration and stream.duration > 0:
        return int(stream.duration)
    if container.duration and stream.time_base:
        return int((container.duration / 1_000_000) / float(stream.time_base))
    return 0


def _sample_frames_sequential(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """Fallback when duration metadata is missing: single sequential pass."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = sum(1 for _ in container.decode(stream))

    if total <= 0:
        raise ValueError(f"No decodable video frames in {video_path}")

    targets = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in targets:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == num_frames:
                    break

    if len(frames) < num_frames:
        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def sample_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """
    Decode `num_frames` evenly-spaced RGB frames from an mp4 as HWC uint8 arrays.

    Seeks to each target PTS (lands on the nearest preceding keyframe) and decodes
    forward only until the target is reached — O(num_frames * keyframe_gap) work
    instead of decoding every frame in the file. Falls back to a single sequential
    pass when duration metadata is unavailable.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration_ts = _duration_in_pts(stream, container)

        if duration_ts <= 0:
            # No reliable duration — can't compute PTS targets without a frame count.
            return _sample_frames_sequential(video_path, num_frames)

        targets = np.linspace(0, duration_ts, num_frames, dtype=np.int64).tolist()
        frames: list[np.ndarray] = []
        last: np.ndarray | None = None

        for target in targets:
            container.seek(int(target), stream=stream, any_frame=False, backward=True)
            chosen: np.ndarray | None = None
            prev: av.VideoFrame | None = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if frame.pts >= target:
                    # Prefer the frame closest to the target (prev may be closer).
                    pick = prev if prev is not None and (target - prev.pts) < (frame.pts - target) else frame
                    chosen = pick.to_ndarray(format="rgb24")
                    break
                prev = frame
            if chosen is None and prev is not None:
                chosen = prev.to_ndarray(format="rgb24")
            if chosen is None:
                chosen = last
            if chosen is None:
                continue
            frames.append(chosen)
            last = chosen

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames
