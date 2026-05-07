# the two encoder (text & video) xclip and then videointern2

from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

from social_memory.utils import read_vtt_file

processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32", use_fast=False)
model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")

# freeze the architecture (go over the params of each of the parameters)
for p in model.parameters():
    p.requires_grad = False
model.eval()

NUM_FRAMES = model.config.vision_config.num_frames  # 8 for base-patch32
DATA_ROOT = Path("datasets/socialiq2/siq2")


# run a forward pass of the video-trancript chunk
def sample_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    """Decode `num_frames` evenly-spaced RGB frames from an mp4 as PIL Images."""
    container = av.open(str(video_path))
    total = container.streams.video[0].frames
    container.close()

    indices = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    container = av.open(str(video_path))
    frames: list[Image.Image] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
        if len(frames) == num_frames:
            break
    container.close()
    return frames


# go into the datasets socialiq2 - siq2 videos as demo videos
video_id = "0GQ8pgQJShg"
frames = sample_frames(DATA_ROOT / "video" / f"{video_id}.mp4", NUM_FRAMES)
transcript = read_vtt_file(DATA_ROOT / "transcript" / f"{video_id}.vtt")

# Split text and video calls: in transformers 5.2.0 the combined
# XCLIPProcessor(text=..., videos=...) path drops the video kwarg.
text_inputs = processor.tokenizer(
    [transcript], return_tensors="pt", padding=True, truncation=True,
)
video_inputs = processor.image_processor([frames], return_tensors="pt")

# encode individually, fuse representations
with torch.no_grad():
    text_out = model.get_text_features(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
    )
    video_out = model.get_video_features(pixel_values=video_inputs["pixel_values"])

# In transformers 5.2.0 both calls return BaseModelOutputWithPooling whose
# pooler_output is the final 512-d projected embedding (text_projection /
# visual_projection + MIT have already been applied).
text_emb = text_out.pooler_output
video_emb = video_out.pooler_output

text_emb = F.normalize(text_emb, dim=-1)
video_emb = F.normalize(video_emb, dim=-1)

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
