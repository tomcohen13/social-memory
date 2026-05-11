"""InternVideo2 Stage2-1B served on Modal.

One-time setup (creates the weights volume — ~3 min, ~3.5GB):
    modal run scripts/modal_internvideo.py::download_weights

Smoke test (samples frames locally, encodes remotely):
    modal run scripts/modal_internvideo.py --video-path assets/<file>.mp4

Deploy persistent HTTP endpoint:
    modal deploy scripts/modal_internvideo.py
After deploy, Modal prints a URL like
    https://<workspace>--internvideo2-stage2-1b-internvideo2stage2-encode-video.modal.run
Call it with a publicly-fetchable video URL:
    curl -X POST "$URL" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"video_url": "https://example.com/clip.mp4", "text": "two people talking"}'

Programmatic use from the pipeline: see RemoteInternVideoEncoder in
src/social_memory/encoder.py.

Prerequisites
-------------
1. `pip install modal` and run `modal token new`.
2. Accept the gated-model license at
   https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4
3. Create two Modal secrets:
   - "huggingface" with HF_TOKEN=<hf token>
   - "internvideo-api-key" with API_KEY=<random string> (e.g.
     `openssl rand -hex 24`). Distribute this key to authorized callers.
"""

from __future__ import annotations

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel

APP_NAME = "internvideo2-stage2-1b"
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MiB cap on downloaded videos
DOWNLOAD_TIMEOUT_S = 60


class EncodeRequest(BaseModel):
    video_url: str
    text: str


HF_MODEL_ID = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
HF_BERT_ID = "bert-large-uncased"
WEIGHTS_FILENAME = "InternVideo2-stage2_1b-224p-f4.pt"

WEIGHTS_DIR = "/weights"
REPO_DIR = "/opt/InternVideo"
MULTI_MODALITY_DIR = f"{REPO_DIR}/InternVideo2/multi_modality"
NUM_FRAMES = 4
INPUT_RES = 224


FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.40.0",
        "huggingface_hub>=0.24",
        "einops",
        "timm",
        "open_clip_torch",
        "ftfy",
        "regex",
        "decord",
        "opencv-python-headless",
        "numpy<2",
        "av",
        "easydict",
        "mmengine",
        "peft==0.11.1",
        "scipy",
        "ninja",
        "packaging",
        "wheel",
        "fastapi",
        "python-multipart",
        "google-cloud-storage",
    )
    .run_commands(f"pip install {FLASH_ATTN_WHEEL}")
    # flash-attn's prebuilt wheel omits the optional `dropout_layer_norm` CUDA
    # extension. InternVideo2's `internvl_clip_vision.py` unconditionally imports
    # `DropoutAddRMSNorm`, which transitively `import dropout_layer_norm`s. The
    # Stage2 path never calls those ops, so a do-nothing module satisfies the
    # import chain. If we ever switch on fused ops, build flash-attn's
    # csrc/layer_norm from source instead.
    .run_commands(
        "echo 'def __getattr__(name):' "
        "> /usr/local/lib/python3.10/site-packages/dropout_layer_norm.py",
        "echo '    raise RuntimeError(\"dropout_layer_norm stub: fused ops unavailable\")' "
        ">> /usr/local/lib/python3.10/site-packages/dropout_layer_norm.py",
    )
    .run_commands(f"git clone --depth 1 https://github.com/OpenGVLab/InternVideo.git {REPO_DIR}")
    # The repo's `models/__init__.py` and `models/backbones/internvideo2/__init__.py`
    # eagerly import CLIP-1B, CLIP-6B (LLaMA-based), and audiovisual variants we
    # don't use — those chains drag in `dropout_layer_norm` extensions, the
    # `..utils.distributed` relative import that fails outside the package context,
    # and the `peft+LLaMA` text stack. Empty both __init__ files; import the one
    # function we actually need (`pretrain_internvideo2_1b_patch14_224`) via its
    # full sub-module path, and drop demo/utils.py's unused `get_sim` import.
    .run_commands(
        f": > {MULTI_MODALITY_DIR}/models/__init__.py",
        f": > {MULTI_MODALITY_DIR}/models/backbones/internvideo2/__init__.py",
        f"sed -i 's|from models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224|from models.backbones.internvideo2.internvideo2 import pretrain_internvideo2_1b_patch14_224|' {MULTI_MODALITY_DIR}/demo/utils.py",
        f"sed -i 's|from models.criterions import get_sim|# get_sim removed: unused, and criterions.py has broken `..utils` imports|' {MULTI_MODALITY_DIR}/demo/utils.py",
        # The bundled BertTokenizer sets self.vocab after super().__init__(),
        # which in transformers>=4.34 triggers get_vocab() before vocab exists.
        # Stock transformers.BertTokenizer is drop-in for bert-large-uncased.
        f"sed -i 's|from models.backbones.bert.tokenization_bert import BertTokenizer|from transformers import BertTokenizer|' {MULTI_MODALITY_DIR}/demo/utils.py",
    )
    .env(
        {
            "PYTHONPATH": MULTI_MODALITY_DIR,
            "HF_HOME": f"{WEIGHTS_DIR}/hf_cache",
            "TRANSFORMERS_CACHE": f"{WEIGHTS_DIR}/hf_cache",
        }
    )
)

app = modal.App(APP_NAME, image=image)
weights_vol = modal.Volume.from_name("internvideo2-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
api_key_secret = modal.Secret.from_name(
    "internvideo-api-key", required_keys=["API_KEY"]
)
gcp_secret = modal.Secret.from_name(
    "gcp-credentials", required_keys=["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
)


@app.function(
    volumes={WEIGHTS_DIR: weights_vol},
    secrets=[hf_secret],
    timeout=1800,
)
def download_weights() -> None:
    import os
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    ckpt_path = os.path.join(WEIGHTS_DIR, WEIGHTS_FILENAME)
    if not os.path.exists(ckpt_path):
        print(f"Downloading {HF_MODEL_ID}/{WEIGHTS_FILENAME}...")
        hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename=WEIGHTS_FILENAME,
            local_dir=WEIGHTS_DIR,
        )
    else:
        print("InternVideo2 checkpoint already present, skipping.")

    bert_dir = os.path.join(WEIGHTS_DIR, "bert-large-uncased")
    if not os.path.exists(os.path.join(bert_dir, "config.json")):
        print(f"Downloading {HF_BERT_ID}...")
        snapshot_download(repo_id=HF_BERT_ID, local_dir=bert_dir)
    else:
        print("bert-large-uncased already present, skipping.")

    weights_vol.commit()
    print("Weights ready:", sorted(os.listdir(WEIGHTS_DIR)))


@app.cls(
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights_vol},
    secrets=[hf_secret, api_key_secret, gcp_secret],
    scaledown_window=600,
    timeout=1800,
)
class InternVideo2Stage2:
    @modal.enter()
    def load(self) -> None:
        import os
        import sys

        # google-cloud-storage's ADC needs a file path, but the Modal secret
        # holds the SA key as a JSON string. Materialize it once at startup.
        sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if sa_json:
            sa_path = "/tmp/gcp-sa.json"
            with open(sa_path, "w") as f:
                f.write(sa_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

        sys.path.insert(0, MULTI_MODALITY_DIR)

        import torch
        from utils.config import Config, eval_dict_leaf
        from demo.utils import setup_internvideo2

        cfg = Config.from_file(f"{MULTI_MODALITY_DIR}/demo/internvideo2_stage2_config.py")
        # Config.from_file does not resolve `${...}` template interpolations —
        # `eval_dict_leaf` does. Without it, text_encoder stays a literal string
        # and other fields (num_frames, max_txt_l) stay as strings too.
        cfg = eval_dict_leaf(cfg)
        cfg.pretrained_path = f"{WEIGHTS_DIR}/{WEIGHTS_FILENAME}"
        cfg.model.vision_encoder.pretrained = cfg.pretrained_path
        cfg.model.text_encoder.pretrained = f"{WEIGHTS_DIR}/bert-large-uncased"
        # bert config JSON path: make absolute so build_bert doesn't depend on cwd.
        cfg.model.text_encoder.config = f"{MULTI_MODALITY_DIR}/configs/config_bert_large.json"
        cfg.device = "cuda"
        # fp32 path: avoids flash-attn fused ops (DropoutAddRMSNorm / FusedMLP),
        # which are stubbed in this image. Re-enable once we build the
        # flash-attn layer_norm extension from source.
        cfg.use_half_precision = False
        cfg.use_bf16 = False
        cfg.model.vision_encoder.use_flash_attn = False
        cfg.model.vision_encoder.use_fused_rmsnorm = False
        cfg.model.vision_encoder.use_fused_mlp = False
        cfg.num_frames = NUM_FRAMES
        cfg.num_frames_test = NUM_FRAMES
        cfg.origin_num_frames = NUM_FRAMES

        model, tokenizer = setup_internvideo2(cfg)
        self.model = model
        self.tokenizer = tokenizer
        self.num_frames = NUM_FRAMES
        self.input_res = INPUT_RES
        self._torch = torch

    @modal.method()
    def num_frames_required(self) -> int:
        return self.num_frames

    @modal.method()
    def encode(self, frames, transcript: str) -> dict:
        """Encode an RGB frame stack + transcript into normalized joint embeddings.

        frames: numpy uint8 array of shape (T, H, W, 3) where T == num_frames.
        Returns dict of float32 numpy arrays: video, text, fused.
        """
        return self._encode(frames, transcript)

    @modal.method()
    def encode_text(self, text: str):
        import torch.nn.functional as F

        torch = self._torch
        with torch.inference_mode():
            emb = self.model.get_txt_feat(text)
            emb = F.normalize(emb, dim=-1)
        return emb.float().cpu().numpy()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def encode_video(
        self,
        payload: EncodeRequest,
        x_api_key: str | None = Header(default=None),
    ):
        """POST {"video_url": "...", "text": "..."} → JSON embeddings.

        Server fetches the video over HTTPS, samples NUM_FRAMES frames, and
        returns three unit-norm 512-d vectors as plain JSON lists.
        """
        import os

        if x_api_key != os.environ["API_KEY"]:
            raise HTTPException(status_code=401, detail="invalid or missing api key")

        try:
            video_bytes = self._download(payload.video_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        import tempfile

        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        try:
            frames = self._sample_frames(tmp_path, self.num_frames)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            os.unlink(tmp_path)

        result = self._encode(np.stack(frames), payload.text)
        return {
            "video_embeddings": result["video_embeddings"].squeeze(0).tolist(),
            "text_embeddings": result["text_embeddings"].squeeze(0).tolist(),
            "fused_embeddings": result["fused_embeddings"].squeeze(0).tolist(),
        }

    def _encode(self, frames, text: str) -> dict:
        import torch.nn.functional as F

        torch = self._torch
        x = self._preprocess(frames)

        with torch.inference_mode():
            video_emb = self.model.get_vid_feat(x)
            text_emb = self.model.get_txt_feat(text)
            video_emb = F.normalize(video_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)
            fused = F.normalize((video_emb + text_emb) / 2.0, dim=-1)

        return {
            "video_embeddings": video_emb.float().cpu().numpy(),
            "text_embeddings": text_emb.float().cpu().numpy(),
            "fused_embeddings": fused.float().cpu().numpy(),
        }

    def _download(self, url: str) -> bytes:
        import urllib.error
        import urllib.request

        if url.startswith("gs://"):
            from google.cloud import storage

            bucket_name, _, blob_name = url[len("gs://"):].partition("/")
            if not bucket_name or not blob_name:
                raise ValueError("gs:// url must be gs://<bucket>/<object>")
            blob = storage.Client().bucket(bucket_name).blob(blob_name)
            try:
                # Size-check first so we don't pull a 10GB object before failing.
                blob.reload()
                if blob.size is not None and blob.size > MAX_VIDEO_BYTES:
                    raise ValueError(
                        f"video exceeds {MAX_VIDEO_BYTES} byte limit"
                    )
                data = blob.download_as_bytes(timeout=DOWNLOAD_TIMEOUT_S)
            except Exception as e:
                raise ValueError(f"failed to fetch video: {e}")
        elif url.startswith(("http://", "https://")):
            req = urllib.request.Request(
                url, headers={"User-Agent": "internvideo2-modal/1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
                    data = resp.read(MAX_VIDEO_BYTES + 1)
            except urllib.error.URLError as e:
                raise ValueError(f"failed to fetch video: {e}")
        else:
            raise ValueError("video_url must be http(s):// or gs://")

        if not data:
            raise ValueError("downloaded video is empty")
        if len(data) > MAX_VIDEO_BYTES:
            raise ValueError(
                f"video exceeds {MAX_VIDEO_BYTES} byte limit"
            )
        return data

    def _sample_frames(self, video_path: str, num_frames: int):
        # Inline copy of social_memory.encoder.sample_frames so the container
        # doesn't need the local package on its PYTHONPATH.
        from collections import Counter

        import av
        import numpy as np

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            total = stream.frames or 0

        if total <= 0:
            with av.open(video_path) as container:
                total = sum(1 for _ in container.decode(video=0))
        if total <= 0:
            raise ValueError("no decodable frames in video")

        targets = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        want = Counter(targets)
        frames: list = []
        with av.open(video_path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                k = want.get(i, 0)
                if k:
                    arr = frame.to_ndarray(format="rgb24")
                    for _ in range(k):
                        frames.append(arr)
                        if len(frames) == num_frames:
                            return frames
        if not frames:
            raise ValueError("no frames decoded")
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return frames

    def _preprocess(self, frames):
        import cv2
        import numpy as np

        torch = self._torch
        if frames.shape[0] != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, got {frames.shape[0]}"
            )
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

        resized = np.stack(
            [cv2.resize(f, (self.input_res, self.input_res)) for f in frames]
        ).astype(np.float32) / 255.0
        resized = (resized - mean) / std  # (T, H, W, 3)
        # InternVideo2 expects [B, T, C, H, W]
        x = torch.from_numpy(resized).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        return x.to("cuda", non_blocking=True)


@app.local_entrypoint()
def main(video_path: str = "assets/example.mp4", text: str = "a person talking"):
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, "src")
    from social_memory.encoder import sample_frames

    enc = InternVideo2Stage2()
    n = enc.num_frames_required.remote()
    print(f"server reports num_frames={n}")

    frames = sample_frames(Path(video_path), n)
    arr = np.stack(frames)
    out = enc.encode.remote(arr, text)
    for k, v in out.items():
        print(f"{k}: shape={v.shape}, norm={float(np.linalg.norm(v)):.4f}")
