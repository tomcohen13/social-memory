"""Modal app for X-CLIP adapter training.

Two entrypoints share one image + two volumes:

* `precompute_features` — loads X-CLIP on the GPU, streams chunk MP4s + VTTs
  from GCS (via `social_memory.gcs.download_to_temp`), encodes them, and
  writes `.npz` archives to the `social-memory-features` volume.
* `train` — loads cached features from the volume, builds the X-CLIP
  feature adapter (`XCLIPFeatureAdapter`), runs `social_memory.train.train`
  with the feature-based loss, and writes checkpoints to the
  `social-memory-checkpoints` volume. Streams metrics to Weights & Biases.

Configuration
-------------
Everything (GCP service-account JSON, GCS_BUCKET / GCS_PREFIX /
GCS_CHUNKS_PREFIX, WANDB_API_KEY, WANDB_ENTITY) lives in your local `.env`
and is forwarded to the Modal function as arguments by the
`local_entrypoint`. No Modal secrets required.

The `local_entrypoint` accepts either `GOOGLE_APPLICATION_CREDENTIALS_JSON`
(raw JSON) or `GOOGLE_APPLICATION_CREDENTIALS` (path) — whichever you have
set. Disable wandb for a run with `--no-wandb`.

Smoke tests
-----------
Precompute on 2 videos (X-CLIP, demo split)::

    modal run scripts/modal_train.py --mode precompute \\
        --encoder xclip --splits demo --limit-videos 2

Train one epoch on cached features::

    modal run scripts/modal_train.py --mode train \\
        --encoder xclip --epochs 1 --batch-size 4 --run-name smoke \\
        --wandb-project social-memory

Pull a checkpoint back::

    modal volume get social-memory-checkpoints smoke/best.pt ./best.pt
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from dotenv import load_dotenv

APP_NAME = "social-memory-train"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "social_memory"
DATA_DIR = REPO_ROOT / "datasets" / "siq2long"

FEATURES_DIR = "/features"
CKPT_DIR = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.44.2",
        "huggingface_hub>=0.24",
        "numpy<2",
        "pandas>=2.0",
        "av>=12",
        "opencv-python-headless",
        "Pillow",
        "google-cloud-storage>=2.18",
        "webvtt-py==0.4.6",
        "python-dotenv>=1.0",
        "tqdm>=4.65",
        "scikit-learn>=1.4",
        "wandb>=0.18",
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/social_memory")
    .add_local_dir(str(DATA_DIR), remote_path="/root/datasets/siq2long")
)

app = modal.App(APP_NAME, image=image)

features_vol = modal.Volume.from_name("social-memory-features", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("social-memory-checkpoints", create_if_missing=True)

def _setup_gcp_credentials(sa_json: str) -> None:
    """Materialize the SA JSON (passed from the local .env) onto disk so
    `google.cloud.storage.Client()` can pick it up via ADC."""
    sa_path = "/tmp/gcp-sa.json"
    with open(sa_path, "w") as f:
        f.write(sa_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path


def _set_gcs_env(bucket: str, prefix: str, chunks_prefix: str) -> None:
    """`social_memory.constants` reads these via os.getenv at import time —
    set them BEFORE importing any social_memory module."""
    os.environ["GCS_BUCKET"] = bucket
    os.environ["GCS_PREFIX"] = prefix
    os.environ["GCS_CHUNKS_PREFIX"] = chunks_prefix


@app.function(
    gpu="A10G",
    volumes={FEATURES_DIR: features_vol},
    timeout=6 * 3600,
)
def precompute_features(
    gcp_credentials_json: str,
    gcs_bucket: str,
    gcs_prefix: str = "siq2/video",
    gcs_chunks_prefix: str = "siq2/chunks",
    encoder_name: str = "xclip",
    splits: list[str] | None = None,
    limit_videos: int | None = None,
    skip_text: bool = False,
    skip_chunks: bool = False,
    max_chunks_per_video: int = 24,
) -> None:
    """Run X-CLIP over every chunk + QA text, dump .npz per video / split."""
    if splits is None:
        splits = ["train", "val"]
    _setup_gcp_credentials(gcp_credentials_json)
    _set_gcs_env(gcs_bucket, gcs_prefix, gcs_chunks_prefix)

    import numpy as np
    import torch

    from social_memory.constants import Datasets
    from social_memory.precompute import (
        chunk_features_exist,
        encode_text_for_split,
        encode_video_chunks,
        list_chunks_for_video,
        save_chunk_features,
    )
    from social_memory.utils import load_qa_dataset

    if encoder_name != "xclip":
        raise ValueError(
            f"only xclip is supported in this image (got {encoder_name!r})"
        )

    from social_memory.encoder import XCLIPEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading X-CLIP on {device}")
    encoder = XCLIPEncoder().to(device)
    encoder.eval()

    out_root = Path(FEATURES_DIR) / encoder_name
    chunk_dir = out_root / "chunks"
    text_dir = out_root / "text"

    split_dfs = {s: load_qa_dataset(str(Datasets.SIQ2LONG), s) for s in splits}
    all_videos = sorted(
        {v for df in split_dfs.values() for v in df["vid_name"].unique().tolist()}
    )
    if limit_videos:
        all_videos = all_videos[:limit_videos]
    print(f"target: {len(all_videos)} videos across splits {list(splits)}")

    if not skip_chunks:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for i, vid_name in enumerate(all_videos):
            try:
                chunks = list_chunks_for_video(vid_name)
            except Exception as e:
                print(f"  [{i+1}/{len(all_videos)}] [error] list {vid_name}: {e}")
                continue
            if not chunks:
                print(f"  [{i+1}/{len(all_videos)}] [warn] no chunks for {vid_name}")
                continue
            if len(chunks) > max_chunks_per_video:
                print(
                    f"  [{i+1}/{len(all_videos)}] [skip] {vid_name}: "
                    f"{len(chunks)} chunks > max {max_chunks_per_video}"
                )
                continue
            if chunk_features_exist(chunk_dir, vid_name, len(chunks)):
                print(f"  [{i+1}/{len(all_videos)}] [skip] {vid_name} cached")
                continue
            try:
                arrays = encode_video_chunks(encoder, vid_name, chunks=chunks)
            except Exception as e:
                print(f"  [{i+1}/{len(all_videos)}] [error] encode {vid_name}: {e}")
                continue
            path = save_chunk_features(chunk_dir, vid_name, arrays)
            print(
                f"  [{i+1}/{len(all_videos)}] [done] {vid_name} → {path.name} "
                f"({arrays['video_emb'].shape[0]} chunks, dim={arrays['video_emb'].shape[1]})"
            )
            if (i + 1) % 25 == 0:
                features_vol.commit()

    if not skip_text:
        text_dir.mkdir(parents=True, exist_ok=True)
        for split, df in split_dfs.items():
            target = text_dir / f"{split}.npz"
            if target.exists():
                with np.load(target, allow_pickle=True) as f:
                    if len(f["qids"]) == len(df):
                        print(f"  [skip] text/{split}.npz")
                        continue
            arrays = encode_text_for_split(encoder, df)
            np.savez(target, **arrays)
            print(
                f"  [done] text/{split}.npz "
                f"({len(arrays['qids'])} rows, dim={arrays['q_emb'].shape[1]})"
            )

    features_vol.commit()
    print("precompute complete")


@app.function(
    gpu="A10G",
    volumes={FEATURES_DIR: features_vol},
    timeout=600,
)
def profile_video(
    gcp_credentials_json: str,
    gcs_bucket: str,
    gcs_prefix: str = "siq2/video",
    gcs_chunks_prefix: str = "siq2/chunks",
    vid_name: str = "9qK9VQDELpc",
) -> None:
    """Time download / decode / forward for every chunk of one video.

    Use this to figure out whether the X-CLIP forward, the av decode, or
    the GCS download is the bottleneck before refactoring anything.
    """
    _setup_gcp_credentials(gcp_credentials_json)
    _set_gcs_env(gcs_bucket, gcs_prefix, gcs_chunks_prefix)

    import statistics
    import time

    import torch

    from social_memory.constants import GCS_BUCKET as _BUCKET
    from social_memory.encoder import XCLIPEncoder, sample_frames
    from social_memory.gcs import download_to_temp
    from social_memory.precompute import list_chunks_for_video
    from social_memory.utils import read_vtt_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading X-CLIP on {device}")
    t0 = time.perf_counter()
    encoder = XCLIPEncoder().to(device)
    encoder.eval()
    print(f"  model load: {time.perf_counter() - t0:.1f}s")

    print(f"profiling {vid_name}")
    chunks = list_chunks_for_video(vid_name)
    print(f"  {len(chunks)} chunks")

    stats: dict[str, list[float]] = {
        "dl_mp4": [],
        "dl_vtt": [],
        "decode": [],
        "fwd_video": [],
        "fwd_text": [],
    }

    for chunk_idx, mp4_blob, vtt_blob in chunks:
        transcript = ""
        if vtt_blob is not None:
            t = time.perf_counter()
            with download_to_temp(_BUCKET, vtt_blob) as vp:
                stats["dl_vtt"].append(time.perf_counter() - t)
                if vp is not None:
                    transcript = read_vtt_file(vp).strip()

        t = time.perf_counter()
        with download_to_temp(_BUCKET, mp4_blob) as mp4_path:
            stats["dl_mp4"].append(time.perf_counter() - t)
            if mp4_path is None:
                print(f"  chunk {chunk_idx}: missing mp4")
                continue
            t = time.perf_counter()
            frames = sample_frames(mp4_path, num_frames=encoder.num_frames)
            stats["decode"].append(time.perf_counter() - t)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        # Replicate XCLIPEncoder forward but split video vs text fwd timing.
        with torch.inference_mode():
            text_inputs = encoder.processor.tokenizer(
                [transcript or " "], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            video_inputs = encoder.processor.image_processor(
                [frames], return_tensors="pt"
            ).to(device)
            t_split = time.perf_counter()
            _ = encoder.model.get_video_features(
                pixel_values=video_inputs["pixel_values"]
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats["fwd_video"].append(time.perf_counter() - t_split)
            t_split = time.perf_counter()
            _ = encoder.model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats["fwd_text"].append(time.perf_counter() - t_split)

    print(f"\nper-stage timings over {len(chunks)} chunks (ms):")
    for k, vals in stats.items():
        if not vals:
            print(f"  {k}: (none)")
            continue
        mean_ms = statistics.mean(vals) * 1000
        med_ms = statistics.median(vals) * 1000
        mx_ms = max(vals) * 1000
        total_s = sum(vals)
        print(
            f"  {k:10s}: mean={mean_ms:6.0f} med={med_ms:6.0f} max={mx_ms:6.0f} "
            f"total={total_s:5.1f}s"
        )

    per_chunk = sum(sum(v) for v in stats.values()) / max(len(chunks), 1)
    print(f"\ntotal per-chunk wall: {per_chunk*1000:.0f}ms")
    print(f"projected: 500 vids × {len(chunks)} chunks × {per_chunk:.2f}s = "
          f"{500 * len(chunks) * per_chunk / 60:.0f} min on 1 GPU")


@app.function(
    gpu="A100-40GB",
    volumes={FEATURES_DIR: features_vol, CKPT_DIR: ckpt_vol},
    timeout=24 * 3600,
)
def train(
    encoder_name: str = "xclip",
    run_name: str = "default",
    split: str = "train",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    temperature: float = 0.07,
    hidden_dim: int = 512,
    output_dim: int = 256,
    num_workers: int = 2,
    max_chunks_per_video: int = 24,
    wandb_project: str = "social-memory",
    wandb_entity: str | None = None,
    wandb_api_key: str | None = None,
    use_wandb: bool = True,
) -> None:
    """Train the X-CLIP adapter on precomputed features."""
    import torch
    from torch.utils.data import DataLoader

    from social_memory.feature_dataset import (
        FeatureCachedDataset,
        XCLIPFeatureAdapter,
        feature_collate_fn,
    )
    from social_memory.train import compute_batch_loss_features, train as run_train

    features_root = Path(FEATURES_DIR) / encoder_name
    if not (features_root / "chunks").exists():
        raise FileNotFoundError(
            f"no precomputed chunks at {features_root}/chunks — "
            f"run precompute first"
        )

    dataset = FeatureCachedDataset(
        split=split,
        features_root=features_root,
        max_chunks_per_video=max_chunks_per_video,
    )
    print(f"dataset: {len(dataset)} videos | feat_dim={dataset.dim}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=feature_collate_fn,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = XCLIPFeatureAdapter(
        feat_dim=dataset.dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"adapter on {device} | trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    out_ckpt_dir = Path(CKPT_DIR) / run_name
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_callback = None
    if use_wandb and wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        import wandb

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "encoder": encoder_name,
                "split": split,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "temperature": temperature,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "feat_dim": dataset.dim,
                "n_trainable": n_trainable,
                "max_chunks_per_video": max_chunks_per_video,
                "dataset_size": len(dataset),
            },
        )
        log_callback = wandb.log
    elif use_wandb:
        print("WARN: no WANDB_API_KEY forwarded — skipping wandb logging")

    try:
        run_train(
            model,
            dataloader,
            optimizer,
            num_epochs=epochs,
            temperature=temperature,
            ckpt_dir=str(out_ckpt_dir),
            compute_loss_fn=compute_batch_loss_features,
            log_callback=log_callback,
        )
    finally:
        if log_callback is not None:
            import wandb

            wandb.finish()
        ckpt_vol.commit()

    print(f"training complete → checkpoints volume:{run_name}/")


def _load_gcp_credentials_json() -> str:
    """Resolve the SA JSON contents from whichever form the local env uses.

    Accepts either `GOOGLE_APPLICATION_CREDENTIALS_JSON` (raw JSON string)
    or `GOOGLE_APPLICATION_CREDENTIALS` (path to a JSON file on disk).
    """
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if raw:
        return raw
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise SystemExit(
                f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {p}"
            )
        return p.read_text()
    raise SystemExit(
        "Neither GOOGLE_APPLICATION_CREDENTIALS_JSON nor "
        "GOOGLE_APPLICATION_CREDENTIALS is set in your .env"
    )


@app.local_entrypoint()
def main(
    mode: str = "train",
    encoder: str = "xclip",
    splits: str = "train,val",
    limit_videos: int = 0,
    skip_text: bool = False,
    skip_chunks: bool = False,
    vid_name: str = "9qK9VQDELpc",
    run_name: str = "default",
    split: str = "train",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    temperature: float = 0.07,
    hidden_dim: int = 512,
    output_dim: int = 256,
    num_workers: int = 2,
    max_chunks_per_video: int = 24,
    wandb_project: str = "social-memory",
    wandb_entity: str = "",
    no_wandb: bool = False,
) -> None:
    """Dispatch to precompute_features or train based on --mode.

    All credentials and config (GCP SA JSON, GCS_*, WANDB_API_KEY,
    WANDB_ENTITY) are read from your local .env and forwarded as function
    args — nothing lands in source or Modal secrets.
    """
    load_dotenv()  # populate from <repo>/.env on the host

    if mode == "profile":
        bucket = os.environ.get("GCS_BUCKET")
        if not bucket:
            raise SystemExit("GCS_BUCKET not set in your .env / shell env")
        profile_video.remote(
            gcp_credentials_json=_load_gcp_credentials_json(),
            gcs_bucket=bucket,
            gcs_prefix=os.environ.get("GCS_PREFIX", "siq2/video"),
            gcs_chunks_prefix=os.environ.get("GCS_CHUNKS_PREFIX", "siq2/chunks"),
            vid_name=vid_name,
        )
        return
    if mode == "precompute":
        bucket = os.environ.get("GCS_BUCKET")
        if not bucket:
            raise SystemExit("GCS_BUCKET not set in your .env / shell env")
        # .spawn() returns immediately; the function runs on Modal and is
        # immune to local-caller disconnects (unlike .remote() under --detach).
        call = precompute_features.spawn(
            gcp_credentials_json=_load_gcp_credentials_json(),
            gcs_bucket=bucket,
            gcs_prefix=os.environ.get("GCS_PREFIX", "siq2/video"),
            gcs_chunks_prefix=os.environ.get("GCS_CHUNKS_PREFIX", "siq2/chunks"),
            encoder_name=encoder,
            splits=[s.strip() for s in splits.split(",") if s.strip()],
            limit_videos=limit_videos or None,
            skip_text=skip_text,
            skip_chunks=skip_chunks,
            max_chunks_per_video=max_chunks_per_video,
        )
        print(f"spawned precompute (function call id: {call.object_id})")
        print("track with:  modal app logs <app-id>   (see Modal UI for app id)")
    elif mode == "train":
        train.remote(
            encoder_name=encoder,
            run_name=run_name,
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            temperature=temperature,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_workers=num_workers,
            max_chunks_per_video=max_chunks_per_video,
            wandb_project=wandb_project,
            wandb_entity=(wandb_entity or os.environ.get("WANDB_ENTITY") or None),
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            use_wandb=not no_wandb,
        )
    else:
        raise SystemExit(f"unknown --mode {mode!r} (expected precompute or train)")
