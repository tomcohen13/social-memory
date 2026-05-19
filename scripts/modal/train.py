"""Modal app for X-CLIP adapter training.

Two thin wrappers around library code in `social_memory`:

* `precompute_features` → `social_memory.precompute.precompute_features_run`
* `train`               → `social_memory.train.train_features_run`

Everything Modal-specific (image, volumes, credential bootstrap) lives
in `scripts/modal/_helpers.py`. The one-off chunk profiler lives in
`scripts/modal/chunk_profile.py`. This file should read top-to-bottom as just:
image → volumes → two wrappers → CLI dispatch.

Configuration
-------------
Credentials and config (GCP SA JSON, GCS_BUCKET / GCS_PREFIX /
GCS_CHUNKS_PREFIX, WANDB_API_KEY, WANDB_ENTITY) live in your local
`.env` and are forwarded to the Modal function as arguments by
`local_entrypoint`. No Modal secrets required.

Smoke tests
-----------
Precompute on 2 videos (X-CLIP, demo split)::

    modal run scripts/modal/train.py --mode precompute \\
        --encoder xclip --splits demo --limit-videos 2

Train one epoch on cached features::

    modal run scripts/modal/train.py --mode train \\
        --encoder xclip --epochs 1 --batch-size 4 --run-name smoke

Pull a checkpoint back::

    modal volume get social-memory-checkpoints smoke/best.pt ./best.pt
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from dotenv import load_dotenv

from _helpers import (
    CKPT_DIR,
    CKPT_VOL,
    FEATURES_DIR,
    FEATURES_VOL,
    IMAGE,
    load_gcp_credentials_json,
    require_gcs_bucket,
    set_gcs_env,
    setup_gcp_credentials,
)

app = modal.App("social-memory-train", image=IMAGE)


@app.function(
    gpu="A10G",
    volumes={FEATURES_DIR: FEATURES_VOL},
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
    """Thin wrapper: bootstrap GCS creds, build encoder, call shared driver."""
    if splits is None:
        splits = ["train", "val"]
    setup_gcp_credentials(gcp_credentials_json)
    set_gcs_env(gcs_bucket, gcs_prefix, gcs_chunks_prefix)

    import torch

    from social_memory.constants import Datasets
    from social_memory.precompute import precompute_features_run
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

    split_dfs = {s: load_qa_dataset(str(Datasets.SIQ2LONG), s) for s in splits}

    precompute_features_run(
        encoder=encoder,
        out_root=Path(FEATURES_DIR) / encoder_name,
        split_dfs=split_dfs,
        limit_videos=limit_videos,
        skip_text=skip_text,
        skip_chunks=skip_chunks,
        max_chunks_per_video=max_chunks_per_video,
        commit_callback=FEATURES_VOL.commit,
    )


@app.function(
    gpu="A100-40GB",
    volumes={FEATURES_DIR: FEATURES_VOL, CKPT_DIR: CKPT_VOL},
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
    ckpt_every: int = 5,
    wandb_project: str = "social-memory",
    wandb_entity: str | None = None,
    wandb_api_key: str | None = None,
    use_wandb: bool = True,
) -> None:
    """Thin wrapper: resolve volume paths, build wandb config, call shared driver."""
    from social_memory.train import train_features_run

    out_ckpt_dir = Path(CKPT_DIR) / run_name
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_config = None
    if use_wandb:
        if wandb_api_key:
            wandb_config = {
                "api_key": wandb_api_key,
                "project": wandb_project,
                "entity": wandb_entity,
                "run_name": run_name,
                "extra_config": {"encoder": encoder_name},
            }
        else:
            print("WARN: no WANDB_API_KEY forwarded — skipping wandb logging")

    train_features_run(
        features_root=Path(FEATURES_DIR) / encoder_name,
        ckpt_dir=out_ckpt_dir,
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_workers=num_workers,
        max_chunks_per_video=max_chunks_per_video,
        ckpt_every=ckpt_every,
        wandb_config=wandb_config,
        post_save_callback=CKPT_VOL.commit,
    )

    print(f"training complete → checkpoints volume:{run_name}/")


@app.local_entrypoint()
def main(
    mode: str = "train",
    encoder: str = "xclip",
    splits: str = "train,val",
    limit_videos: int = 0,
    skip_text: bool = False,
    skip_chunks: bool = False,
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
    ckpt_every: int = 5,
    wandb_project: str = "social-memory",
    wandb_entity: str = "",
    no_wandb: bool = False,
) -> None:
    """Dispatch to `precompute_features` or `train` based on `--mode`."""
    load_dotenv()  # populate from <repo>/.env on the host

    if mode == "precompute":
        # .spawn() returns immediately; the function runs on Modal and is
        # immune to local-caller disconnects (unlike .remote() under --detach).
        call = precompute_features.spawn(
            gcp_credentials_json=load_gcp_credentials_json(),
            gcs_bucket=require_gcs_bucket(),
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
            ckpt_every=ckpt_every,
            wandb_project=wandb_project,
            wandb_entity=(wandb_entity or os.environ.get("WANDB_ENTITY") or None),
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            use_wandb=not no_wandb,
        )
    else:
        raise SystemExit(f"unknown --mode {mode!r} (expected precompute or train)")
