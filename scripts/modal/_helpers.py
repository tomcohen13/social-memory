"""Shared Modal scaffolding for `scripts/modal/*.py`.

Centralises three things that every Modal app in this repo otherwise
duplicates:

* `IMAGE` — the debian-slim + torch + transformers + GCS + wandb image,
  with `src/social_memory/` and `datasets/siq2long/` mounted in.
* GCS / GCP credential helpers — env-var bootstrap, SA JSON resolution
  from the host `.env`.
* Volume handles — `FEATURES_VOL`, `CKPT_VOL` so all apps point at the
  same persistent storage.

Keep this module import-light: it runs on both the local host (to read
`.env`) and inside the Modal container (to set up creds before any
`social_memory` import). No `torch` / `transformers` at module top.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# scripts/modal/_helpers.py → parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "social_memory"
DATA_DIR = REPO_ROOT / "datasets" / "siq2long"

FEATURES_DIR = "/features"
CKPT_DIR = "/checkpoints"


IMAGE = (
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

FEATURES_VOL = modal.Volume.from_name("social-memory-features", create_if_missing=True)
CKPT_VOL = modal.Volume.from_name("social-memory-checkpoints", create_if_missing=True)


def setup_gcp_credentials(sa_json: str) -> None:
    """Materialise the SA JSON (passed from the local .env) onto disk so
    `google.cloud.storage.Client()` picks it up via ADC."""
    sa_path = "/tmp/gcp-sa.json"
    with open(sa_path, "w") as f:
        f.write(sa_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path


def set_gcs_env(bucket: str, prefix: str, chunks_prefix: str) -> None:
    """`social_memory.constants` reads these via `os.getenv` at import time —
    set them BEFORE importing any `social_memory` module inside a Modal
    function."""
    os.environ["GCS_BUCKET"] = bucket
    os.environ["GCS_PREFIX"] = prefix
    os.environ["GCS_CHUNKS_PREFIX"] = chunks_prefix


def load_gcp_credentials_json() -> str:
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


def require_gcs_bucket() -> str:
    bucket = os.environ.get("GCS_BUCKET")
    if not bucket:
        raise SystemExit("GCS_BUCKET not set in your .env / shell env")
    return bucket
