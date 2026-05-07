"""Script to divide full videos into chunks while storing the ground truth trim index for each video."""

import json
import sys
from pathlib import Path

from tqdm import tqdm

# src/ layout: plain `python scripts/...` does not put `src` on sys.path unless the package is on
# that interpreter's site-packages. Keeps the script runnable even when the active `python` is not
# the one you installed with `uv pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_src = _REPO_ROOT / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

from social_memory.constants import GCS_BUCKET, GCS_PREFIX, PATH_TO_AUGMENTED_DATA
from social_memory.data_augmentation.chunking import compute_chunks_around_oracle
from social_memory.gcs import download_to_temp, list_blobs
from social_memory.utils import get_duration

BUFFER_TIME = 10 # seconds
CHUNK_SIZE = 60

with open(PATH_TO_AUGMENTED_DATA / "oracles.json", "r") as j:
    oracles = json.load(j)

all_chunks = {}

for blob in tqdm(list(list_blobs(bucket=GCS_BUCKET, prefix=GCS_PREFIX))):

    video_id = Path(blob.name).stem
    print(f"Processing video: {video_id}")

    with download_to_temp(GCS_BUCKET, blob.name) as path:
        if path is None:
            raise FileNotFoundError("missing blob")
        full_duration = get_duration(str(path))

    oracle = oracles.get(video_id)
    if not oracle:
        print(f"Could not find oracle for video id {video_id}")
        continue

    try:
        chunks, oracle_idx = compute_chunks_around_oracle(oracle, full_duration)
    except:
        print(f"Could not compute chunks for video id {video_id}")
        continue

    all_chunks[video_id] = {
        "chunks": chunks,
        "oracle_idx": oracle_idx,
    }

with open(PATH_TO_AUGMENTED_DATA / "video_chunks.json", "w") as j:
    json.dump(all_chunks, j, indent=2)

