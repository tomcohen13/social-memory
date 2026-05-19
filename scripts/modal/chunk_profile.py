"""Modal app: per-stage timing breakdown for one X-CLIP video.

Use this to figure out whether GCS download, av decode, or the X-CLIP
forward is the per-chunk bottleneck before optimising the precompute
pipeline. Lives in its own file (its own `modal.App`) so it doesn't
clutter `modal/train.py`.

NOTE: do NOT name this file `profile.py` — that shadows the stdlib
`profile` module, which `cProfile` (pulled in by torch._dynamo via
transformers) needs to import.

Run::

    modal run scripts/modal/chunk_profile.py --vid-name 9qK9VQDELpc
"""

from __future__ import annotations

import os

import modal
from dotenv import load_dotenv

from _helpers import (
    FEATURES_DIR,
    FEATURES_VOL,
    IMAGE,
    load_gcp_credentials_json,
    require_gcs_bucket,
    set_gcs_env,
    setup_gcp_credentials,
)

app = modal.App("social-memory-profile", image=IMAGE)


@app.function(
    gpu="A10G",
    cpu=4.0,
    volumes={FEATURES_DIR: FEATURES_VOL},
    timeout=600,
)
def profile_video(
    gcp_credentials_json: str,
    gcs_bucket: str,
    gcs_prefix: str = "siq2/video",
    gcs_chunks_prefix: str = "siq2/chunks",
    vid_name: str = "9qK9VQDELpc",
) -> None:
    """Time download / decode / forward for every chunk of one video."""
    setup_gcp_credentials(gcp_credentials_json)
    set_gcs_env(gcs_bucket, gcs_prefix, gcs_chunks_prefix)

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
    print(
        f"projected: 500 vids × {len(chunks)} chunks × {per_chunk:.2f}s = "
        f"{500 * len(chunks) * per_chunk / 60:.0f} min on 1 GPU"
    )


@app.local_entrypoint()
def main(vid_name: str = "9qK9VQDELpc") -> None:
    load_dotenv()
    profile_video.remote(
        gcp_credentials_json=load_gcp_credentials_json(),
        gcs_bucket=require_gcs_bucket(),
        gcs_prefix=os.environ.get("GCS_PREFIX", "siq2/video"),
        gcs_chunks_prefix=os.environ.get("GCS_CHUNKS_PREFIX", "siq2/chunks"),
        vid_name=vid_name,
    )
