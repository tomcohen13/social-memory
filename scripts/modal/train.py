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
    cpu=4.0,
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
    num_workers: int = 4,
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
        num_workers=num_workers,
        commit_callback=FEATURES_VOL.commit,
    )


@app.function(
    gpu="A100-40GB",
    volumes={FEATURES_DIR: FEATURES_VOL, CKPT_DIR: CKPT_VOL},
    timeout=24 * 3600,
)
def train(
    encoder_name: str = "xclip",
    adapter: str = "xclip",
    run_name: str = "default",
    split: str = "train",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-5,
    temperature: float = 0.07,
    hidden_dim: int = 512,
    output_dim: int = 256,
    num_workers: int = 0,
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
                "extra_config": {"encoder": encoder_name, "adapter": adapter},
            }
        else:
            print("WARN: no WANDB_API_KEY forwarded — skipping wandb logging")

    train_features_run(
        features_root=Path(FEATURES_DIR) / encoder_name,
        ckpt_dir=out_ckpt_dir,
        adapter=adapter,
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


@app.function(
    gpu="A100-40GB",
    volumes={FEATURES_DIR: FEATURES_VOL, CKPT_DIR: CKPT_VOL},
    timeout=24 * 3600,
)
def train_li(
    encoder_name: str = "xclip",
    run_name: str = "li-default",
    split: str = "train",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    hidden_dim: int = 512,
    output_dim: int = 128,
    num_chunk_tokens: int = 8,
    num_qformer_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.0,
    init_temperature: float = 0.07,
    num_workers: int = 0,
    max_chunks_per_video: int = 24,
    ckpt_every: int = 5,
    wandb_project: str = "social-memory",
    wandb_entity: str | None = None,
    wandb_api_key: str | None = None,
    use_wandb: bool = True,
) -> None:
    """Thin wrapper: resolve volume paths, build wandb config, call LI driver."""
    from social_memory.train import train_late_interaction_run

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

    train_late_interaction_run(
        features_root=Path(FEATURES_DIR) / encoder_name,
        ckpt_dir=out_ckpt_dir,
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_chunk_tokens=num_chunk_tokens,
        num_qformer_layers=num_qformer_layers,
        num_heads=num_heads,
        dropout=dropout,
        init_temperature=init_temperature,
        num_workers=num_workers,
        max_chunks_per_video=max_chunks_per_video,
        ckpt_every=ckpt_every,
        wandb_config=wandb_config,
        post_save_callback=CKPT_VOL.commit,
    )

    print(f"LI training complete → checkpoints volume:{run_name}/")


@app.function(
    gpu="A10G",
    volumes={FEATURES_DIR: FEATURES_VOL, CKPT_DIR: CKPT_VOL},
    timeout=2 * 3600,
)
def validate(
    encoder_name: str = "xclip",
    adapter: str = "xclip",
    run_name: str = "xclip-adapter-full-30ep-lr5",
    split: str = "val",
    ckpts: list[str] | None = None,
    batch_size: int = 16,
    temperature: float = 0.07,
    hidden_dim: int = 512,
    output_dim: int = 256,
    max_chunks_per_video: int = 24,
    topk: tuple[int, ...] = (1, 3),
    fuse_modes: tuple[str, ...] = ("fused", "text", "video"),
) -> list[dict]:
    """Load each checkpoint in `<CKPT_DIR>/<run_name>`, run inference on the
    val split's precomputed features, report top-k oracle-chunk accuracy and
    contrastive val loss per checkpoint and per `fuse_mode`.

    `fuse_modes` controls what gets fed into the chunk adapter at eval time:
    - "fused": (video_emb + transcript_emb) / 2  (training-time input)
    - "text":  transcript_emb only
    - "video": video_emb only
    Single-modality inputs are slightly OOD for an MLP trained on the fused
    input, but the contrast reveals which modality drives the score.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from social_memory.feature_dataset import (
        EarlyFusionAdapter,
        FeatureCachedDataset,
        XCLIPFeatureAdapter,
        feature_collate_fn,
    )

    adapter_classes = {
        "xclip": XCLIPFeatureAdapter,
        "early_fusion": EarlyFusionAdapter,
    }
    if adapter not in adapter_classes:
        raise ValueError(
            f"unknown adapter {adapter!r} (expected one of {sorted(adapter_classes)})"
        )

    features_root = Path(FEATURES_DIR) / encoder_name
    ckpt_dir = Path(CKPT_DIR) / run_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    print(f"evaluating split={split!r} | run={run_name!r} | adapter={adapter!r}")

    if ckpts:
        ckpt_paths = []
        for n in ckpts:
            p = ckpt_dir / (n if n.endswith(".pt") else f"{n}.pt")
            if not p.exists():
                print(f"WARN: missing checkpoint {p}, skipping")
                continue
            ckpt_paths.append(p)
    else:
        ckpt_paths = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"no checkpoints to evaluate in {ckpt_dir}")

    dataset = FeatureCachedDataset(
        split=split,
        features_root=features_root,
        max_chunks_per_video=max_chunks_per_video,
    )
    print(f"val dataset: {len(dataset)} videos | feat_dim={dataset.dim}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=feature_collate_fn,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = adapter_classes[adapter](
        feat_dim=dataset.dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)
    model.eval()

    def _chunk_embs_for_mode(chunks, mode: str) -> torch.Tensor:
        """Run the chunk adapter on a single-modality or fused input."""
        if mode == "fused":
            return model.encode_chunks(
                [c["video_emb"] for c in chunks],
                [c["transcript_emb"] for c in chunks],
            )
        key = {"text": "transcript_emb", "video": "video_emb"}[mode]
        x = torch.stack(
            [model._to_tensor(c[key]) for c in chunks], dim=0
        )
        out = model.chunk_adapter(x)
        return F.normalize(out, dim=-1)

    results: list[dict] = []
    for p in ckpt_paths:
        state = torch.load(p, map_location=device)
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(sd)
        model.eval()
        epoch = int(state["epoch"]) if isinstance(state, dict) and "epoch" in state else -1

        for mode in fuse_modes:
            n_q_total = 0
            n_v_total = 0
            correct = {k: 0 for k in topk}
            loss_sum = 0.0
            loss_n = 0

            with torch.no_grad():
                for batch in dataloader:
                    if not batch.get("vid_name"):
                        continue
                    for v_idx in range(len(batch["vid_name"])):
                        chunks = sorted(
                            batch["chunks"][v_idx].values(),
                            key=lambda c: c["chunk_idx"],
                        )
                        sorted_chunk_ids = [c["chunk_idx"] for c in chunks]
                        oracle = batch["oracle_idx"][v_idx]
                        if oracle not in sorted_chunk_ids:
                            continue
                        oracle_pos = sorted_chunk_ids.index(oracle)

                        chunk_embs = _chunk_embs_for_mode(chunks, mode)
                        q_emb = batch["q_emb"][v_idx]
                        question_embs = model.encode_questions(q_emb)

                        logits = (question_embs @ chunk_embs.T) / temperature
                        n_q = int(question_embs.shape[0])
                        labels = torch.full(
                            (n_q,), oracle_pos, dtype=torch.long, device=logits.device,
                        )
                        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
                        loss_n += n_q

                        max_k = min(max(topk), logits.shape[1])
                        top_idx = logits.topk(max_k, dim=1).indices
                        hit = (top_idx == oracle_pos)
                        for k in topk:
                            kk = min(k, logits.shape[1])
                            correct[k] += int(hit[:, :kk].any(dim=1).sum().item())

                        n_q_total += n_q
                        n_v_total += 1

            metrics = {
                "ckpt": p.name,
                "epoch": epoch,
                "fuse_mode": mode,
                "n_videos": n_v_total,
                "n_questions": n_q_total,
                "val_loss": (loss_sum / loss_n) if loss_n else float("nan"),
            }
            for k in topk:
                metrics[f"top{k}_acc"] = (correct[k] / n_q_total) if n_q_total else float("nan")
            results.append(metrics)
            topk_str = " ".join(f"top{k}={metrics[f'top{k}_acc']:.4f}" for k in topk)
            print(
                f"{p.name:>20s} | epoch {epoch:>3d} | mode={mode:>5s} | "
                f"loss {metrics['val_loss']:.4f} | {topk_str} | "
                f"n_q={n_q_total} n_v={n_v_total}"
            )

    return results


@app.function(
    gpu="A10G",
    volumes={FEATURES_DIR: FEATURES_VOL, CKPT_DIR: CKPT_VOL},
    timeout=2 * 3600,
)
def validate_li(
    encoder_name: str = "xclip",
    run_name: str = "li-train-002",
    split: str = "val",
    ckpts: list[str] | None = None,
    batch_size: int = 16,
    hidden_dim: int = 512,
    output_dim: int = 128,
    num_chunk_tokens: int = 8,
    num_qformer_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.0,
    init_temperature: float = 0.07,
    max_chunks_per_video: int = 24,
    topk: tuple[int, ...] = (1, 3),
    fuse_modes: tuple[str, ...] = ("fused", "text", "video"),
) -> list[dict]:
    """Validate a `LateInteractionAdapter` checkpoint via MaxSim scoring.

    Mirrors `validate` but builds a multi-vector adapter, scores via
    `LateInteractionAdapter.maxsim_matrix(...) * model.temperature()`, and
    reports the same top-k oracle accuracy + cross-entropy loss.

    `fuse_modes` mirrors `validate`: for single-modality ablations we zero out
    the dropped modality's input tokens before `encode_chunk`. The Q-Former
    still sees the modality embedding for the zeroed stream, so this is
    slightly OOD vs. training (where both inputs carry real signal).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from social_memory.adapters.late_interaction import LateInteractionAdapter
    from social_memory.feature_dataset import (
        FeatureCachedDataset,
        feature_collate_fn,
    )

    features_root = Path(FEATURES_DIR) / encoder_name
    ckpt_dir = Path(CKPT_DIR) / run_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    print(f"evaluating split={split!r} | run={run_name!r}")

    if ckpts:
        ckpt_paths = []
        for n in ckpts:
            p = ckpt_dir / (n if n.endswith(".pt") else f"{n}.pt")
            if not p.exists():
                print(f"WARN: missing checkpoint {p}, skipping")
                continue
            ckpt_paths.append(p)
    else:
        ckpt_paths = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"no checkpoints to evaluate in {ckpt_dir}")

    dataset = FeatureCachedDataset(
        split=split,
        features_root=features_root,
        max_chunks_per_video=max_chunks_per_video,
    )
    print(f"val dataset: {len(dataset)} videos | feat_dim={dataset.dim}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=feature_collate_fn,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LateInteractionAdapter(
        d_video=dataset.dim,
        d_text=dataset.dim,
        d_out=output_dim,
        d_hidden=hidden_dim,
        num_chunk_tokens=num_chunk_tokens,
        num_qformer_layers=num_qformer_layers,
        num_heads=num_heads,
        dropout=dropout,
        init_temperature=init_temperature,
    ).to(device)
    model.eval()

    import numpy as np

    results: list[dict] = []
    for p in ckpt_paths:
        state = torch.load(p, map_location=device)
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(sd)
        model.eval()
        epoch = int(state["epoch"]) if isinstance(state, dict) and "epoch" in state else -1

        for mode in fuse_modes:
            n_q_total = 0
            n_v_total = 0
            correct = {k: 0 for k in topk}
            loss_sum = 0.0
            loss_n = 0

            with torch.no_grad():
                for batch in dataloader:
                    if not batch.get("vid_name"):
                        continue
                    for v_idx in range(len(batch["vid_name"])):
                        chunks = sorted(
                            batch["chunks"][v_idx].values(),
                            key=lambda c: c["chunk_idx"],
                        )
                        sorted_chunk_ids = [c["chunk_idx"] for c in chunks]
                        oracle = batch["oracle_idx"][v_idx]
                        if oracle not in sorted_chunk_ids:
                            continue
                        oracle_pos = sorted_chunk_ids.index(oracle)

                        v = torch.from_numpy(
                            np.stack([np.asarray(c["video_emb"], dtype=np.float32) for c in chunks])
                        ).to(device).unsqueeze(1)
                        t = torch.from_numpy(
                            np.stack([np.asarray(c["transcript_emb"], dtype=np.float32) for c in chunks])
                        ).to(device).unsqueeze(1)
                        if mode == "text":
                            v = torch.zeros_like(v)
                        elif mode == "video":
                            t = torch.zeros_like(t)
                        elif mode != "fused":
                            raise ValueError(f"unknown fuse_mode {mode!r}")
                        chunk_tokens = model.encode_chunk(v, t)  # (Nc, K, d_out)

                        q_emb = batch["q_emb"][v_idx]
                        if isinstance(q_emb, torch.Tensor):
                            q = q_emb.to(device=device, dtype=torch.float32)
                        else:
                            q = torch.from_numpy(np.asarray(q_emb, dtype=np.float32)).to(device)
                        if q.ndim == 1:
                            q = q.unsqueeze(0)
                        q = q.unsqueeze(1)  # (Nq, 1, D)
                        question_tokens = model.encode_question(q)  # (Nq, 1, d_out)

                        sims = LateInteractionAdapter.maxsim_matrix(
                            question_tokens, chunk_tokens
                        )  # (Nq, Nc)
                        logits = sims * model.temperature()

                        n_q = int(question_tokens.shape[0])
                        labels = torch.full(
                            (n_q,), oracle_pos, dtype=torch.long, device=logits.device,
                        )
                        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
                        loss_n += n_q

                        max_k = min(max(topk), logits.shape[1])
                        top_idx = logits.topk(max_k, dim=1).indices
                        hit = (top_idx == oracle_pos)
                        for k in topk:
                            kk = min(k, logits.shape[1])
                            correct[k] += int(hit[:, :kk].any(dim=1).sum().item())

                        n_q_total += n_q
                        n_v_total += 1

            metrics = {
                "ckpt": p.name,
                "epoch": epoch,
                "fuse_mode": mode,
                "n_videos": n_v_total,
                "n_questions": n_q_total,
                "val_loss": (loss_sum / loss_n) if loss_n else float("nan"),
            }
            for k in topk:
                metrics[f"top{k}_acc"] = (correct[k] / n_q_total) if n_q_total else float("nan")
            results.append(metrics)
            topk_str = " ".join(f"top{k}={metrics[f'top{k}_acc']:.4f}" for k in topk)
            print(
                f"{p.name:>20s} | epoch {epoch:>3d} | mode={mode:>5s} | "
                f"loss {metrics['val_loss']:.4f} | {topk_str} | "
                f"n_q={n_q_total} n_v={n_v_total}"
            )

    return results


@app.local_entrypoint()
def main(
    mode: str = "train",
    encoder: str = "xclip",
    adapter: str = "xclip",
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
    num_workers: int = 0,
    max_chunks_per_video: int = 24,
    ckpt_every: int = 5,
    wandb_project: str = "social-memory",
    wandb_entity: str = "",
    no_wandb: bool = False,
    ckpts: str = "",
    num_chunk_tokens: int = 8,
    num_qformer_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.0,
    init_temperature: float = 0.07,
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
            num_workers=num_workers,
        )
        print(f"spawned precompute (function call id: {call.object_id})")
        print("track with:  modal app logs <app-id>   (see Modal UI for app id)")
    elif mode == "validate":
        ckpt_list = [c.strip() for c in ckpts.split(",") if c.strip()] or None
        # Entrypoint `--split` default is "train" (for train mode). For
        # validate we default to "val" unless the user explicitly asked for
        # something else (e.g. --split test).
        val_split = "val" if split == "train" else split
        validate.remote(
            encoder_name=encoder,
            adapter=adapter,
            run_name=run_name,
            split=val_split,
            ckpts=ckpt_list,
            batch_size=batch_size,
            temperature=temperature,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_chunks_per_video=max_chunks_per_video,
        )
    elif mode == "validate_li":
        ckpt_list = [c.strip() for c in ckpts.split(",") if c.strip()] or None
        val_split = "val" if split == "train" else split
        validate_li.remote(
            encoder_name=encoder,
            run_name=run_name,
            split=val_split,
            ckpts=ckpt_list,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_chunk_tokens=num_chunk_tokens,
            num_qformer_layers=num_qformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            init_temperature=init_temperature,
            max_chunks_per_video=max_chunks_per_video,
        )
    elif mode == "train":
        train.remote(
            encoder_name=encoder,
            adapter=adapter,
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
    elif mode == "train_li":
        train_li.remote(
            encoder_name=encoder,
            run_name=run_name,
            split=split,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_chunk_tokens=num_chunk_tokens,
            num_qformer_layers=num_qformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            init_temperature=init_temperature,
            num_workers=num_workers,
            max_chunks_per_video=max_chunks_per_video,
            ckpt_every=ckpt_every,
            wandb_project=wandb_project,
            wandb_entity=(wandb_entity or os.environ.get("WANDB_ENTITY") or None),
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            use_wandb=not no_wandb,
        )
    else:
        raise SystemExit(
            f"unknown --mode {mode!r} (expected precompute, train, train_li, validate, or validate_li)"
        )
