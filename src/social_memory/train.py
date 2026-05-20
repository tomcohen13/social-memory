
import time, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict


from social_memory.constants import PATH_TO_AUGMENTED_DATA, SIQDatasetColumns
from social_memory.transforms import Transform
from social_memory.transforms.video import load_chunks

class SIQ2LongDataset:
    def __init__(
        self,
        split: str,
        group_by_video: bool = True,
        # transform: TransformList | None = None,
        num_frames_per_video: int = 32,
        max_chunks_per_video: int = 24,
    ):

        self.split = split
        self.transform = Transform(
            fn=load_chunks,
            num_frames=num_frames_per_video,
        )
        self.max_chunks_per_video = max_chunks_per_video

        self.data = pd.read_json(
            PATH_TO_AUGMENTED_DATA / f"{split}_contrastive.jsonl",
            lines=True,
        ).reset_index(drop=True)
        
        self.fields = [
            'qid',
            'vid_name',
            'question',
            'oracle_idx',
            # 'hard_negatives'
        ]
        def _is_real_mp4(path, min_bytes=10_000):
            """Quick sanity check: a real mp4 chunk is way bigger than 261 bytes."""
            try:
                return path.stat().st_size >= min_bytes
            except OSError:
                return False

        def _has_valid_chunks(vid_name):
            chunk_dir = Path(f"../chunks/{vid_name}")
            if not chunk_dir.exists():
                return True  # will fall back to GCS path
            chunks = list(chunk_dir.glob("*.mp4"))
            return (
                bool(chunks) and
                len(chunks) <= self.max_chunks_per_video and
                all(_is_real_mp4(c) for c in chunks)
            )

        valid_mask = self.data[SIQDatasetColumns.VIDEO_ID].apply(_has_valid_chunks)
        n_dropped = (~valid_mask).sum()
        print(f"Dropping {n_dropped} videos with corrupt / too many chunks")
        self.data = self.data[valid_mask].reset_index(drop=True)
        
        if group_by_video:
            self._group_data_by_video_id()
    
    def _group_data_by_video_id(self):
        self.data = self.data.groupby(SIQDatasetColumns.VIDEO_ID).agg(
            {
                "qid": list,
                "question": list,
                "oracle_idx": "first",  # should be same for all questions of same video
                "hard_negatives": "first",  # ^^
            }
        ).reset_index()

    def __getitem__(self, idx):
        t0 = time.perf_counter()
        x = self.data.iloc[idx][self.fields].to_dict()
        try:
            result = self.transform(x)
            elapsed = time.perf_counter() - t0
            n_chunks = len(result.get("chunks", {}))
            print(f"[worker {os.getpid()}] {x.get('vid_name')}: {n_chunks} chunks, {elapsed:.2f}s")
            return result
        except BaseException as e:  # catch even keyboard-interrupts / system exits within the worker
            print(f"Skipping idx={idx}: {type(e).__name__}: {e}", flush=True)
            return None

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    out = defaultdict(list)
    for item in batch:
        if item is None:
            continue
        for k, v in item.items():
            out[k].append(v)
    return dict(out)

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer(name, timings):
    """Accumulate elapsed time under `name` in the timings dict."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # GPU ops are async; sync for honest timing
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)


def compute_batch_loss(model, batch, temperature=0.07, timings=None):
    if timings is None:
        timings = {}

    losses = []
    n_chunks_total = 0
    n_questions_total = 0

    for v_idx in range(len(batch["vid_name"])):
        print(f"  [compute] v_idx={v_idx} vid={batch['vid_name'][v_idx]}", flush=True)
        chunks = sorted(batch["chunks"][v_idx].values(), key=lambda c: c["chunk_idx"])
        sorted_chunk_ids = [c["chunk_idx"] for c in chunks]
        n_chunks_total += len(chunks)

        try:
            with timer("encode_chunks", timings):
                chunk_embs = model.encode_chunks(
                    [c["frames"] for c in chunks],
                    [c["transcript"] for c in chunks],
                )
        except Exception as e:
            print(f"problem encoding {v_idx}: {e}")
            continue

        questions = batch["question"][v_idx]
        n_questions_total += len(questions)

        with timer("encode_questions", timings):
            question_embs = model.encode_questions(questions)

        with timer("loss_compute", timings):
            logits = (question_embs @ chunk_embs.T) / temperature
            oracle = batch["oracle_idx"][v_idx]
            oracle = sorted_chunk_ids.index(oracle)
            labels = torch.full(
                (len(questions),), oracle, dtype=torch.long, device=logits.device,
            )
            video_loss = F.cross_entropy(logits, labels)
            losses.append(video_loss)

    timings["n_chunks_total"] = n_chunks_total
    timings["n_questions_total"] = n_questions_total

    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_batch_loss_features(model, batch, temperature=0.07, timings=None):
    """Like `compute_batch_loss`, but each chunk carries precomputed
    `video_emb`/`transcript_emb` and the batch carries `q_emb` per video.
    Used by FeatureCachedDataset + XCLIPFeatureAdapter — no backbone in path.
    """
    if timings is None:
        timings = {}

    losses = []
    n_chunks_total = 0
    n_questions_total = 0

    for v_idx in range(len(batch["vid_name"])):
        chunks = sorted(batch["chunks"][v_idx].values(), key=lambda c: c["chunk_idx"])
        sorted_chunk_ids = [c["chunk_idx"] for c in chunks]
        n_chunks_total += len(chunks)

        with timer("encode_chunks", timings):
            chunk_embs = model.encode_chunks(
                [c["video_emb"] for c in chunks],
                [c["transcript_emb"] for c in chunks],
            )

        q_emb = batch["q_emb"][v_idx]
        n_questions_total += int(q_emb.shape[0])

        with timer("encode_questions", timings):
            question_embs = model.encode_questions(q_emb)

        with timer("loss_compute", timings):
            logits = (question_embs @ chunk_embs.T) / temperature
            oracle = batch["oracle_idx"][v_idx]
            try:
                oracle_pos = sorted_chunk_ids.index(oracle)
            except ValueError:
                # Oracle chunk missing from precomputed features — skip video.
                continue
            labels = torch.full(
                (question_embs.shape[0],),
                oracle_pos,
                dtype=torch.long,
                device=logits.device,
            )
            losses.append(F.cross_entropy(logits, labels))

    timings["n_chunks_total"] = n_chunks_total
    timings["n_questions_total"] = n_questions_total

    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_batch_loss_late_interaction(model, batch, temperature=0.07, timings=None):
    """InfoNCE over MaxSim scores for `LateInteractionAdapter`.

    Pooled XCLIP features are fed as 1-token sequences (B, 1, D); the Q-Former
    expands each chunk to K output tokens. Per video, score (Nq, Nc) via
    MaxSim, multiply by the adapter's learned `logit_scale`, cross-entropy
    against the oracle chunk index. `temperature` is ignored — the adapter
    owns its own learnable scale.
    """
    from social_memory.adapters.late_interaction import LateInteractionAdapter

    if timings is None:
        timings = {}

    device = next(model.parameters()).device
    losses = []
    n_chunks_total = 0
    n_questions_total = 0

    for v_idx in range(len(batch["vid_name"])):
        chunks = sorted(batch["chunks"][v_idx].values(), key=lambda c: c["chunk_idx"])
        sorted_chunk_ids = [c["chunk_idx"] for c in chunks]
        n_chunks_total += len(chunks)

        oracle = batch["oracle_idx"][v_idx]
        try:
            oracle_pos = sorted_chunk_ids.index(oracle)
        except ValueError:
            continue

        v = torch.from_numpy(
            np.stack([np.asarray(c["video_emb"], dtype=np.float32) for c in chunks])
        ).to(device).unsqueeze(1)  # (Nc, 1, D)
        t = torch.from_numpy(
            np.stack([np.asarray(c["transcript_emb"], dtype=np.float32) for c in chunks])
        ).to(device).unsqueeze(1)  # (Nc, 1, D)

        with timer("encode_chunks", timings):
            chunk_tokens = model.encode_chunk(v, t)  # (Nc, K, d_out)

        q_emb = batch["q_emb"][v_idx]
        if isinstance(q_emb, torch.Tensor):
            q = q_emb.to(device=device, dtype=torch.float32)
        else:
            q = torch.from_numpy(np.asarray(q_emb, dtype=np.float32)).to(device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q = q.unsqueeze(1)  # (Nq, 1, D)
        n_questions_total += int(q.shape[0])

        with timer("encode_questions", timings):
            question_tokens = model.encode_question(q)  # (Nq, 1, d_out)

        with timer("loss_compute", timings):
            sims = LateInteractionAdapter.maxsim_matrix(
                question_tokens, chunk_tokens
            )  # (Nq, Nc)
            logits = sims * model.temperature()
            labels = torch.full(
                (logits.shape[0],), oracle_pos, dtype=torch.long, device=logits.device,
            )
            losses.append(F.cross_entropy(logits, labels))

    timings["n_chunks_total"] = n_chunks_total
    timings["n_questions_total"] = n_questions_total

    if not losses:
        return None
    return torch.stack(losses).mean()


def save_checkpoint(state, ckpt_dir, name):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_dir / f"{name}.tmp"
    final_path = ckpt_dir / f"{name}.pt"
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


def train(
    model,
    dataloader,
    optimizer,
    num_epochs=3,
    temperature=0.07,
    ckpt_dir="checkpoints",
    ckpt_every=5,
    log_every=20,
    compute_loss_fn=None,
    log_callback=None,
    post_save_callback=None,
):
    if compute_loss_fn is None:
        compute_loss_fn = compute_batch_loss
    if log_callback is None:
        log_callback = lambda metrics: None  # noqa: E731
    model.train()
    step_losses = []
    best_epoch_loss = float("inf")
    global_step = 0
    keep_last_n = 5
    recent_ckpts: list[tuple[int, "Path"]] = []  # (epoch, path) sliding window

    def _save(state, name):
        path = save_checkpoint(state, ckpt_dir, name)
        if post_save_callback is not None:
            post_save_callback()
        return path

    # GPU info up front
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, running on CPU")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_losses = []

        # Time the wait for the first batch separately — that's dataloader startup
        t_step_end = time.perf_counter()

        for step, batch in enumerate(dataloader):
            t_dataload = time.perf_counter() - t_step_end
            verbose = step % log_every == 0
            if verbose:
                print(
                    f"[main] step {step} START at {time.strftime('%H:%M:%S')} | "
                    f"got {len(batch['vid_name'])} videos: {batch['vid_name']}",
                    flush=True,
                )

            if not batch.get("vid_name"):
                t_step_end = time.perf_counter()
                continue

            timings = {}

            with timer("forward", timings):
                loss = compute_loss_fn(model, batch, temperature, timings)

            if loss is None:
                t_step_end = time.perf_counter()
                continue

            with timer("backward", timings):
                optimizer.zero_grad()
                loss.backward()

            with timer("optimizer_step", timings):
                optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            step_losses.append({
                "global_step": global_step,
                "epoch": epoch,
                "step": step,
                "loss": loss_val,
            })
            log_callback({
                "train/loss": loss_val,
                "train/epoch": epoch,
                "train/step": step,
                "train/global_step": global_step,
                "perf/dataload_s": t_dataload,
                "perf/forward_s": timings.get("forward", 0.0),
                "perf/backward_s": timings.get("backward", 0.0),
                "perf/optimizer_s": timings.get("optimizer_step", 0.0),
                "perf/encode_chunks_s": timings.get("encode_chunks", 0.0),
                "perf/encode_questions_s": timings.get("encode_questions", 0.0),
                "perf/n_chunks": timings.get("n_chunks_total", 0),
                "perf/n_questions": timings.get("n_questions_total", 0),
            })

            if step % log_every == 0:
                total_step_time = t_dataload + timings["forward"] + timings["backward"] + timings["optimizer_step"]

                # compute GPU memory allocation
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    mem_str = f" | gpu_mem alloc={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB"
                else:
                    mem_str = ""

                print(
                    f"\nepoch {epoch} step {step} | loss {loss_val:.4f} | "
                    f"total {total_step_time:.2f}s"
                    f"{mem_str}"
                )
                print(
                    f"  dataload:        {t_dataload:.2f}s  "
                    f"({100 * t_dataload / total_step_time:.0f}%)"
                )
                print(
                    f"  forward:         {timings['forward']:.2f}s  "
                    f"({100 * timings['forward'] / total_step_time:.0f}%)"
                )
                print(
                    f"    encode_chunks:    {timings.get('encode_chunks', 0):.2f}s  "
                    f"({timings['n_chunks_total']} chunks, "
                    f"{timings.get('encode_chunks', 0) / max(timings['n_chunks_total'], 1):.3f}s/chunk)"
                )
                print(
                    f"    encode_questions: {timings.get('encode_questions', 0):.2f}s  "
                    f"({timings['n_questions_total']} questions)"
                )
                print(
                    f"    loss_compute:     {timings.get('loss_compute', 0):.3f}s"
                )
                print(
                    f"  backward:        {timings['backward']:.2f}s  "
                    f"({100 * timings['backward'] / total_step_time:.0f}%)"
                )
                print(
                    f"  optimizer_step:  {timings['optimizer_step']:.3f}s"
                )

            global_step += 1
            t_step_end = time.perf_counter()

        if not epoch_losses:
            print(f"=== Epoch {epoch + 1} had no valid batches, skipping checkpoint ===")
            continue

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"=== Epoch {epoch + 1} complete | avg loss {avg_loss:.4f} ===")
        log_callback({
            "epoch/avg_loss": avg_loss,
            "epoch/index": epoch,
            "epoch/n_steps": len(epoch_losses),
        })

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "temperature": temperature,
        }

        # Save every epoch, but only delete non-milestone snapshots that fall
        # out of the last-N window. Milestones (every `ckpt_every` epochs) are
        # preserved permanently.
        last_path = _save(state, f"epoch_{epoch:04d}")
        recent_ckpts.append((epoch, last_path))
        while len(recent_ckpts) > keep_last_n:
            old_epoch, old_path = recent_ckpts.pop(0)
            is_milestone = (old_epoch + 1) % ckpt_every == 0
            if not is_milestone:
                old_path.unlink(missing_ok=True)

        _save(state, "latest")

        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            _save(state, "best")
            print(f"new best avg loss: {avg_loss:.4f}")

    return step_losses


def train_features_run(
    features_root,
    ckpt_dir,
    *,
    adapter: str = "xclip",
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
    wandb_config: dict | None = None,
    post_save_callback=None,
):
    """Shared driver: feature-cached dataset → adapter → train loop.

    Used by both the local CLI and the Modal entrypoint so they execute the
    same code path. Hyperparameters and IO roots come in as args; nothing
    about Modal or the local filesystem layout leaks into here.

    Args:
        features_root: dir containing `chunks/<vid>.npz` and `text/<split>.npz`.
        ckpt_dir: where to write `epoch_XXXX.pt`, `latest.pt`, `best.pt`.
        wandb_config: optional dict with keys `api_key`, `project`, `entity`,
            `run_name`, `extra_config`. Pass None to disable wandb.
        post_save_callback: invoked after every checkpoint write — Modal
            passes `ckpt_vol.commit` so intermediate checkpoints become
            durable in the volume mid-run.
    """
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

    features_root = Path(features_root)
    if not (features_root / "chunks").exists():
        raise FileNotFoundError(
            f"no precomputed chunks at {features_root}/chunks — run precompute first"
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
    model = adapter_classes[adapter](
        feat_dim=dataset.dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{adapter} adapter on {device} | trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    log_callback = None
    wandb_run = None
    if wandb_config and wandb_config.get("api_key"):
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        import wandb

        wandb_run = wandb.init(
            project=wandb_config.get("project", "social-memory"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("run_name"),
            config={
                "adapter": adapter,
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
                "ckpt_every": ckpt_every,
                "dataset_size": len(dataset),
                **(wandb_config.get("extra_config") or {}),
            },
        )
        log_callback = wandb.log
    elif wandb_config:
        print("WARN: wandb_config given but no api_key — skipping wandb logging")

    try:
        train(
            model,
            dataloader,
            optimizer,
            num_epochs=epochs,
            temperature=temperature,
            ckpt_dir=str(ckpt_dir),
            ckpt_every=ckpt_every,
            compute_loss_fn=compute_batch_loss_features,
            log_callback=log_callback,
            post_save_callback=post_save_callback,
        )
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()


def train_late_interaction_run(
    features_root,
    ckpt_dir,
    *,
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
    wandb_config: dict | None = None,
    post_save_callback=None,
):
    """Shared driver for `LateInteractionAdapter` on cached XCLIP features.

    Mirrors `train_features_run` but builds a multi-vector adapter and uses
    the MaxSim-based loss. Pooled cached embeddings are fed as 1-token
    sequences — the Q-Former expands chunks to `num_chunk_tokens` views the
    question can MaxSim against.
    """
    from torch.utils.data import DataLoader

    from social_memory.adapters.late_interaction import LateInteractionAdapter
    from social_memory.feature_dataset import (
        FeatureCachedDataset,
        feature_collate_fn,
    )

    features_root = Path(features_root)
    if not (features_root / "chunks").exists():
        raise FileNotFoundError(
            f"no precomputed chunks at {features_root}/chunks — run precompute first"
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
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"late-interaction adapter on {device} | trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    log_callback = None
    wandb_run = None
    if wandb_config and wandb_config.get("api_key"):
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        import wandb

        wandb_run = wandb.init(
            project=wandb_config.get("project", "social-memory"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("run_name"),
            config={
                "adapter": "late_interaction",
                "split": split,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "init_temperature": init_temperature,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "num_chunk_tokens": num_chunk_tokens,
                "num_qformer_layers": num_qformer_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "feat_dim": dataset.dim,
                "n_trainable": n_trainable,
                "max_chunks_per_video": max_chunks_per_video,
                "ckpt_every": ckpt_every,
                "dataset_size": len(dataset),
                **(wandb_config.get("extra_config") or {}),
            },
        )
        log_callback = wandb.log
    elif wandb_config:
        print("WARN: wandb_config given but no api_key — skipping wandb logging")

    try:
        train(
            model,
            dataloader,
            optimizer,
            num_epochs=epochs,
            temperature=init_temperature,
            ckpt_dir=str(ckpt_dir),
            ckpt_every=ckpt_every,
            compute_loss_fn=compute_batch_loss_late_interaction,
            log_callback=log_callback,
            post_save_callback=post_save_callback,
        )
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()