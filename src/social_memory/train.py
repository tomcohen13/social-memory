
import time, os
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
    keep_last_n=3,
    log_every=1,  # log every step while diagnosing; bump back up later
):
    model.train()
    step_losses = []
    best_epoch_loss = float("inf")
    global_step = 0
    recent_ckpts = []

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
            print(f"[main] step {step} START at {time.strftime('%H:%M:%S')} | got {len(batch['vid_name'])} videos: {batch['vid_name']}", flush=True)
    

            if not batch.get("vid_name"):
                t_step_end = time.perf_counter()
                continue

            timings = {}

            print(f"[main] step {step} calling compute_batch_loss", flush=True)
            with timer("forward", timings):
                loss = compute_batch_loss(model, batch, temperature, timings)

            if loss is None:
                t_step_end = time.perf_counter()
                continue
            print(f"[main] step {step} compute_batch_loss done in {time.perf_counter() - t_dataload:.2f}s, loss={loss}", flush=True)

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

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "temperature": temperature,
        }

        last_path = save_checkpoint(state, ckpt_dir, f"epoch_{epoch:04d}")
        recent_ckpts.append(last_path)
        while len(recent_ckpts) > keep_last_n:
            old = recent_ckpts.pop(0)
            old.unlink(missing_ok=True)

        save_checkpoint(state, ckpt_dir, "latest")

        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            save_checkpoint(state, ckpt_dir, "best")
            print(f"new best avg loss: {avg_loss:.4f}")

    return step_losses