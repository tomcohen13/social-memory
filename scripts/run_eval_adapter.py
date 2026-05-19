import sys
import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_src := _REPO_ROOT / "src").is_dir():
    sys.path.insert(0, str(_src))

import torch
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from social_memory.adapters.xclipadapter import XCLIPAdapter
from social_memory.train import SIQ2LongDataset, collate_fn
from social_memory.utils import check_device


def evaluate(model, dataloader):
    model.eval()
    results = []
    n_by_k = {1: 0, 3: 0, 5: 0}
    n_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if not batch.get("vid_name"):
                continue

            for v_idx in range(len(batch["vid_name"])):
                vid_name = batch["vid_name"][v_idx]
                chunks = sorted(batch["chunks"][v_idx].values(), key=lambda c: c["chunk_idx"])
                sorted_chunk_ids = [c["chunk_idx"] for c in chunks]
                oracle_idx = batch["oracle_idx"][v_idx]

                try:
                    oracle_pos = sorted_chunk_ids.index(oracle_idx)
                except ValueError:
                    print(f"[WARN] oracle_idx {oracle_idx} not in chunks for {vid_name}, skipping")
                    continue

                try:
                    chunk_embs = model.encode_chunks(
                        [c["frames"] for c in chunks],
                        [c["transcript"] for c in chunks],
                    )
                except Exception as e:
                    print(f"[WARN] chunk encoding failed for {vid_name}: {e}")
                    continue

                questions = batch["question"][v_idx]
                question_embs = model.encode_questions(questions)

                # (n_questions, n_chunks) — both normalized, dot product = cosine sim
                sims = question_embs @ chunk_embs.T
                n_chunks = sims.shape[1]

                for qid, sim_row in zip(batch["qid"][v_idx], sims):
                    ranked = sim_row.argsort(descending=True).tolist()
                    for k in n_by_k:
                        if oracle_pos in ranked[: min(k, n_chunks)]:
                            n_by_k[k] += 1
                    n_total += 1
                    results.append({
                        "qid": qid,
                        "vid_name": vid_name,
                        "oracle_idx": oracle_idx,
                        "predicted_chunk_idx": sorted_chunk_ids[ranked[0]],
                        "correct": ranked[0] == oracle_pos,
                    })

    return results, n_by_k, n_total


def main():
    parser = argparse.ArgumentParser(description="Evaluate XCLIPAdapter on oracle-finding (test split)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to .pt checkpoint saved by train.py")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional CSV path to write per-question results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_chunks", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=256)
    args = parser.parse_args()

    device = check_device()
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = XCLIPAdapter(hidden_dim=args.hidden_dim, output_dim=args.output_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Checkpoint: epoch={ckpt.get('epoch', '?')}, avg_loss={ckpt.get('avg_loss', float('nan')):.4f}")

    print(f"Loading '{args.split}' dataset...")
    dataset = SIQ2LongDataset(
        split=args.split,
        group_by_video=True,
        num_frames_per_video=args.num_frames,
        max_chunks_per_video=args.max_chunks,
    )
    print(f"Dataset: {len(dataset)} videos")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    results, n_by_k, n_total = evaluate(model, dataloader)

    print(f"\n=== Results ({n_total} questions) ===")
    for k, n_correct in n_by_k.items():
        acc = n_correct / n_total if n_total > 0 else float("nan")
        print(f"  Top-{k}: {acc:.4f}  ({n_correct}/{n_total})")

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
