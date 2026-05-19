import sys
import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_src := _REPO_ROOT / "src").is_dir():
    sys.path.insert(0, str(_src))

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

load_dotenv()

from social_memory.adapters.xclipadapter import XCLIPAdapter
from social_memory.train import SIQ2LongDataset, collate_fn, train
from social_memory.utils import check_device


def main():
    parser = argparse.ArgumentParser(description="Train XCLIPAdapter on the oracle-finding task")

    # Data
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_chunks", type=int, default=24)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=256)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # DataLoader
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # Checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--keep_last_n", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from")

    # Logging
    parser.add_argument("--log_every", type=int, default=1)

    args = parser.parse_args()

    device = check_device()
    print(f"Device: {device}")

    model = XCLIPAdapter(hidden_dim=args.hidden_dim, output_dim=args.output_dim)
    model.to(device)

    # Only the adapter parameters require gradients; backbone is frozen inside XCLIPAdapter.
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}, avg_loss={ckpt.get('avg_loss', float('nan')):.4f}")

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
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        ckpt_dir=args.ckpt_dir,
        keep_last_n=args.keep_last_n,
        log_every=args.log_every,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()
