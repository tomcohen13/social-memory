
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict


from social_memory.constants import PATH_TO_AUGMENTED_DATA, SIQDatasetColumns
from social_memory.transforms import Transform
from social_memory.transforms.video import load_chunks
from social_memory.utils import check_device

class SIQ2LongDataset:
    def __init__(
        self,
        split: str,
        group_by_video: bool = True,
        # transform: TransformList | None = None,
        num_frames_per_video: int = 32,
    ):

        self.split = split
        self.transform = Transform(
            fn=load_chunks,
            num_frames=num_frames_per_video,
        )

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
        
        if group_by_video:
            self._group_data_by_video_id()
    
    def _group_data_by_video_id(self):
        self.data = self.data.groupby(SIQDatasetColumns.VIDEO_ID).agg(
            {
                SIQDatasetColumns.VIDEO_ID: "first",
                "qid": list,
                "question": list,
                "oracle_idx": "first",  # should be same for all questions of same video
                "hard_negatives": "first",  # ^^
            }
        )

    def __getitem__(self, idx):

        x = self.data.iloc[idx][self.fields].to_dict()
        try:
            return self.transform(x)
        except Exception as e:
            print(f"Skipping {x.get('vid_name')}: {e}")
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


def compute_batch_loss(model, batch, temperature=0.07):
    """
    For each video in the batch:
      - encode its chunks once -> (n_chunks, D)
      - encode each of its questions -> (n_q, D)
      - logits[q, c] = sim(question_q, chunk_c) / temperature
      - cross-entropy against the oracle index
    Average across all questions in the batch.
    """
    losses = []

    for v_idx in range(len(batch["vid_name"])):
        # Encode this video's chunks
        chunks = sorted(batch["chunks"][v_idx].values(), key=lambda c: c["chunk_idx"])
        sorted_chunk_ids = [c["chunk_idx"] for c in chunks]
        chunk_embs = model.encode_chunks(
            [c["frames"] for c in chunks],
            [c["transcript"] for c in chunks],
        )                                            # (n_chunks, D), L2-normalized

        # Encode this video's questions
        questions = batch["question"][v_idx]
        question_embs = model.encode_questions(questions)   # (n_q, D), L2-normalized

        print(f"  video={batch['vid_name'][v_idx]} | chunks={len(chunks)} | questions={len(questions)} | oracle_chunk={batch['oracle_idx'][v_idx]}")

        # Logits: each question scored against every chunk in this video
        logits = (question_embs @ chunk_embs.T) / temperature  # (n_q, n_chunks)

        # All questions for this video share the same oracle
        oracle = batch["oracle_idx"][v_idx]
        oracle = sorted_chunk_ids.index(oracle)
        labels = torch.full(
            (len(questions),), oracle, dtype=torch.long, device=logits.device,
        )

        video_loss = F.cross_entropy(logits, labels)
        print(f"  video={batch['vid_name'][v_idx]} | loss={video_loss.item():.4f}")
        losses.append(video_loss)

    return torch.stack(losses).mean()

def train(model, dataloader, optimizer, num_epochs=3, temperature=0.07):
    model.train()
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_losses = []
        for step, batch in enumerate(dataloader):
            if not batch.get("vid_name"):
                continue  # whole batch failed to load

            loss = compute_batch_loss(model, batch, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f}")

        print(f"=== Epoch {epoch + 1} complete | avg loss {sum(epoch_losses) / len(epoch_losses):.4f} ===")