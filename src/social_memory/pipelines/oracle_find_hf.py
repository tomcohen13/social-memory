
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
import os
import torch
import torch.nn as nn
import traceback

from pathlib import Path
from typing import List
from tqdm import tqdm
import pandas as pd

from social_memory.constants import SIQDatasetColumns
from social_memory.pipelines.base import BasePipeline
from social_memory.transforms import TransformList, apply_transform_with_concurrency
from social_memory.utils import check_device, load_dataset, group_inputs_by_video_id, read_vtt_file
from social_memory.constants import GCS_BUCKET, GCS_CHUNKS_PREFIX
from social_memory.gcs import list_blobs, download_to_temp
# from social_memory.encoders.xclip import sample_frames

from social_memory.encoders.hf import HFVideoTextEncoder


class OracleFindPipeline(BasePipeline):

    NAME = "oracle_find"

    def __init__(self, configs):
        super().__init__(configs)

        self.transforms = TransformList([
            self.load_chunks,
            self.encode_input,
            self.compute_embed_similarity,
            self.to_df_rows,
        ])
    
    def _load_model_runner(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        self.encoder: HFVideoTextEncoder = HFVideoTextEncoder.from_pretrained(self.configs.model)
        device = check_device()
        self.encoder.to(device)

    async def process_inputs(self, inputs: List[dict]) -> List[dict]:

        for transform in self.transforms:
            inputs = await apply_transform_with_concurrency(transform, inputs, max_concurrency=1)

        # to_df_rows expands each grouped input into a list of per-question dicts;
        # flatten so callers get a single list of rows.
        return [row for group in inputs for row in group]

    def load_chunks(self, input: dict) -> dict:
        """
        Loads frames and transcripts per chunk
        """

        from social_memory.transforms.video import load_chunks
        return load_chunks(input, num_frames=self.encoder.num_frames)

    def encode_input(self, input: dict) -> dict:
        """
        Encode frames, transcripts, and queries per (grouped) input
        """

        chunks = input["chunks"]
        ordered_chunk_ids = sorted(chunks.keys(), key=int)
        input["embeddings"] = {}

        with torch.no_grad():
            video_embeddings: torch.Tensor = self.encoder.encode_video(
                frames=[chunks[chunk_id].pop("frames") for chunk_id in ordered_chunk_ids],
            ).detach().cpu()  # (n_chunks, embed_size)

            input["embeddings"]["video_embeddings"] = video_embeddings
            
            text_embeddings: torch.Tensor = self.encoder.encode_text(
                texts=[chunks[chunk_id]["transcript"] for chunk_id in ordered_chunk_ids],
            ).detach().cpu()  # (n_chunks, embed_size)

            input["embeddings"]["text_embeddings"] = text_embeddings

            query_embeds: torch.Tensor = self.encoder.encode_text(input["questions"]).detach().cpu() # (n_questions, embed_size)
            input["embeddings"]["query_embeddings"] = query_embeds
        return input
    
    def compute_embed_similarity(self, input: dict) -> dict:
        """
        Compute cosine similarity between query embedding and modality embeddings
        """

        embeddings = input.pop("embeddings")
        E_vid = embeddings["video_embeddings"]
        E_text = embeddings["text_embeddings"]
        E_query = embeddings["query_embeddings"]
        
        video_sim = E_query @ E_vid.T  # (n_questions, n_chunks)
        text_sim = E_query @ E_text.T  # (n_questions, n_chunks)

        input["similarity_scores"] = {
            "video": video_sim,
            "text": text_sim,
        }
        return input
    
    def to_df_rows(self, input: dict) -> List[dict]:
        """
        Converts grouped inputs back to question-based rows
        """
        _, top_chunks_video = torch.topk(
            input["similarity_scores"]["video"],
            k=min(3, input["num_chunks"]),
            sorted=True,
        )
        _, top_chunks_text = torch.topk(
            input["similarity_scores"]["text"],
            k=min(3, input["num_chunks"]),
            sorted=True,
        )

        top_chunks_video = top_chunks_video.tolist()  # (n_questions, 3)
        top_chunks_text = top_chunks_text.tolist()  # (n_questions, 3)
        
        return [
            {
                SIQDatasetColumns.VIDEO_ID: input[SIQDatasetColumns.VIDEO_ID],
                "qid": input["qid"][i],
                "question": question,
                "oracle_idx": input["oracle_idx"],
                "most_similar_video": top_chunks_video[i][0],
                "most_similar_text": top_chunks_text[i][0],
                "top_chunks_video": top_chunks_video[i],
                "top_chunks_text": top_chunks_text[i],
            }
            for i, question in enumerate(input["questions"])
        ]
    
    async def run(self) -> None:
        """
        Run oracle-find pipeline given configs
        """

        self.logger.info("Starting pipeline...")
        self.logger.info(self.__repr__())
        start_time = datetime.now()

        self.logger.info("loading dataset...")
        dataset = load_dataset(dataset_name=self.configs.dataset, split=self.configs.split)

        if os.path.exists(self.path_to_output):
            results = pd.read_json(self.path_to_output, lines=True)
            results_id = results["qid"].unique()
            dataset = dataset[~dataset["qid"].isin(results_id)].reset_index(drop=True)

        inputs = dataset.to_dict(orient='records')
        grouped_inputs = group_inputs_by_video_id(inputs)

        for video in tqdm(grouped_inputs, desc="videos"):
            try:
                batch_results = await self.process_inputs([video])
                self.write_results_to_json(batch_results)
            except Exception as e:
                self.logger.error(f"Failed on video {video.get(SIQDatasetColumns.VIDEO_ID)!r}: {traceback.format_exc()}")

        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Finished processing {len(dataset)} documents in {elapsed_time:.1f}s.")


# CONVERT THIS UGLY-ASS THING INTO PIPELINE
def evaluate_encoder(
    inputs: List[dict],
    encoder: nn.Module,
    output_path: str
) -> List[dict]:
    f"""
    Evaluate zero-shot oracle-finding performance of dual encoder.
    """

    if os.path.exists(output_path):
        results_df = pd.read_csv(output_path).drop_duplicates()
    else:
        results_df = pd.DataFrame()
    
    device = check_device()
    print(f"loading encoder to device {device}...")
    encoder.to(device)

    for inp in tqdm(inputs, desc="videos"):
        try: 
            vid_id = inp[SIQDatasetColumns.VIDEO_ID]
            questions = inp.get("questions", [])
            if not questions:
                print(f"[ERROR] no question for {vid_id!r}, skipping")
                continue
            print(f"found {len(questions)} questions for video")
            
            chunks = defaultdict(dict)

            print("Downloading frames, transcripts")
            for blob in list_blobs(GCS_BUCKET, GCS_CHUNKS_PREFIX + f"/{vid_id}/"):
                path = Path(blob.name)
                chunk_idx = int(path.stem)
                ext = path.suffix
                if ext == ".mp4":
                    with download_to_temp(GCS_BUCKET, blob.name) as tmp_chunk_path:
                        frames = sample_frames(tmp_chunk_path, num_frames=encoder.num_frames)
                    chunks[chunk_idx]["frames"] = frames
                elif ext == ".vtt":
                    chunks[chunk_idx]["transcript"] = blob.download_as_text()

            print(f"Downloaded {len(chunks)}/{len(inp["chunk_ids"])} chunks for video")
            assert len(chunks) == len(inp["chunk_ids"])
            assert all([chunks[k]["frames"] and chunks[k]["transcript"] for k in chunks])
            
            ordered_chunk_ids = sorted(chunks.keys(), key=int)

            with torch.no_grad():
                video_embeddings: torch.Tensor = encoder.encode_video(
                    frames=[chunks[chunk_id]["frames"] for chunk_id in ordered_chunk_ids],
                ).detach().cpu()  # (n_chunks, embed_size)
                
                text_embeddings: torch.Tensor = encoder.encode_text(
                    texts=[chunks[chunk_id]["transcript"] for chunk_id in ordered_chunk_ids],
                ).detach().cpu()  # (n_chunks, embed_size)

                query_embeds: torch.Tensor = encoder.encode_text(questions).detach().cpu() # (n_questions, embed_size)
            
            video_sim = query_embeds @ video_embeddings.T  # (n_questions, n_chunks)
            text_sim = query_embeds @ text_embeddings.T

            oracle_vec = inp["oracle_idx"] * torch.ones(len(questions))

            _, top_chunks_video = torch.topk(video_sim, k=min(3, len(chunks)), dim=1, largest=True, sorted=True)
            _, top_chunks_text = torch.topk(text_sim, k=min(3, len(chunks)), dim=1, largest=True, sorted=True)

            print(f"Video-only accuracy @ 1: {(top_chunks_video[:,0] == oracle_vec).float().mean()} | random chance: {1/len(chunks)}")
            print(f"Text-only accuracy @ 1:{(top_chunks_text[:,0] == oracle_vec).float().mean()}  | random chance: {1/len(chunks)}")

            top_chunks_video = top_chunks_video.tolist()  # (n_questions, 3)
            top_chunks_text = top_chunks_text.tolist()  # (n_questions, 3)
            
            new_rows = [
                {
                    SIQDatasetColumns.VIDEO_ID: inp[SIQDatasetColumns.VIDEO_ID],
                    "qid": inp["qid"][i],
                    "question": question,
                    "oracle_idx": inp["oracle_idx"],
                    "most_similar_video": top_chunks_video[i][0],
                    "most_similar_text": top_chunks_text[i][0],
                    "top_chunks_video": top_chunks_video[i],
                    "top_chunks_text": top_chunks_text[i],
                }
                for i, question in enumerate(questions)
            ]
            results_df = pd.concat(
                [results_df, pd.DataFrame(new_rows)], ignore_index=True
            )
            results_df.to_csv(output_path)

        except Exception as e:
            print(f"there was a problem with input: {inp["vid_name"]}, error: {e}")
            continue
        print()


    return inputs
