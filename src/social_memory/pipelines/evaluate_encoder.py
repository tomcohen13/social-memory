
import os
import torch

from pathlib import Path
from typing import List
from tqdm import tqdm

from social_memory.constants import SIQDatasetColumns
# from social_memory.pipelines.base import BasePipeline
from social_memory.utils import check_device, read_vtt_file
from social_memory.constants import GCS_BUCKET, GCS_CHUNKS_PREFIX
from social_memory.gcs import list_blobs, get_blob_from_path, download_to_temp
from social_memory.encoder import XCLIPEncoder, sample_frames
    
# CONVERT THIS UGLY-ASS THING INTO PIPELINE
def evaluate_encoder(inputs: List[dict], encoder, output_path: str) -> List[dict]:
    """
    Evaluate zero-shot oracle-finding performance of the X-CLIP dual encoder.

    For each input, fetches all 1-minute video chunks and their VTT transcripts
    from GCS, encodes each chunk as a fused video+text embedding, and ranks them
    by cosine similarity against the encoded question. The chunk with the highest
    similarity is predicted as the one containing the answer.

    Args:
        inputs: List of dicts, each with at least:
            - ``video_id`` (str): ID of the source video.
            - ``question`` (str): Natural-language question to answer.
            - ``oracle`` (tuple[float, float], optional): Ground-truth answer
              window ``(start_sec, end_sec)`` used for qualitative comparison.

    Returns:
        The same list of dicts, each augmented with:
            - ``embeddings`` (dict[str, Tensor]): Fused embedding per chunk, keyed
              by chunk stem (e.g. ``"002"``), shape ``[1, 512]``.
            - ``similarities`` (Tensor): Cosine similarity of each chunk against
              the query, shape ``[C, 1]``.
            - ``most_similar`` (Tensor): Index of the highest-scoring chunk.

    * TODO:
        1. Currently X-CLIP is hardcoded in. should be encoder-agnostic.
        2. Convert into a BasePipeline
        3. Recycling chunks across questions from same video_id.
    """
    import pandas as pd

    if os.path.exists(output_path):
        results_df = pd.read_csv(output_path).drop_duplicates()
    else:
        results_df = pd.DataFrame()

    video_ids = set([inp[SIQDatasetColumns.VIDEO_ID] for inp in inputs])
    video_to_chunks = {vid_id: {"transcripts": [], "chunks": []} for vid_id in video_ids}

    print(f"fetching chunks from GCS for {len(video_ids)} video(s)...")
    for vid_id in video_ids:
        for blob in list_blobs(GCS_BUCKET, GCS_CHUNKS_PREFIX + f"/{vid_id}/"):
            path = blob.name
            ext = Path(path).suffix
            if ext == ".mp4":
                video_to_chunks[vid_id]["chunks"].append(path)
            elif ext == ".vtt":
                video_to_chunks[vid_id]["transcripts"].append(path)
        n_chunks = len(video_to_chunks[vid_id]["chunks"])
        n_trans = len(video_to_chunks[vid_id]["transcripts"])
        print(f"  {vid_id}: {n_chunks} chunks, {n_trans} transcripts")
        if n_chunks != n_trans:
            print(f"  [WARNING] count mismatch for {vid_id} — zip will truncate to {min(n_chunks, n_trans)}")

    device = check_device()
    print(f"loading encoder to device {device}...")
    encoder.to(device)

    for inp in inputs:
        try: 
            vid_id = inp[SIQDatasetColumns.VIDEO_ID]
            question = inp.get(SIQDatasetColumns.QUESTION, "")
            chunks = sorted(video_to_chunks[vid_id]["chunks"])
            transcripts = sorted(video_to_chunks[vid_id]["transcripts"])

            if not chunks:
                print(f"[ERROR] no chunks for {vid_id!r}, skipping")
                continue

            print(f"\n{vid_id} | {question!r}")
            inp["embeddings"] = {}
            for path_to_chunk, path_to_transcript in tqdm(zip(chunks, transcripts), total=min(len(chunks), len(transcripts)), desc=vid_id):
                chunk_idx = Path(path_to_chunk).stem

                with download_to_temp(GCS_BUCKET, path_to_transcript) as tmp_transcript_path:
                    transcript = read_vtt_file(tmp_transcript_path)
                if not transcript.strip():
                    print(f"  [WARNING] empty transcript for chunk {chunk_idx}")
                    continue

                print(f"Sample {encoder.num_frames} frames.")
                with download_to_temp(GCS_BUCKET, path_to_chunk) as tmp_chunk_path:
                    frames = sample_frames(tmp_chunk_path, num_frames=encoder.num_frames)

                if not frames:
                    print(f"  [WARNING] no frames sampled for chunk {chunk_idx}")
                    continue

                with torch.no_grad():
                    embeddings = encoder(frames, transcript)
                inp["embeddings"][chunk_idx] = embeddings["fused_embeddings"].detach().cpu()

            query_embed = encoder.encode_text(question).detach().cpu()
            stacked = torch.vstack(list(inp["embeddings"].values()))
            inp["similarities"] = stacked @ query_embed.T
            inp["most_similar"] = torch.argmax(inp["similarities"]).item()

            print(f"  most similar: {inp['most_similar']} ", end="")
            if "oracle_idx" in inp:
                print(f" | oracle: {inp['oracle_idx']}", end="")
            
            new_row = {k: v for k, v in inp.items() if k != "embeddings"}
            if len(results_df) == 0:
                # fill in columns as well
                results_df = pd.concat([results_df, pd.DataFrame([new_row])])
            else:
                results_df.loc[len(results_df)] = new_row
            results_df.to_csv(output_path)

        except Exception as e:
            print(f"there was a problem with input: {inp["qid"]}, error: {e}")
            continue
        print()


    return inputs
