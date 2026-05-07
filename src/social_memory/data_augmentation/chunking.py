"""Chunking and segmentation functions"""
from typing import List, Tuple

def compute_chunks_around_oracle(
    oracle: List[float] | Tuple[float, float],
    full_duration: float,
    chunk_size: int = 60,
    buffer_size: int = 10,
):
    """
    Computes time ranges that a video should be segmented into around the oracle.

    Args:
        oracle: a list of start time and end time of the oracle
        full_duration: full duration (in seconds) of the video
        chunk_size: desired time range for each chunk
        buffer_size: if the outermost chunks end up being within buffer_size seconds of start or end of video, 
            absorb that into chunk
    
    Returns:
        chunks (list[list[float]])
        oracle_idx (int)
    
    Example:
        >> TODO
    """

    chunks = []
    oracle_start, oracle_end = oracle
    chunk_start, chunk_end = oracle_start - chunk_size, oracle_start
    while chunk_start >= 0:
        chunks.append([chunk_start, chunk_end])
        chunk_end = chunk_start
        chunk_start -= chunk_size
    # If the leftmost window starts within BUFFER_TIME of 0, absorb the sliver into the first
    # chunk by extending its start to 0 (instead of adding a tiny [0, chunk_end] segment).
    # After the loop, chunk_start is < 0 and chunk_end is the start time of that leftmost window.
    if 0 <= chunk_end <= buffer_size:
        if chunks:
            chunks[-1][0] = 0
    else:
        chunks.append([0, chunk_end])
    
    chunks = chunks[::-1]
    chunks.append(oracle)
    oracle_idx = len(chunks) - 1

    chunk_start, chunk_end = oracle_end, oracle_end + chunk_size
    while chunk_end <= full_duration:
        chunks.append([chunk_start, chunk_end])
        chunk_start = chunk_end
        chunk_end += chunk_size

    # Merge a short tail into the last *post–ground-truth* chunk only. If the forward while never
    # ran, chunks[-1] is still gt — do not extend that or the QA window would change.
    if len(chunks) > oracle_idx + 1 and full_duration - chunk_start <= buffer_size:
        chunks[-1][1] = full_duration
    elif chunk_start < full_duration:
        chunks.append([chunk_start, full_duration])
    
    return chunks, oracle_idx
