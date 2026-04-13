"""Script to divide full videos into chunks while storing the ground truth trim index for each video."""

import json 
import pandas as pd
from moviepy.editor import VideoFileClip
from pathlib import Path

BUFFER_TIME = 10 # seconds
TEST_SET_DIR = Path("datasets/socialiq2/siq2/validation_augmented")

qa = pd.read_json(TEST_SET_DIR / "qa_augmented.json", lines=True)
with open(TEST_SET_DIR.parent / "trims.json", "r") as j:
    trims = json.load(j)

qa["trim_start"] = qa["vid_name"].map(trims)
qa["trim_end"] = qa["trim_start"] + 60
qa["video_duration"] = qa["vid_name"].apply(lambda vid: VideoFileClip(f"{TEST_SET_DIR}/video/{vid}.mp4").duration)
qa[["vid_name", "trim_start", "trim_end", "video_duration"]].sample(5)

all_chunks = {}

for i, row in qa.drop_duplicates(subset=["vid_name"]).iterrows():

    all_chunks[row["vid_name"]] = {}
    chunks = []
    full_duration = float(row["video_duration"])
    gt_start, gt_end = gt = row[["trim_start", "trim_end"]].tolist()
    print(f"Processing video: {row['vid_name']} | gt: {gt} | full_duration: {full_duration}")
    chunk_start, chunk_end = gt_start - 60, gt_start
    while chunk_start >= 0:
        chunks.append([chunk_start, chunk_end])
        chunk_end = chunk_start
        chunk_start -= 60
    # two options: 
    #   1. within buffer space, extend to 0 and add
    if (0 <= chunk_end <= BUFFER_TIME):
        if chunks:
            chunks[-1][0] = 0
    else:
        chunks.append([0, chunk_end])
    
    chunks = chunks[::-1]
    chunks.append(gt)
    gt_idx = len(chunks) - 1

    chunk_start, chunk_end = gt_end, gt_end + 60
    while chunk_end <= full_duration:
        chunks.append([chunk_start, chunk_end])
        chunk_start = chunk_end
        chunk_end += 60
    
    if chunk_end <= full_duration:
        # extend last chunk
        chunks[-1][1] = full_duration
    else:
        chunks.append([chunk_start, full_duration])
    all_chunks[row["vid_name"]]["chunks"] = chunks
    all_chunks[row["vid_name"]]["gt_idx"] = gt_idx

with open(f"{TEST_SET_DIR}/video_chunks.json", "w") as j:
    json.dump(all_chunks, j, indent=2)

