"""Video transforms"""
import av
import av.logging
av.logging.set_level(av.logging.ERROR)

import base64
import glob
import io
import numpy as np
import subprocess
import os

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple

from social_memory.constants import (
    GCS_BUCKET,
    GCS_CHUNKS_PREFIX,
    GCS_PREFIX,
    PATH_TO_DATA,
    SIQDatasetColumns,
    DirPaths,
)
from social_memory.gcs import download_to_memory, download_to_temp, list_blobs, video_blob_name
from social_memory.utils import get_duration, read_vtt_buffer, read_vtt_file


def _compute_clip_window(
    video_id: str,
    oracle: Tuple[int, int] | List[int],
    output_length: int,
    duration: float,
) -> Tuple[int, int]:
    """
    Return a (start, end) window of `output_length` seconds centered around the oracle segment.

    Expands symmetrically from the oracle boundaries first; if one side hits the video edge,
    the remaining expansion is applied to the other side. Returns (0, duration) for videos
    shorter than `output_length`.
    """
    if duration < output_length:
        print(f"Skipping {video_id} due to short duration: {duration} seconds < {output_length} seconds")
        return 0, duration

    print(f"Processing {video_id} with duration {duration:.2f} seconds and oracle {oracle}")

    start, end = oracle
    target_expansion = (output_length - (end - start)) / 2

    expand_left = min(target_expansion, start)
    expand_right = min(target_expansion, duration - end)

    start -= expand_left
    end += expand_right

    current_length = end - start
    if current_length < output_length:
        remaining = output_length - current_length
        if start > 0:
            expand_left_more = min(remaining, start)
            start -= expand_left_more
            remaining -= expand_left_more
        if remaining > 0:
            expand_right_more = min(remaining, duration - end)
            end += expand_right_more

    return int(start), int(end)


def clip_around_oracle(input: dict, output_length: int) -> dict:
    """
    Clip a video to desired length around oracle.

    Args:
        input: dict with keys "video_id", "oracle", and "duration".
        output_length: desired output length, in seconds.
    Returns:
        input dict with "video_raw" replaced by the clipped video bytes.
    
    Note: this transform assumes the full video bytes are already loaded in input["video_raw"].
    """

    raw: bytes | None = input.get(SIQDatasetColumns.VIDEO_RAW)
    if raw is None:
        return input

    start, end = _compute_clip_window(
        video_id=input[SIQDatasetColumns.VIDEO_ID],
        oracle=input["oracle"],
        output_length=output_length,
        duration=input["duration"]
    )
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", "pipe:0",
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            "-movflags", "+frag_keyframe+empty_moov",
            "-f", "mp4",
            "pipe:1",
        ],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    input[SIQDatasetColumns.VIDEO_RAW] = result.stdout
    return input


def load_video_from_local(
    input: dict,
    directory: Path = PATH_TO_DATA / DirPaths.VIDEO,
    with_audio: bool = True,
) -> dict:
    """
    Load a video file as raw bytes into input["video_raw"].

    Stores None if the file is missing. Downstream transforms operate on
    input["video_raw"]; call encode_video() as the final step to produce input["video"].

    Args:
        input: dict with at least SocialIQDatasetColumns.VIDEO_ID ("vid_name").
        directory: directory containing .mp4 files.
        with_audio: if False, audio is stripped via ffmpeg before storing.
    """
    video_id = input[SIQDatasetColumns.VIDEO_ID]
    path = directory / f"{video_id}.mp4"

    if not path.is_file():
        input[SIQDatasetColumns.VIDEO_RAW] = None
        return input

    input["duration"] = get_duration(path)

    if with_audio:
        with open(path, "rb") as f:
            input[SIQDatasetColumns.VIDEO_RAW] = f.read()
        return input

    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(path),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-movflags", "+frag_keyframe+empty_moov",
            "-f", "mp4",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    input[SIQDatasetColumns.VIDEO_RAW] = result.stdout
    return input


def encode_video(input: dict) -> dict:
    """Convert input["video_raw"] bytes to a base64 string in input[SIQDatasetColumns.VIDEO]."""
    raw: bytes | None = input.pop(SIQDatasetColumns.VIDEO_RAW, None)
    input[SIQDatasetColumns.VIDEO] = base64.b64encode(raw).decode("utf-8") if raw else None
    return input


def _load_video_from_gcs(
    video_id: str,
    bucket_name: str = GCS_BUCKET,
    prefix: str = GCS_PREFIX,
    with_audio: bool = False,
):
    """"""
    with download_to_temp(bucket_name, video_blob_name(video_id, prefix)) as tmp_path:
        if tmp_path is None:
            return {
                SIQDatasetColumns.VIDEO_RAW: None,
                "duration": None,
            }

        duration = get_duration(tmp_path)

        ffmpeg_cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(tmp_path),
            "-movflags", "+frag_keyframe+empty_moov",
            "-f", "mp4",
        ]
        if not with_audio:
            ffmpeg_cmd += ["-map", "0:v:0", "-an"]
        ffmpeg_cmd += ["-c", "copy", "pipe:1"]

        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    return {
        SIQDatasetColumns.VIDEO_RAW: result.stdout,
        "duration": duration,
    }


def load_chunks(input: dict, num_frames: int) -> dict:
    """
    Loads frames and transcripts per chunk
    """

    chunks = defaultdict(dict)
    video_id = input.get(SIQDatasetColumns.VIDEO_ID.value)

    # When running inside a DataLoader worker, processes already provide video-level parallelism.
    # Each worker only loads one video at a time, so a small thread pool suffices.
    # In the main process (num_workers=0) we want more threads since there's no process parallelism.
    from torch.utils.data import get_worker_info
    in_worker = get_worker_info() is not None
    inner_workers = 2 if in_worker else 8

    local_dir = f"../chunks/{video_id}/"
    if os.path.exists(local_dir):
        paths = glob.glob(f"{local_dir}*")
        with ThreadPoolExecutor(max_workers=min(len(paths), inner_workers)) as ex:
            futures = [ex.submit(partial(process_file, num_frames=num_frames), p) for p in paths]
            for f in as_completed(futures):
                chunk_idx, result = f.result()
                chunks[chunk_idx].update(result)
        input["chunks"] = chunks
        return input

    print("loading from GCS...")
    blobs = list(list_blobs(GCS_BUCKET, GCS_CHUNKS_PREFIX + f"/{video_id}/"))
    with ThreadPoolExecutor(max_workers=inner_workers) as ex:
        futures = [ex.submit(partial(process_blob, num_frames=num_frames), b) for b in blobs]
        for f in as_completed(futures):
            result = f.result()
            if result:
                chunk_idx, key, value = result
                chunks[chunk_idx][key] = value
                chunks[chunk_idx]["chunk_idx"] = chunk_idx

    # assert len(chunks) == input["num_chunks"]
    input["chunks"] = chunks
    return input


def process_file(path_to_file, num_frames):
    path = Path(path_to_file)
    ext = path.suffix
    chunk_idx = int(path.stem)
    result = {"chunk_idx": chunk_idx}
    if ext == ".mp4":
        result["frames"] = sample_frames(path_to_file, num_frames=num_frames)
    elif ext == ".vtt":
        result["transcript"] = read_vtt_file(path_to_file)
    return chunk_idx, result


def process_blob(blob, num_frames: int):
    path = Path(blob.name)
    chunk_idx = int(path.stem)
    ext = path.suffix

    data = download_to_memory(GCS_BUCKET, blob.name)
    if data is None:
        return None

    if ext == ".mp4":
        frames = sample_frames(io.BytesIO(data), num_frames=num_frames)
        return chunk_idx, "frames", frames
    elif ext == ".vtt":
        transcript = read_vtt_buffer(io.StringIO(data.decode("utf-8")))
        return chunk_idx, "transcript", transcript
    return None


def sample_frames(video_source: Path | io.BytesIO, num_frames: int, target_size: int = 224) -> list[np.ndarray]:
    src = video_source if isinstance(video_source, io.BytesIO) else str(video_source)
    with av.open(src) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        duration = float(stream.duration * time_base) if stream.duration else None

        if duration is None or duration <= 0:
            raise ValueError(f"No duration info in {video_source}")

        target_times = np.linspace(0, duration, num_frames, endpoint=False)
        frames = []

        for t in target_times:
            container.seek(int(t / time_base), stream=stream, any_frame=False, backward=True)
            for frame in container.decode(video=0):
                if float(frame.pts * time_base) >= t:
                    # Resize with libswscale immediately — keeps worker RAM O(num_frames * target_size^2)
                    # instead of O(num_frames * native_resolution), which can be 30-50× larger.
                    resized = frame.reformat(width=target_size, height=target_size, format="rgb24")
                    frames.append(resized.to_ndarray())
                    break

        if len(frames) < num_frames:
            if not frames:
                raise ValueError(f"No frames decoded from {video_source}")
            frames.extend([frames[-1]] * (num_frames - len(frames)))
        return frames


def load_video_from_gcs(
    input: dict,
    bucket_name: str,
    prefix: str,
    with_audio: bool = True,
) -> dict:
    """
    Download a video from GCS to a temp file, process with ffmpeg, then discard the temp file.

    The large blob lands on disk (not in RAM); only the ffmpeg output — which may
    be smaller after audio stripping — is kept in input["video_raw"]. The temp
    file is always deleted before returning.

    Args:
        input: dict with at least SIQDatasetColumns.VIDEO_ID ("vid_name").
        bucket_name: GCS bucket name.
        prefix: path prefix within the bucket (default: "siq2/video").
        with_audio: if False, audio is stripped before storing.
    """

    video_id = input[SIQDatasetColumns.VIDEO_ID]

    video = _load_video_from_gcs(video_id, bucket_name, prefix, with_audio)
    input.update(**video)
    return input
