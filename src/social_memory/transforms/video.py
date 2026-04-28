"""Video transforms"""
import base64
import subprocess
from pathlib import Path
from typing import List, Tuple

from social_memory.constants import (
    PATH_TO_DATA,
    SIQDatasetColumns,
    DirPaths,
)
from social_memory.gcs import download_to_temp, video_blob_name
from social_memory.utils import get_duration


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
    start, end = _compute_clip_window(
        video_id=input[SIQDatasetColumns.VIDEO_ID],
        oracle=input["oracle"],
        output_length=output_length,
        duration=input["duration"]
    )

    raw: bytes | None = input.get(SIQDatasetColumns.VIDEO_RAW)
    if raw is None:
        return input

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


def load_video(
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

    with download_to_temp(bucket_name, video_blob_name(video_id, prefix)) as tmp_path:
        if tmp_path is None:
            input[SIQDatasetColumns.VIDEO_RAW] = None
            return input

        input["duration"] = get_duration(tmp_path)

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

    input[SIQDatasetColumns.VIDEO_RAW] = result.stdout
    return input
