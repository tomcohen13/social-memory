import base64
import subprocess
from pathlib import Path
from typing import List, Tuple

from social_memory.constants import (
    PATH_TO_DATA,
    PATH_TO_AUGMENTED_DATA,
    SocialIQDatasetColumns,
    DirPaths,
)
from social_memory.utils import get_duration


def _clip_around_oracle(
    video_id: str,
    oracle: Tuple[int, int] | List[int],
    output_length: int,
) -> Tuple[int, int]:
    """
    Return a (start, end) window of `output_length` seconds centered around the oracle segment.

    Expands symmetrically from the oracle boundaries first; if one side hits the video edge,
    the remaining expansion is applied to the other side. Returns (0, duration) for videos
    shorter than `output_length`.
    """
    full_video_path = f"{PATH_TO_AUGMENTED_DATA}/video/{video_id}.mp4"
    duration = get_duration(full_video_path)
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
    video_id = input[SocialIQDatasetColumns.VIDEO_ID]
    oracle = input["oracle"]
    start, end = _clip_around_oracle(video_id, oracle, output_length)

    raw: bytes | None = input.get("video_raw")
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
    input["video_raw"] = result.stdout
    return input


def load_video(
    input: dict,
    directory: Path = Path(PATH_TO_DATA) / DirPaths.VIDEO,
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
    video_id = input[SocialIQDatasetColumns.VIDEO_ID]
    path = directory / f"{video_id}.mp4"

    if not path.is_file():
        input["video_raw"] = None
        return input

    if with_audio:
        with open(path, "rb") as f:
            input["video_raw"] = f.read()
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
    input["video_raw"] = result.stdout
    return input


def encode_video(input: dict) -> dict:
    """Convert input["video_raw"] bytes to a base64 string in input["video"]."""
    raw: bytes | None = input.pop("video_raw", None)
    input["video"] = base64.b64encode(raw).decode("utf-8") if raw is not None else None
    return input
