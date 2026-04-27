"""Text transforms"""

from pathlib import Path

from social_memory.constants import PATH_TO_DATA, SIQDatasetColumns
from social_memory.utils import _read_vtt_file


def load_transcript(input: dict) -> dict:
    """
    Load transcript from .vtt file for a given video id and add it to the input dict
    Args:
        input: a dictionary with at least the key "video_id" corresponding to the video id for which to load transcript"
    Returns:
        input dict with an additional key "transcript" containing the transcript text.
    """
    video_id = input[SIQDatasetColumns.VIDEO_ID]
    path = Path(PATH_TO_DATA) / "transcript" / f"{video_id}.vtt"
    if not path.is_file():
        input["transcript"] = ""
    else:
        input["transcript"] = _read_vtt_file(path)
    return input


def empty_transcript(input: dict) -> dict:
    """Overwrite transcript field with empty string."""

    input["transcript"] = ""
    return input
