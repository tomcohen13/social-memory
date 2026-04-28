"""Unit tests for transforms"""
import base64
from unittest.mock import MagicMock, patch

import pytest

from social_memory.constants import SIQDatasetColumns
from social_memory.transforms.video import (
    _compute_clip_window,
    clip_around_oracle,
    encode_video,
    load_video_from_local,
    load_video_from_gcs,
)


# ── _compute_clip_window ──────────────────────────────────────────────────────

@pytest.mark.parametrize("oracle,output_length,duration,expected", [
    ((40, 50), 20, 100.0, (35, 55)),   # symmetric expansion around oracle
    ((2,  8),  20, 100.0, (0,  20)),   # near start, overflows right
    ((92, 98), 20, 100.0, (80, 100)),  # near end, overflows left
    ((10, 30), 20, 100.0, (10, 30)),   # oracle already fills window
    ((2,  5),  30, 10.0,  (0,  10)),   # short video, returns full duration
])
def test_compute_clip_window(oracle, output_length, duration, expected):
    assert _compute_clip_window("v", oracle, output_length, duration) == expected

# ── encode_video ──────────────────────────────────────────────────────────────

class TestEncodeVideo:
    def test_bytes_become_base64_string(self):
        raw = b"fake video bytes"
        result = encode_video({SIQDatasetColumns.VIDEO_RAW: raw})
        assert result[SIQDatasetColumns.VIDEO] == base64.b64encode(raw).decode("utf-8")
        assert SIQDatasetColumns.VIDEO_RAW not in result

    def test_none_raw_gives_none_video(self):
        result = encode_video({SIQDatasetColumns.VIDEO_RAW: None})
        assert result[SIQDatasetColumns.VIDEO] is None


# ── clip_around_oracle ────────────────────────────────────────────────────────

class TestClipAroundOracle:
    def _input(self, raw=b"raw bytes"):
        return {
            SIQDatasetColumns.VIDEO_ID: "vid1",
            SIQDatasetColumns.VIDEO_RAW: raw,
            "oracle": (40, 50),
            "duration": 100.0,
        }

    def test_missing_raw_returns_input_unchanged(self):
        inp = self._input(raw=None)
        result = clip_around_oracle(inp, output_length=20)
        assert result is inp
        assert result[SIQDatasetColumns.VIDEO_RAW] is None

    @patch("social_memory.transforms.video.subprocess.run")
    def test_clips_to_correct_window(self, mock_run):
        mock_run.return_value = MagicMock(stdout=b"clipped")
        result = clip_around_oracle(self._input(), output_length=20)

        assert result[SIQDatasetColumns.VIDEO_RAW] == b"clipped"
        args = mock_run.call_args[0][0]
        ss_idx = args.index("-ss")
        to_idx = args.index("-to")
        assert args[ss_idx + 1] == "35"
        assert args[to_idx + 1] == "55"


# ── load_video ────────────────────────────────────────────────────────────────

class TestLoadVideo:
    def test_missing_file_stores_none(self, tmp_path):
        inp = {SIQDatasetColumns.VIDEO_ID: "missing"}
        result = load_video_from_local(inp, directory=tmp_path)
        assert result[SIQDatasetColumns.VIDEO_RAW] is None

    def test_loads_bytes_and_duration(self, tmp_path):
        raw = b"fake mp4"
        (tmp_path / "vid1.mp4").write_bytes(raw)

        with patch("social_memory.transforms.video.get_duration", return_value=42.0):
            result = load_video_from_local({SIQDatasetColumns.VIDEO_ID: "vid1"}, directory=tmp_path)

        assert result[SIQDatasetColumns.VIDEO_RAW] == raw
        assert result["duration"] == 42.0

    @patch("social_memory.transforms.video.subprocess.run")
    def test_strips_audio_via_ffmpeg(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout=b"no audio")
        (tmp_path / "vid1.mp4").write_bytes(b"original")

        with patch("social_memory.transforms.video.get_duration", return_value=10.0):
            result = load_video_from_local(
                {SIQDatasetColumns.VIDEO_ID: "vid1"}, directory=tmp_path, with_audio=False
            )

        assert result[SIQDatasetColumns.VIDEO_RAW] == b"no audio"
        assert "-an" in mock_run.call_args[0][0]


# ── load_video_from_gcs ───────────────────────────────────────────────────────

class TestLoadVideoFromGcs:
    def _input(self):
        return {SIQDatasetColumns.VIDEO_ID: "vid1"}

    def _gcs_ctx(self, tmp_path_or_none):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=tmp_path_or_none)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    def test_missing_blob_stores_none(self):
        with patch("social_memory.transforms.video.download_to_temp", return_value=self._gcs_ctx(None)):
            result = load_video_from_gcs(self._input(), bucket_name="b", prefix="p")
        assert result[SIQDatasetColumns.VIDEO_RAW] is None

    @patch("social_memory.transforms.video.subprocess.run")
    def test_stores_bytes_and_duration(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout=b"video")
        (tmp_path / "tmp.mp4").write_bytes(b"raw")

        with patch("social_memory.transforms.video.download_to_temp", return_value=self._gcs_ctx(tmp_path / "tmp.mp4")), \
             patch("social_memory.transforms.video.get_duration", return_value=55.0):
            result = load_video_from_gcs(self._input(), bucket_name="b", prefix="p")

        assert result[SIQDatasetColumns.VIDEO_RAW] == b"video"
        assert result["duration"] == 55.0
