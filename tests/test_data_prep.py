"""Tests for the download → chunk → transcript pipeline in scripts/data_prep.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import webvtt

# ── load script as a module so we can test internal functions ─────────────────
_SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "data_prep", _SCRIPTS_DIR / "data_prep.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_slice_transcript = _mod._slice_transcript
_cut_chunk        = _mod._cut_chunk
_chunk_video      = _mod._chunk_video
_process_video    = _mod._process_video
_TRANSCRIPT_DIR   = _mod._TRANSCRIPT_DIR
_CHUNKS_FILE      = _mod._CHUNKS_FILE

from social_memory.transforms.text import load_chunk_transcript
from social_memory.constants import PATH_TO_DATA, SIQDatasetColumns

_VIDEO_DIR = Path(__file__).parents[1] / PATH_TO_DATA / "video"

# ── helpers ───────────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _local_test_videos(n: int = 5) -> list[str]:
    """Return up to n video IDs that have both a local .mp4 and a local .vtt."""
    with open(_CHUNKS_FILE) as f:
        chunks_map = json.load(f)
    result = []
    for vid_id in chunks_map:
        has_video = (_VIDEO_DIR / f"{vid_id}.mp4").exists()
        has_vtt   = (_TRANSCRIPT_DIR / f"{vid_id}.vtt").exists()
        if has_video and has_vtt:
            result.append(vid_id)
            if len(result) == n:
                break
    return result


# ── shared VTT fixture ────────────────────────────────────────────────────────
#  caption at 5s, 20s, 65s — lets tests check boundary behaviour cleanly

_SAMPLE_VTT = """\
WEBVTT

00:00:05.000 --> 00:00:10.000
Hello world

00:00:20.000 --> 00:00:30.000
This is the middle

00:01:05.000 --> 00:01:10.000
Near the end
"""


# ── _slice_transcript ─────────────────────────────────────────────────────────

class TestSliceTranscript:

    def _write_vtt(self, tmp_path: Path, vid_id: str = "testvid") -> None:
        (tmp_path / f"{vid_id}.vtt").write_text(_SAMPLE_VTT)

    def test_returns_captions_within_window(self, tmp_path, monkeypatch):
        self._write_vtt(tmp_path)
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)

        result = _slice_transcript("testvid", start=0, end=60)

        assert "Hello world" in result
        assert "This is the middle" in result
        assert "Near the end" not in result   # starts at 65s, outside window

    def test_excludes_captions_before_start(self, tmp_path, monkeypatch):
        self._write_vtt(tmp_path)
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)

        result = _slice_transcript("testvid", start=15, end=60)

        assert "Hello world" not in result    # starts at 5s, before window
        assert "This is the middle" in result

    def test_exact_boundary_start_is_inclusive(self, tmp_path, monkeypatch):
        self._write_vtt(tmp_path)
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)

        result = _slice_transcript("testvid", start=20, end=60)

        assert "This is the middle" in result

    def test_exact_boundary_end_is_exclusive(self, tmp_path, monkeypatch):
        self._write_vtt(tmp_path)
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)

        # caption starts at exactly 65s, window ends at 65s → excluded
        result = _slice_transcript("testvid", start=0, end=65)

        assert "Near the end" not in result

    def test_missing_vtt_returns_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)

        result = _slice_transcript("no_such_video", start=0, end=100)

        assert result == ""

    def test_adjacent_chunks_dont_overlap(self, tmp_path, monkeypatch):
        self._write_vtt(tmp_path)
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)

        chunk0 = _slice_transcript("testvid", start=0,  end=20)
        chunk1 = _slice_transcript("testvid", start=20, end=65)

        lines0 = set(chunk0.splitlines())
        lines1 = set(chunk1.splitlines())
        overlap = lines0 & lines1 - {""}
        assert not overlap, f"Overlapping lines between adjacent chunks: {overlap}"


# ── _cut_chunk ────────────────────────────────────────────────────────────────

class TestCutChunk:

    @patch("subprocess.run")
    def test_calls_ffmpeg_with_correct_times(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        src  = tmp_path / "video.mp4"
        dest = tmp_path / "chunk.mp4"
        src.write_bytes(b"fake")

        _cut_chunk(src, start=10.0, end=70.0, dest=dest)

        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert args[args.index("-ss") + 1] == "10.0"
        assert args[args.index("-to") + 1] == "70.0"
        assert str(src)  in args
        assert str(dest) in args

    @patch("subprocess.run")
    def test_uses_copy_codec(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        _cut_chunk(tmp_path / "src.mp4", 0, 10, tmp_path / "out.mp4")

        args = mock_run.call_args[0][0]
        assert "-c" in args
        assert args[args.index("-c") + 1] == "copy"

    @patch("subprocess.run")
    def test_creates_parent_directories(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        dest = tmp_path / "nested" / "deep" / "chunk.mp4"

        _cut_chunk(tmp_path / "src.mp4", 0, 10, dest)

        assert dest.parent.exists()


# ── _chunk_video ──────────────────────────────────────────────────────────────

class TestChunkVideo:

    def _fake_cut(self, src, start, end, dest):
        dest.write_bytes(b"chunk data")

    def test_correct_number_of_chunk_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)
        chunks = [[0, 60.0], [60.0, 120.0], [120.0, 148.0]]

        with patch.object(_mod, "_cut_chunk", side_effect=self._fake_cut), \
             patch.object(_mod, "_slice_transcript", return_value="text"):
            created = _chunk_video("vid1", tmp_path / "vid1.mp4", chunks, tmp_path)

        assert len(created) == 3
        for i in range(3):
            assert (tmp_path / f"vid1_chunk_{i:03d}.mp4").exists()

    def test_transcript_file_written_per_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)
        chunks = [[0, 60.0], [60.0, 120.0]]

        with patch.object(_mod, "_cut_chunk", side_effect=self._fake_cut), \
             patch.object(_mod, "_slice_transcript_vtt", return_value="WEBVTT\n"):
            _chunk_video("vid1", tmp_path / "vid1.mp4", chunks, tmp_path)

        assert (tmp_path / "vid1_chunk_000.vtt").exists()
        assert (tmp_path / "vid1_chunk_001.vtt").exists()

    def test_skips_existing_chunk_and_transcript(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)
        chunks = [[0, 60.0]]

        (tmp_path / "vid1_chunk_000.mp4").write_bytes(b"already here")
        (tmp_path / "vid1_chunk_000.vtt").write_text("already here")

        with patch.object(_mod, "_cut_chunk") as mock_cut, \
             patch.object(_mod, "_slice_transcript_vtt") as mock_slice:
            _chunk_video("vid1", tmp_path / "vid1.mp4", chunks, tmp_path)

        mock_cut.assert_not_called()
        mock_slice.assert_not_called()

    def test_chunk_indices_are_zero_padded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)
        chunks = [[i * 60.0, (i + 1) * 60.0] for i in range(12)]

        with patch.object(_mod, "_cut_chunk", side_effect=self._fake_cut), \
             patch.object(_mod, "_slice_transcript", return_value=""):
            _chunk_video("vid1", tmp_path / "vid1.mp4", chunks, tmp_path)

        assert (tmp_path / "vid1_chunk_000.mp4").exists()
        assert (tmp_path / "vid1_chunk_011.mp4").exists()


# ── _process_video ────────────────────────────────────────────────────────────

class TestProcessVideo:

    def test_returns_empty_string_for_unknown_id(self, tmp_path):
        result = _process_video("unknown_id", "bucket", "prefix", tmp_path, chunks_map={})
        assert result == ""

    def test_skips_download_when_all_chunks_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)
        chunks_map = {"vid1": {"chunks": [[0, 60.0], [60.0, 120.0]]}}

        for i in range(2):
            (tmp_path / f"vid1_chunk_{i:03d}.mp4").write_bytes(b"chunk")

        with patch.object(_mod, "_download_blob") as mock_dl:
            result = _process_video("vid1", "bucket", "prefix", tmp_path, chunks_map)

        mock_dl.assert_not_called()
        assert result == "vid1"

    def test_deletes_full_video_after_chunking(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)
        chunks_map = {"vid1": {"chunks": [[0, 60.0]]}}

        def fake_download(bucket, blob, dest):
            dest.write_bytes(b"full video")
            return True

        with patch.object(_mod, "_download_blob", side_effect=fake_download), \
             patch.object(_mod, "_chunk_video", return_value=[tmp_path / "vid1_chunk_000.mp4"]):
            _process_video("vid1", "bucket", "prefix", tmp_path, chunks_map)

        assert not (tmp_path / "_tmp_vid1.mp4").exists()

    def test_deletes_full_video_even_on_chunk_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, "_TRANSCRIPT_DIR", tmp_path)
        chunks_map = {"vid1": {"chunks": [[0, 60.0]]}}

        def fake_download(bucket, blob, dest):
            dest.write_bytes(b"full video")
            return True

        # The exception propagates out (finally still runs to clean up the file)
        with patch.object(_mod, "_download_blob", side_effect=fake_download), \
             patch.object(_mod, "_chunk_video", side_effect=RuntimeError("ffmpeg died")):
            with pytest.raises(RuntimeError):
                _process_video("vid1", "bucket", "prefix", tmp_path, chunks_map)

        assert not (tmp_path / "_tmp_vid1.mp4").exists()

    def test_returns_empty_string_when_download_fails(self, tmp_path):
        chunks_map = {"vid1": {"chunks": [[0, 60.0]]}}

        with patch.object(_mod, "_download_blob", return_value=False):
            result = _process_video("vid1", "bucket", "prefix", tmp_path, chunks_map)

        assert result == ""


# ── load_chunk_transcript transform ──────────────────────────────────────────

class TestLoadChunkTranscript:

    def _setup_vtt(self, tmp_path: Path, vid_id: str = "vid1") -> None:
        transcript_dir = tmp_path / "transcript"
        transcript_dir.mkdir()
        (transcript_dir / f"{vid_id}.vtt").write_text(_SAMPLE_VTT)

    def test_filters_by_time_window(self, tmp_path, monkeypatch):
        self._setup_vtt(tmp_path)
        monkeypatch.setattr("social_memory.transforms.text.PATH_TO_DATA", tmp_path)

        result = load_chunk_transcript({SIQDatasetColumns.VIDEO_ID: "vid1"}, start=0, end=60)

        assert "Hello world" in result["transcript"]
        assert "Near the end" not in result["transcript"]

    def test_missing_vtt_gives_empty_transcript(self, tmp_path, monkeypatch):
        (tmp_path / "transcript").mkdir()
        monkeypatch.setattr("social_memory.transforms.text.PATH_TO_DATA", tmp_path)

        result = load_chunk_transcript(
            {SIQDatasetColumns.VIDEO_ID: "no_such_video"}, start=0, end=60
        )
        assert result["transcript"] == ""

    def test_preserves_other_input_keys(self, tmp_path, monkeypatch):
        self._setup_vtt(tmp_path)
        monkeypatch.setattr("social_memory.transforms.text.PATH_TO_DATA", tmp_path)

        inp = {SIQDatasetColumns.VIDEO_ID: "vid1", "question": "What happened?"}
        result = load_chunk_transcript(inp, start=0, end=60)

        assert result["question"] == "What happened?"


# ── integration: real local videos ────────────────────────────────────────────

_LOCAL_VIDEOS = _local_test_videos(5)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
@pytest.mark.skipif(not _LOCAL_VIDEOS, reason="no local videos with matching VTTs found")
class TestRealVideoChunking:
    """End-to-end test on up to 5 videos that have local .mp4 and .vtt files."""

    def test_chunk_files_created_for_each_video(self, tmp_path):
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)

        for vid_id in _LOCAL_VIDEOS:
            video_path = _VIDEO_DIR / f"{vid_id}.mp4"
            chunks = chunks_map[vid_id]["chunks"]

            created = _chunk_video(vid_id, video_path, chunks, tmp_path)

            assert len(created) == len(chunks), f"{vid_id}: expected {len(chunks)} chunks"
            for i, chunk_path in enumerate(created):
                assert chunk_path.exists(),           f"{vid_id} chunk {i:03d} mp4 missing"
                assert chunk_path.stat().st_size > 0, f"{vid_id} chunk {i:03d} mp4 is empty"

    def test_transcript_files_created_alongside_chunks(self, tmp_path):
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)

        for vid_id in _LOCAL_VIDEOS:
            video_path = _VIDEO_DIR / f"{vid_id}.mp4"
            chunks = chunks_map[vid_id]["chunks"]

            _chunk_video(vid_id, video_path, chunks, tmp_path)

            for i in range(len(chunks)):
                txt = tmp_path / f"{vid_id}_chunk_{i:03d}.vtt"
                assert txt.exists(), f"{vid_id} transcript {i:03d} missing"

    def test_chunk_vtts_are_valid_and_parseable(self, tmp_path):
        """Each output .vtt must be parseable by webvtt."""
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)

        for vid_id in _LOCAL_VIDEOS:
            _chunk_video(vid_id, _VIDEO_DIR / f"{vid_id}.mp4", chunks_map[vid_id]["chunks"], tmp_path)

            for vtt_path in sorted(tmp_path.glob(f"{vid_id}_chunk_*.vtt")):
                try:
                    webvtt.read(str(vtt_path))
                except Exception as e:
                    pytest.fail(f"{vtt_path.name} is not valid VTT: {e}")

    def test_chunk_vtt_timestamps_are_relative_to_chunk_start(self, tmp_path):
        """All caption start times in a chunk VTT must be < chunk duration (not original video time)."""
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)

        for vid_id in _LOCAL_VIDEOS:
            chunks = chunks_map[vid_id]["chunks"]
            _chunk_video(vid_id, _VIDEO_DIR / f"{vid_id}.mp4", chunks, tmp_path)

            for i, (start, end) in enumerate(chunks):
                vtt_path = tmp_path / f"{vid_id}_chunk_{i:03d}.vtt"
                duration = end - start
                for caption in webvtt.read(str(vtt_path)):
                    assert caption.start_in_seconds < duration, (
                        f"{vid_id} chunk {i:03d}: caption starts at {caption.start_in_seconds:.3f}s "
                        f"but chunk duration is only {duration:.1f}s — timestamp not offset from chunk start"
                    )

    def test_chunk_vtt_text_matches_original(self, tmp_path):
        """Caption text in chunk VTTs must appear in the original video VTT."""
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)

        for vid_id in _LOCAL_VIDEOS:
            chunks = chunks_map[vid_id]["chunks"]
            _chunk_video(vid_id, _VIDEO_DIR / f"{vid_id}.mp4", chunks, tmp_path)

            full_text = "\n".join(c.text for c in webvtt.read(str(_TRANSCRIPT_DIR / f"{vid_id}.vtt")))

            for i in range(len(chunks)):
                vtt_path = tmp_path / f"{vid_id}_chunk_{i:03d}.vtt"
                for caption in webvtt.read(str(vtt_path)):
                    for line in caption.text.splitlines():
                        if line.strip():
                            assert line in full_text, (
                                f"{vid_id} chunk {i:03d}: line {line!r} not found in original VTT"
                            )

    def test_transcript_slices_stay_within_video_vtt(self, tmp_path):
        """Every line in a chunk transcript must appear in the full video VTT."""
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)

        for vid_id in _LOCAL_VIDEOS:
            chunks = chunks_map[vid_id]["chunks"]
            full_text = "\n".join(
                c.text for c in webvtt.read(str(_TRANSCRIPT_DIR / f"{vid_id}.vtt"))
            )
            for start, end in chunks:
                chunk_text = _slice_transcript(vid_id, start, end)
                for line in chunk_text.splitlines():
                    if line.strip():
                        assert line in full_text, (
                            f"{vid_id} [{start}-{end}]: spurious line {line!r}"
                        )

    def test_all_chunks_together_cover_full_vtt(self, tmp_path):
        """Union of all chunk transcripts should equal the full VTT text."""
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)

        for vid_id in _LOCAL_VIDEOS:
            chunks = chunks_map[vid_id]["chunks"]
            all_lines = set()
            for start, end in chunks:
                all_lines |= set(_slice_transcript(vid_id, start, end).splitlines())

            # Split caption text on newlines for both sides so multi-line
            # captions compare consistently with the joined _slice_transcript output
            full_lines = set(
                line
                for c in webvtt.read(str(_TRANSCRIPT_DIR / f"{vid_id}.vtt"))
                for line in c.text.splitlines()
            )
            for line in full_lines:
                if line.strip():
                    assert line in all_lines, (
                        f"{vid_id}: VTT line {line!r} not covered by any chunk"
                    )


# ── integration: upload path with real local video ────────────────────────────

_upload_chunks   = _mod._upload_chunks
_CHUNKS_PREFIX   = "siq2/chunks"
_ONE_VIDEO       = _LOCAL_VIDEOS[:1]  # just one video is enough


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
@pytest.mark.skipif(not _ONE_VIDEO, reason="no local videos with matching VTTs found")
class TestUploadChunks:
    """Chunk a real video locally, then verify GCS upload calls use the right blob names."""

    def _run(self, tmp_path, blob_exists_retval=False):
        """Chunk the first local video and run _upload_chunks with mocked GCS."""
        vid_id = _ONE_VIDEO[0]
        with open(_CHUNKS_FILE) as f:
            chunks_map = json.load(f)
        chunks = chunks_map[vid_id]["chunks"]

        _chunk_video(vid_id, _VIDEO_DIR / f"{vid_id}.mp4", chunks, tmp_path)

        uploaded = []
        def fake_upload(bucket, local_path, blob_name):
            uploaded.append(blob_name)

        with patch.object(_mod, "blob_exists", return_value=blob_exists_retval), \
             patch.object(_mod, "upload_file", side_effect=fake_upload):
            _upload_chunks("my-bucket", _CHUNKS_PREFIX, vid_id, len(chunks), tmp_path)

        return vid_id, len(chunks), uploaded

    def test_blob_names_use_vid_id_subfolder(self, tmp_path):
        vid_id, n_chunks, uploaded = self._run(tmp_path)
        for blob in uploaded:
            assert blob.startswith(f"{_CHUNKS_PREFIX}/{vid_id}/"), \
                f"unexpected blob path: {blob}"

    def test_each_chunk_produces_mp4_and_txt_blob(self, tmp_path):
        vid_id, n_chunks, uploaded = self._run(tmp_path)
        for i in range(n_chunks):
            assert f"{_CHUNKS_PREFIX}/{vid_id}/{i:03d}.mp4" in uploaded
            assert f"{_CHUNKS_PREFIX}/{vid_id}/{i:03d}.vtt" in uploaded

    def test_total_blobs_is_two_per_chunk(self, tmp_path):
        vid_id, n_chunks, uploaded = self._run(tmp_path)
        assert len(uploaded) == n_chunks * 2

    def test_skips_blobs_already_in_gcs(self, tmp_path):
        """When blob_exists returns True, upload_file should never be called."""
        vid_id, n_chunks, uploaded = self._run(tmp_path, blob_exists_retval=True)
        assert uploaded == [], "should skip all blobs that already exist in GCS"
