
from typing import List
import pytest
from social_memory.utils import get_duration
from social_memory.constants import PATH_TO_AUGMENTED_DATA
from social_memory.gcs import download_to_temp, video_blob_name


# ── get duration ──────────────────────────────────────────────────────────────

class TestGetDuration:

    VIDEO_ID = "4Vic0qKl64Y"
    TRUE_LENGTH = 319
    
    def test_on_local_video(self):
        PATH_TO_VIDEO = PATH_TO_AUGMENTED_DATA / "video" / f"{self.VIDEO_ID}.mp4"
        assert int(get_duration(PATH_TO_VIDEO)) == self.TRUE_LENGTH


    def test_on_gcs_video(self):
        from social_memory.constants import GCS_BUCKET, GCS_PREFIX
        with download_to_temp(GCS_BUCKET, video_blob_name(self.VIDEO_ID, GCS_PREFIX)) as path:
            if path is None:
                raise FileNotFoundError("missing blob")
            assert int(get_duration(str(path))) == self.TRUE_LENGTH



# ── group_inputs_by_video_id ──────────────────────────────────────────────────

from social_memory.utils import group_inputs_by_video_id


def _make_input(vid: str, q: str, qid: str, oracle_idx: int = 0, chunk_ids: list = None) -> dict:
    return {
        "vid_name": vid,
        "q": q,
        "qid": qid,
        "oracle_idx": oracle_idx,
        "chunk_ids": chunk_ids or [0, 1, 2],
    }


class TestGroupInputsByVideoId:

    def test_multiple_questions_same_video_are_merged(self):
        inputs = [
            _make_input("vid1", "why?", "q1"),
            _make_input("vid1", "what?", "q2"),
        ]
        result = group_inputs_by_video_id(inputs)
        assert len(result) == 1
        assert set(result[0]["questions"]) == {"why?", "what?"}
        assert set(result[0]["qid"]) == {"q1", "q2"}

    def test_different_videos_produce_separate_groups(self):
        inputs = [
            _make_input("vid1", "why?", "q1"),
            _make_input("vid2", "how?", "q2"),
        ]
        result = group_inputs_by_video_id(inputs)
        assert len(result) == 2
        vids = {r["vid_name"] for r in result}
        assert vids == {"vid1", "vid2"}

    def test_q_column_renamed_to_questions(self):
        inputs = [_make_input("vid1", "why?", "q1")]
        result = group_inputs_by_video_id(inputs)
        assert "questions" in result[0]
        assert "q" not in result[0]

    def test_single_question_wrapped_in_list(self):
        inputs = [_make_input("vid1", "why?", "q1")]
        result = group_inputs_by_video_id(inputs)
        assert isinstance(result[0]["questions"], list)
        assert result[0]["questions"] == ["why?"]

    def test_oracle_idx_and_chunk_ids_preserved(self):
        inputs = [_make_input("vid1", "why?", "q1", oracle_idx=3, chunk_ids=[0, 1])]
        result = group_inputs_by_video_id(inputs)
        assert result[0]["oracle_idx"] == 3
        assert result[0]["chunk_ids"] == [0, 1]