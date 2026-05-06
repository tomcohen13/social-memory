
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
