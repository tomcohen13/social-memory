"""GCS utilities for streaming and uploading dataset files."""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, Optional

_gcs_client = None


def _client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def default_bucket() -> str:
    """Return the bucket name from the GCS_BUCKET env var. Raises if not set."""
    import os
    bucket = os.getenv("GCS_BUCKET", "")
    if not bucket:
        raise ValueError("GCS_BUCKET environment variable is not set.")
    return bucket


def video_blob_name(video_id: str, prefix: str) -> str:
    return f"{prefix}/{video_id}.mp4"


def blob_exists(bucket_name: str, name: str) -> bool:
    return _client().bucket(bucket_name).blob(name).exists()


def get_blob_from_path(bucket_name, path):
    return _client().bucket(bucket_name).get_blob(path)


def list_blobs(bucket, prefix) -> Iterator[Any]:
    return _client().bucket(bucket).list_blobs(prefix=prefix)


def download_to_memory(bucket_name: str, name: str) -> Optional[bytes]:
    """
    Download a GCS blob into memory and return the raw bytes.

    Returns None if the blob does not exist. Prefer this over download_to_temp
    when the caller can work with bytes or a BytesIO directly — it skips the
    disk write + read round-trip entirely.
    """
    from google.cloud.exceptions import NotFound
    try:
        return _client().bucket(bucket_name).blob(name).download_as_bytes()
    except NotFound:
        return None


@contextmanager
def download_to_temp(bucket_name: str, name: str) -> Generator[Optional[Path], None, None]:
    """
    Download a GCS blob to a temporary file and yield its Path.

    Yields None if the blob does not exist. The temp file is deleted on exit
    regardless of whether an exception occurred.

    Prefer download_to_memory when the caller supports bytes or file-like objects.
    """
    from google.cloud.exceptions import NotFound
    fd, tmp_path = tempfile.mkstemp(suffix=Path(name).suffix)
    os.close(fd)
    try:
        _client().bucket(bucket_name).blob(name).download_to_filename(tmp_path)
    except NotFound:
        os.unlink(tmp_path)
        yield None
        return
    try:
        yield Path(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def upload_bytes(bucket_name: str, data: bytes, name: str, content_type: str = "video/mp4") -> None:
    """Upload raw bytes to a GCS blob."""
    _client().bucket(bucket_name).blob(name).upload_from_string(data, content_type=content_type)


def upload_file(bucket_name: str, local_path: Path, name: str) -> None:
    """Upload a local file to a GCS blob."""
    _client().bucket(bucket_name).blob(name).upload_from_filename(str(local_path))
