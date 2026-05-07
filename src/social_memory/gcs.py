"""GCS utilities for streaming and uploading dataset files."""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, Optional


def _client():
    from google.cloud import storage
    return storage.Client()


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


def list_blobs(bucket, prefix) -> Iterator[Any]:
    from google.cloud import storage
    client = storage.Client()
    return client.bucket(bucket).list_blobs(prefix=prefix)


@contextmanager
def download_to_temp(bucket_name: str, name: str) -> Generator[Optional[Path], None, None]:
    """
    Download a GCS blob to a temporary file and yield its Path.

    Yields None if the blob does not exist. The temp file is deleted on exit
    regardless of whether an exception occurred.

    Usage:
        with download_to_temp(bucket, blob_name) as path:
            if path is None:
                ...  # blob not found
            else:
                ...  # use path
    """
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(name)
    if not blob.exists():
        yield None
        return

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        blob.download_to_filename(tmp_path)
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
