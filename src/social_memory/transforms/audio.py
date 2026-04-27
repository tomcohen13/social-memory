from pathlib import Path

def _load_single_audio(vid: str, directory: Path) -> tuple[str, tuple[str, str]]:
    mime_map = {"mp3": "audio/mpeg", "wav": "audio/wav"}
    for ext in ("mp3", "wav"):
        path = directory / ext / f"{vid}.{ext}"
        if path.is_file():
            return vid, (str(path), mime_map[ext])
    return vid, ("", "")

def load_audio(input: dict) -> dict:
    pass
