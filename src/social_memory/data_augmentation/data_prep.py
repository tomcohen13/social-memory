"""
Data preparation pipeline: augments qa_augmented.json with scene-context questions.

For each question in siq2_augmented/qa_augmented.json, uploads the corresponding
60-second clip to the Gemini Files API and rewrites the question with a brief
scene description prepended. Results are written to the `augmented_q` column
and saved incrementally so the run is fully resumable.

Usage:
    python scripts/run_data_prep.py [--model gemini-2.5-pro] [--max_concurrency 4]
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types as google_types
from tqdm.asyncio import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CLIP_DIR  = _REPO_ROOT / "datasets" / "socialiq2" / "siq2" / "video"
_QA_PATH   = _REPO_ROOT / "datasets" / "socialiq2" / "siq2_augmented" / "qa_augmented.json"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "You will be given a short video clip and a question about it.\n\n"
    "Your task: Rewrite the question by prepending a brief scene description that captures (max 20 words):\n"
    "- Who is in the scene (appearance, relationship, setting)\n"
    "- What is happening in the clip (actions, emotions, interactions)\n"
    "- Any relevant context needed to understand the situation\n"
    "- ONLY add information that is essential and can be observed in the clip, "
    "do not assume anything beyond it.\n\n"
    "Then append the original question unchanged.\n\n"
    "Format: [Scene: <description>] <original question>\n\n"
    "Return ONLY the rewritten question, nothing else."
)

MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_prep.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload_clip(client: google_genai.Client, vid_id: str) -> google_types.File | None:
    """Upload a 60-second clip to the Gemini Files API. Returns None if clip is missing."""
    path = _CLIP_DIR / f"{vid_id}.mp4"
    if not path.exists():
        logger.warning(f"[{vid_id}] clip not found at {path}")
        return None

    logger.info(f"[{vid_id}] uploading clip...")
    with open(path, "rb") as f:
        uploaded = client.files.upload(
            file=f,
            config=google_types.UploadFileConfig(mime_type="video/mp4"),
        )

    for _ in range(30):
        if client.files.get(name=uploaded.name).state.name == "ACTIVE":
            break
        time.sleep(3)

    logger.info(f"[{vid_id}] ready: {uploaded.name}")
    return uploaded


async def rewrite_question(
    client: google_genai.Client,
    model: str,
    clip_file: google_types.File,
    question: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call Gemini to prepend scene context to a single question."""
    loop = asyncio.get_event_loop()

    def _invoke() -> str:
        response = client.models.generate_content(
            model=model,
            contents=[clip_file, f"Original question: {question}"],
            config=google_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=8192,
                temperature=1.0,
            ),
        )
        text = response.text
        if not text and response.candidates:
            parts = response.candidates[0].content.parts or []
            text_parts = [p.text for p in parts if hasattr(p, "text") and p.text]
            text = text_parts[-1] if text_parts else ""
        return (text or "").strip()

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                return await loop.run_in_executor(None, _invoke)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise


def needs_augmentation(row: pd.Series) -> bool:
    """True if this row still needs a valid augmented_q."""
    val = str(row.get("augmented_q", ""))
    if not val or val == "nan":
        return True
    # Truncated: original ends with ? but augmented_q doesn't
    if str(row.get("q", "")).endswith("?") and not val.endswith("?"):
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(model: str = "gemini-2.5-pro", max_concurrency: int = 4) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set — check your .env file")

    client    = google_genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    qa = pd.read_json(_QA_PATH, lines=True)
    if "augmented_q" not in qa.columns:
        qa["augmented_q"] = ""
    qa["augmented_q"] = qa["augmented_q"].fillna("").astype(str)

    todo = qa[qa.apply(needs_augmentation, axis=1)]
    logger.info(f"Rows to augment: {len(todo)} / {len(qa)}")

    if todo.empty:
        logger.info("Nothing to do.")
        return

    for vid_id, group in todo.groupby("vid_name"):
        logger.info(f"\n[{vid_id}] {len(group)} questions")

        clip_file = upload_clip(client, vid_id)
        if clip_file is None:
            continue

        # Fire all questions for this video in parallel under the semaphore
        indices = list(group.index)
        questions = [group.loc[idx, "q"] for idx in indices]
        tasks = [rewrite_question(client, model, clip_file, q, semaphore) for q in questions]

        results = await tqdm(
            asyncio.gather(*tasks, return_exceptions=True),
            total=len(tasks),
            desc=vid_id,
        )

        for idx, result in zip(indices, results):
            if isinstance(result, Exception):
                logger.error(f"  [{idx}] error: {result}")
            else:
                qa.at[idx, "augmented_q"] = result
                logger.info(f"  [{idx}] {result}")

        # Save after each video so progress is never lost
        qa.to_json(_QA_PATH, orient="records", lines=True)
        logger.info(f"[{vid_id}] saved.")

    # Final column ordering
    col_order = [
        "qid", "vid_name", "source_type",
        "q", "augmented_q",
        "ts", "full_video_ts", "full_video_duration",
        "ans_corr", "answer_idx", "idx_types",
        "a0", "a1", "a2", "a3",
        "result", "is_correct",
    ]
    qa = qa[[c for c in col_order if c in qa.columns]]
    qa.to_json(_QA_PATH, orient="records", lines=True)

    remaining = qa[qa.apply(needs_augmentation, axis=1)]
    logger.info(f"Done. Rows still needing augmentation: {len(remaining)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment qa_augmented.json with scene-context questions")
    parser.add_argument("--model",           type=str, default="gemini-2.5-pro")
    parser.add_argument("--max_concurrency", type=int, default=4)
    args = parser.parse_args()

    asyncio.run(run(model=args.model, max_concurrency=args.max_concurrency))
