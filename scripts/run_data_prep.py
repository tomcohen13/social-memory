"""Runner for the data preparation pipeline."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_src = _REPO_ROOT / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

import asyncio
import argparse
from dotenv import load_dotenv

load_dotenv()

from social_memory.data_augmentation.data_prep import run

parser = argparse.ArgumentParser(description="Augment qa_augmented.json with scene-context questions")
parser.add_argument("--model",           type=str, default="gemini-2.5-pro")
parser.add_argument("--max_concurrency", type=int, default=4)
args = parser.parse_args()

asyncio.run(run(model=args.model, max_concurrency=args.max_concurrency))
