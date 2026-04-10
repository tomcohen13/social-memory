"""Test zero-shot performance of LLMs on transcript-only data"""

from pathlib import Path
import sys

# Repo uses a src/ layout; running `python scripts/run_pipeline.py` does not put `src` on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_src = _REPO_ROOT / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

from argparse import ArgumentParser
from dotenv import load_dotenv

from social_memory.constants import DEFAULT_MODEL, DEFAULT_MODEL_PROVIDER
from social_memory.pipelines.base import PipelineConfig
from social_memory.pipelines.registry import PIPELINE_REGISTRY

load_dotenv()

parser = ArgumentParser()
parser.add_argument(
    "--pipeline",
    type=str,
    choices=sorted(PIPELINE_REGISTRY),
    help="Which pipeline implementation to run.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default=":".join([DEFAULT_MODEL_PROVIDER, DEFAULT_MODEL]),
)
parser.add_argument(
    "--split",
    type=str,
    choices=["train", "val", "test", "demo"],
    default="demo",
    help="The split of the dataset to run the pipeline on",
)
parser.add_argument(
    "--max_concurrency",
    type=int,
    default=2,
    help="Max concurrent requests to llm",
)

args = parser.parse_args()

async def main():
    configs = PipelineConfig(
        model=args.model,
        split=args.split,
        max_concurrency=args.max_concurrency,
    )
    if args.pipeline not in PIPELINE_REGISTRY:
        raise ValueError(f"No pipeline found for key: {args.pipeline}")
    
    pipeline_cls = PIPELINE_REGISTRY[args.pipeline]
    pipeline = pipeline_cls(configs=configs)
    await pipeline.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
