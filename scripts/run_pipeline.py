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
load_dotenv()

from social_memory.pipelines.base import PipelineConfig
from social_memory.pipelines.registry import PIPELINE_REGISTRY


parser = ArgumentParser()
parser.add_argument(
    "--conf",
    type=str,
    required=False,
    help="Path to yaml file with pipeline configurations. Command-line args will override yaml configs.",
)
parser.add_argument(
    "--pipeline",
    type=str,
    choices=sorted(PIPELINE_REGISTRY),
    help="Which pipeline implementation to run.",
)
parser.add_argument(
    "--model",
    type=str,
    # default=":".join([DEFAULT_MODEL_PROVIDER, DEFAULT_MODEL]),
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["siq2", "siq2long"]
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
parser.add_argument(
    "--experiment_name",
    type=str,
    help="Name of the experiment for output file name",
)
parser.add_argument(
    "--run_id",
    type=str,
    required=False,
    help="Run id if referencing a previous one",
)

args = parser.parse_args()

async def main():
    
    raw = {}
    
    if args.conf:
        import yaml
        raw = yaml.safe_load(open(args.conf))
    
    # override yaml configs with command-line args if provided
    for arg in args.__dict__:
        value = getattr(args, arg)
        if value is not None:
            raw[arg] = value
    
    pipeline_name = raw.pop("pipeline")
    configs = PipelineConfig(**raw)
    
    if pipeline_name not in PIPELINE_REGISTRY:
        raise ValueError(f"No pipeline found for key: {pipeline_name}")
    
    pipeline_cls = PIPELINE_REGISTRY[pipeline_name]
    pipeline = pipeline_cls(configs=configs)
    await pipeline.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
