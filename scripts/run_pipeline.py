"""Run selected pipeline on specific split"""


from argparse import ArgumentParser

from src.constants import MODELS_TO_TEST


parser = ArgumentParser()
parser.add_argument(
    "--pipeline",
    type=str,
    required=True,
    choices=["language", "audio", "vision", "all"],
    help="Choose from: transcript, full",
)
parser.add_argument(
    "--model",
    required=True,
    choices=[f"{provider}:{model}" for provider, models in MODELS_TO_TEST.items() for model in models],
    help="A str of provider:model to run the pipeline on."
)
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["train", "val", "test"],
    help="The split of the dataset to run the pipeline on",
)
parser.add_argument(
    "--max_docs",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--experiment",
    type=str,
    required=True,
    default="",
    help="Experiment name. Results will be saved under results/<benchmark>/<dataset>/<experiment or all_results>.csv"
)
parser.add_argument(
    "--debug",
    type=bool,
    help="Enable debug mode with more verbose logging.",
    default=False,
)
args = parser.parse_args()


