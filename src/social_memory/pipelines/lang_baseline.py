"""Test zero-shot performance of LLMs on transcript-only data"""
from ast import Dict, List
import logging
import os
import sys
from typing import Any, Iterable
import pandas as pd

from argparse import ArgumentParser
from dotenv import load_dotenv

from social_memory.pipelines.base import Pipeline
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from tqdm.asyncio import tqdm

from social_memory.constants import DEFAULT_MODEL, DEFAULT_MODEL_PROVIDER, PipelineNames
from social_memory.prompts import PROMPT_TEMPLATE_TRANSCRIPT
from social_memory.utils import compute_correctness, load_qa_dataset, load_transcripts


parser = ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    # required=True,
    default=":".join([DEFAULT_MODEL_PROVIDER, DEFAULT_MODEL]),
)
parser.add_argument(
    "--split",
    type=str,
    # required=True,
    choices=["train", "val", "test", "demo"],
    default="demo",
    help="The split of the dataset to run the pipeline on",
)
parser.add_argument(
    "--max_concurrency",
    type=int,
    # required=True,
    default=2,
    help="Max concurrent requests to llm",
)

args = parser.parse_args()

PIPELINE_NAME = "language-only"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/{PIPELINE_NAME}_{args.model.replace(":", "_")}_{args.split}.log'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LanguagePipeline(Pipeline):

    NAME = PipelineNames.LANGUAGE


    def _load_model_runner(self) -> None:
        llm_with_retry = init_chat_model(self.configs.model).with_retry(wait_exponential_jitter=True, stop_after_attempt=4)
        self.model_runner = self.prompt_template | llm_with_retry


    def process_inputs(self, dataset: pd.DataFrame) -> Iterable[Dict[str, str]]:
        """
        Prepare inputs as dictionaries with keys: 'qid', 'transcript', 'question', 'options'

        Args:
            dataset: a pandas dataframe, assumed to have the following columns:
                qid (str): question id
                vid_name (str): the video id
                q (str): question content
                a0, a1, a2, a3: answer options
        
        Return: iterable object with inputs (dict) ready for model processing
        """

        if os.path.exists(self.path_to_output):
            self.logger("loading previous results...")
            prev_results = pd.read_csv(self.path_to_output)
            ids_to_skip = set(prev_results['qid'].unique())
        else:
            ids_to_skip = set()

        transcripts = load_transcripts(
            video_ids=set(dataset['vid_name'].unique()) - ids_to_skip,
            max_workers=args.max_concurrency
        )

        inputs = [
            {
                "qid": row["qid"],
                "transcript": transcripts.get(row['vid_name'], ""),
                "question": row["q"],
                "options": "\n".join([f"{i}: {row[f'a{i}']}" for i in range(4)])
            }
            for i, row in dataset.iterrows()
            if transcripts.get(row['vid_name']) != ""
        ]
        if len(inputs) < len(dataset):
            self.logger.warning(f"{len(dataset) - len(inputs)} were missing a transcript and will be skipped.")
        
        return inputs


    async def run_model_on_inputs(self, inputs: List[Dict[str, str]]):
        """
        Execute selected model on inputs concurrently
        """

        try: # try loading previous results
            results_df = pd.read_csv(self.path_to_output)
        except:
            results_df = pd.DataFrame(columns=["qid", "result"])
        
        # set up concurrency configs 
        config = RunnableConfig(max_concurrency=self.configs.max_concurrency)

        results: List[dict] = []
        errors = 0

        async for i, res in tqdm(
            self.model_runner.abatch_as_completed(inputs=inputs, config=config, return_exceptions=True),
            total=len(inputs),
            desc=f"Running model {self.configs.model}..."
        ):
            if isinstance(res, AIMessage):
                result = int(res.content)
                results.append({"qid": inputs[i]["qid"], "result": result})
            else:
                # if isinstance(review, (Exception, ValueError, ValidationError)):
                self.logger.error(f"There was an issue with: {inputs[i]['qid']}, error: {res}")
                errors += 1

            if i > 0 and i % 10 == 0:
                correctness = compute_correctness(results_df)
                logger.info(f"Accuracy: {correctness}")
            
        self.logger.info(f"Finished processing: {len(inputs)} inputs | errors: {errors}")
        return results
