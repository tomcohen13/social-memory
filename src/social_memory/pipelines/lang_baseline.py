"""Test zero-shot performance of LLMs on transcript-only data"""
import os
import pandas as pd
from typing import List, Dict

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from tqdm.asyncio import tqdm

from social_memory.constants import PipelineNames
from social_memory.pipelines.base import Pipeline
from social_memory.utils import load_transcripts


class LanguagePipeline(Pipeline):

    NAME = PipelineNames.LANGUAGE

    def _load_model_runner(self) -> None:
        """
        Load LLM runner, restricting output tokens and with exponential-backoff retry.
        """
        llm = self._load_model()
        llm_with_retry = llm.with_retry(wait_exponential_jitter=True, stop_after_attempt=4)
        self.model_runner = self.prompt_template | llm_with_retry

    async def process_inputs(self, dataset: pd.DataFrame) -> List[Dict[str, str]]:
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

        if os.path.exists(self.path_to_output) and os.path.getsize(self.path_to_output) > 0:
            self.logger.info("loading previous results...")
            prev_results = pd.read_json(self.path_to_output, lines=True)
            ids_to_skip = set(prev_results['qid'].unique())
        else:
            ids_to_skip = set()

        transcripts = load_transcripts(
            video_ids=set(dataset['vid_name'].unique()) - ids_to_skip,
            max_workers=self.configs.max_concurrency
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

        # set up concurrency configs 
        config = RunnableConfig(max_concurrency=self.configs.max_concurrency)

        results: List[dict] = []
        unsaved: List[dict] = []
        errors = 0

        async for i, res in tqdm(
            self.model_runner.abatch_as_completed(inputs=inputs, config=config, return_exceptions=True),
            total=len(inputs),
            desc=f"Running model {self.configs.model}..."
        ):
            if isinstance(res, AIMessage):
                try:
                    result = int(res.content)
                    entry = {"qid": inputs[i]["qid"], "result": result}
                    results.append(entry)
                    unsaved.append(entry)
                except:
                    self.logger.error(f"There was an issue with: {inputs[i]['qid']}, error: {res}")
                    errors += 1
            else:
                self.logger.error(f"There was an issue with: {inputs[i]['qid']}, error: {res}")
                errors += 1

            if len(unsaved) >= 10:
                self.write_results_to_json(unsaved)
                unsaved = []
        if unsaved:
            self.write_results_to_json(unsaved)
            
        self.logger.info(f"Finished processing: {len(inputs)} inputs | errors: {errors}")
        return results
