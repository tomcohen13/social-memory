"""Test zero-shot performance of LLMs on transcript-only data"""

from typing import List, Dict

from social_memory.constants import PipelineNames
from social_memory.pipelines.base import BasePipeline
from social_memory.transforms import apply_transform_with_concurrency
from social_memory.transforms.text import load_transcript

class LanguagePipeline(BasePipeline):

    NAME = PipelineNames.LANGUAGE

    def __init__(self, configs):
        super().__init__(configs)
        
        self.transforms = [
            load_transcript,
        ]
        

    def _load_model_runner(self) -> None:
        """
        Load LLM runner, restricting output tokens and with exponential-backoff retry.
        """
        llm = self._load_model()
        llm_with_retry = llm.with_retry(wait_exponential_jitter=True, stop_after_attempt=4)
        self.model_runner = self.prompt_template | llm_with_retry

    async def process_inputs(self, inputs: List[Dict]) -> List[Dict[str, str]]:
        """
        Prepare inputs as dictionaries with keys: 'qid', 'transcript', 'question', 'options'

        Args:
            inputs: a list of dictionaries, each containing the following keys:
                qid (str): question id
                vid_name (str): the video id
                q (str): question content
                a0, a1, a2, a3: answer options
        
        Return: iterable object with inputs (dict) ready for model processing
        """
        
        num_inputs_before = len(inputs)
        
        for transform in self.transforms:
            # apply each transform to inputs using max_concurrency workers
            self.logger.info(f"Applying transform {transform.__name__}...")
            inputs = await apply_transform_with_concurrency(
                transform=transform,
                inputs=inputs,
                max_concurrency=self.configs.max_concurrency
            )
     
        inputs = [
            {
                "qid": input["qid"],
                "transcript": input["transcript"],
                "question": input["q"],
                "options": "\n".join([f"{i}: {input[f'a{i}']}" for i in range(4)])
            }
            for input in inputs
            if input["transcript"] != ""
        ]
        if len(inputs) < num_inputs_before:
            self.logger.warning(f"{num_inputs_before - len(inputs)} were missing a transcript and will be skipped.")
        
        return inputs
