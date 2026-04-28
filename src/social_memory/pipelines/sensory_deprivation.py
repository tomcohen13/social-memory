"""Sensory deprivation pipeline for testing MCQ validity"""

from typing import Dict, List

from social_memory.pipelines.base import BasePipeline


class SensoryDeprivationPipeline(BasePipeline):
    """
    Pipeline for testing MCQ validity by removing any contextual information
    and only providing the question and answer options to the model.
    If model performance is above chance (25% for 4 options), MCQs are likely too easy or leading.
    """

    NAME = "sensory_deprivation"

    def __init__(self, configs):
        super().__init__(configs)
        
        # add transform to empty the transcript
        self.transforms = []
    
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
        
        inputs = [
            {
                "qid": input["qid"],
                "question": input["q"],
                "options": "\n".join([f"{i}: {input[f'a{i}']}" for i in range(4)])
            }
            for input in inputs
        ]
        if len(inputs) < num_inputs_before:
            self.logger.warning(f"{num_inputs_before - len(inputs)} were missing a transcript and will be skipped.")
        
        return inputs
