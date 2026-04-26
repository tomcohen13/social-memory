
from typing import List

from social_memory.pipelines.lang_baseline import LanguagePipeline


class MCQValidationPipeline(LanguagePipeline):
    """
    Pipeline for testing MCQ validity by removing any contextual information (i.e., transcripts)
    and only providing the question and answer options to the model.
    Model performance should be ~chance (25% for 4 options) if the MCQs are valid.
    """

    NAME = "mcq_validation"

    def transform_inputs(self, inputs: List[dict]) -> List[dict]:
        inputs_without_transcripts = [
            {
                k: (v if k != "transcript" else "") # replace transcript with empty string
                for k, v in input.items()
            }
            for input in inputs
        ]
        return inputs_without_transcripts
