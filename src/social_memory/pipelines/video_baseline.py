"""Test zero-shot performance of LLMs on video data"""
from typing import List, Dict

from social_memory.constants import PipelineNames
from social_memory.pipelines.base import BasePipeline
from social_memory.transforms import Transform, TransformList, apply_transform_with_concurrency
from social_memory.transforms.video import (
    encode_video,
    load_video_from_local,
    load_video_from_gcs,
)


class VideoPipeline(BasePipeline):
    """
    Video-only pipeline (i.e., audio is stripped from videos).

    When PipelineConfig.gcs_bucket is set, videos are streamed directly from
    GCS (gs://<gcs_bucket>/<gcs_prefix>/<vid_id>.mp4). Otherwise, videos are
    read from the local filesystem under PATH_TO_DATA/video/.
    """

    NAME = PipelineNames.VIDEO

    def __init__(self, configs):
        super().__init__(configs)

        if configs.gcs_bucket:
            self.logger.info(f"Video source: GCS — gs://{configs.gcs_bucket}/{configs.gcs_prefix}")
            load_video = Transform(
                load_video_from_gcs,
                bucket_name=configs.gcs_bucket,
                prefix=configs.gcs_prefix,
            )
        else:
            from social_memory.constants import PATH_TO_DATA, DirPaths
            self.logger.info(f"Video source: local — {PATH_TO_DATA / DirPaths.VIDEO}")
            load_video = load_video_from_local

        self.transforms = TransformList([load_video, encode_video])

    def _load_model_runner(self) -> None:
        llm = self._load_model()
        llm_with_retry = llm.with_retry(wait_exponential_jitter=True, stop_after_attempt=4)
        self.model_runner = self.prompt_template | llm_with_retry

    async def process_inputs(self, inputs: List[Dict]) -> List[Dict]:
        """
        Prepare inputs as dictionaries with keys: 'qid', 'video', 'question', 'options'
        Args:
            dataset: a pandas dataframe, assumed to have the following columns:
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

            inputs = await apply_transform_with_concurrency(transform, inputs, max_concurrency=2)

        inputs = [
            {
                "qid": input["qid"],
                "video": input["video"],
                "question": input["q"],
                "options": "\n".join([f"{i}: {input[f'a{i}']}" for i in range(4)])
            }
            for input in inputs
            if input.get("video")
        ]
        if len(inputs) < num_inputs_before:
            self.logger.warning(f"{num_inputs_before - len(inputs)} were missing a video and will be skipped.")

        return inputs
