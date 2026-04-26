"""Test zero-shot performance of LLMs on video data"""
import base64
import os
import pandas as pd
from typing import List, Dict

from tqdm import tqdm

from social_memory.constants import PipelineNames
from social_memory.pipelines.base import BasePipeline
from social_memory.utils import load_audios


class AudioPipeline(BasePipeline):
    NAME = PipelineNames.AUDIO

    def _load_model_runner(self) -> None:
        llm = self._load_model()
        llm_with_retry = llm.with_retry(wait_exponential_jitter=True, stop_after_attempt=4)
        self.model_runner = self.prompt_template | llm_with_retry

    async def process_inputs(self, dataset: pd.DataFrame) -> List[Dict]:
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

        if os.path.exists(self.path_to_output) and os.path.getsize(self.path_to_output) > 0:
            self.logger.info("loading previous results...")
            prev_results = pd.read_json(self.path_to_output, lines=True)
            ids_to_skip = set(prev_results['qid'].unique())
        else:
            ids_to_skip = set()

        audios = load_audios(
            video_ids=set(dataset['vid_name'].unique()) - ids_to_skip,
            max_workers=self.configs.max_concurrency
        )

        audio_paths = {vid: (path, mime) for vid, (path, mime) in audios.items() if path}
        self.logger.info(f"Uploading {len(audio_paths)} audio files to Gemini Files API...")

        audio_files = {}
        for vid, (path, mime_type) in tqdm(audio_paths.items(), desc="Uploading audio"):
            with open(path, "rb") as audio_file:
                try:
                    encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
                    audio_files[vid] = encoded_audio
                except Exception as e:
                    self.logger.error(f"Failed to upload audio for {vid}: {e}")

        inputs = [
            {
                "qid": row["qid"],
                "audio": audio_files.get(row['vid_name']),
                "question": row["q"],
                "mime_type": audio_paths.get(row["vid_name"])[1],
                "options": "\n".join([f"{i}: {row[f'a{i}']}" for i in range(4)])
            }
            for i, row in dataset.iterrows()
            if audio_files.get(row['vid_name'])
        ]
        if len(inputs) < len(dataset):
            self.logger.warning(f"{len(dataset) - len(inputs)} were missing a video and will be skipped.")

        return inputs
