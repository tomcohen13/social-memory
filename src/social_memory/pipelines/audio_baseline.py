"""Test zero-shot performance of LLMs on audio data"""
import asyncio
import os
import re
import pandas as pd
from typing import List, Dict

from google import genai as google_genai
from google.genai import types as google_types
from tqdm.asyncio import tqdm
from tqdm import tqdm as tqdm_sync

from social_memory.constants import PipelineNames
from social_memory.pipelines.base import Pipeline
from social_memory.prompts import AUDIO_SYSTEM_PROMPT
from social_memory.utils import load_audios

MAX_RETRIES = 4

class AudioPipeline(Pipeline):
    """
    Audio-only baseline pipeline using the Google GenAI SDK directly due to the limitations in langchain-google-genai.

    Uses the Gemini Files API to upload audio before inference, since
    langchain-google-genai does not support audio file URIs. The rest
    of the pipeline structure mirrors VideoPipeline and LanguagePipeline.
    """

    NAME = PipelineNames.AUDIO

    def _load_model_runner(self) -> None:
        _, model = self.configs.model.split(":")
        self._genai_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._genai_model = model

    async def process_inputs(self, dataset: pd.DataFrame) -> List[Dict]:
        """
        Upload audio files to the Gemini Files API and prepare per-question inputs.

        Returns a list of dicts with keys: qid, audio_file, question, options.
        """
        if os.path.exists(self.path_to_output) and os.path.getsize(self.path_to_output) > 0:
            self.logger.info("loading previous results...")
            prev_results = pd.read_json(self.path_to_output, lines=True)
            qids_done = set(prev_results['qid'].unique())
            self.logger.info(f"Skipping {len(qids_done)} already-processed questions.")
        else:
            qids_done = set()

        remaining = dataset[~dataset['qid'].isin(qids_done)]

        if remaining.empty:
            self.logger.info("All questions already processed.")
            return []

        audio_paths = load_audios(
            video_ids=set(remaining['vid_name'].unique()),
            max_workers=self.configs.max_concurrency,
        )

        valid_paths = {vid: (path, mime) for vid, (path, mime) in audio_paths.items() if path}
        self.logger.info(f"Uploading {len(valid_paths)} audio files to Gemini Files API...")

        audio_files = {}
        for vid, (path, mime_type) in tqdm_sync(valid_paths.items(), desc="Uploading audio"):
            try:
                with open(path, "rb") as f:
                    uploaded = self._genai_client.files.upload(
                        file=f,
                        config=google_types.UploadFileConfig(mime_type=mime_type),
                    )
                audio_files[vid] = uploaded
            except Exception as e:
                self.logger.error(f"Failed to upload audio for {vid}: {e}")

        inputs = [
            {
                "qid": row["qid"],
                "audio_file": audio_files[row["vid_name"]],
                "question": row["q"],
                "options": "\n".join([f"{i}: {row[f'a{i}']}" for i in range(4)]),
            }
            for _, row in remaining.iterrows()
            if row["vid_name"] in audio_files
        ]

        if len(inputs) < len(remaining):
            self.logger.warning(f"{len(remaining) - len(inputs)} were missing an audio file and will be skipped.")

        return inputs

    async def _call_model(self, inp: Dict) -> str:
        """Call the Gemini model for a single input, running in a thread executor."""
        loop = asyncio.get_event_loop()

        def _invoke():
            response = self._genai_client.models.generate_content(
                model=self._genai_model,
                contents=[
                    inp["audio_file"],
                    f"QUESTION:\n{inp['question']}\n\nANSWER CHOICES (Choose one):\n{inp['options']}",
                ],
                config=google_types.GenerateContentConfig(
                    system_instruction=AUDIO_SYSTEM_PROMPT,
                    max_output_tokens=2048,
                    temperature=0.0,
                ),
            )
            # Gemini 2.5 Pro uses thinking tokens; extract the final text part
            text = response.text
            if not text and response.candidates:
                candidate = response.candidates[0]
                content = candidate.content
                parts = content.parts if content else []
                text_parts = [p.text for p in (parts or []) if hasattr(p, "text") and p.text]
                text = text_parts[-1] if text_parts else ""
            return text or ""

        for attempt in range(MAX_RETRIES):
            try:
                return await loop.run_in_executor(None, _invoke)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

    async def run_model_on_inputs(self, inputs: List[Dict]):
        """Execute the model on all inputs with bounded concurrency."""
        semaphore = asyncio.Semaphore(self.configs.max_concurrency)

        async def _run(i: int, inp: Dict):
            async with semaphore:
                return i, await self._call_model(inp)

        tasks = [_run(i, inp) for i, inp in enumerate(inputs)]

        results: List[dict] = []
        unsaved: List[dict] = []
        errors = 0

        async for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(inputs),
            desc=f"Running model {self.configs.model}...",
        ):
            i, text = await fut
            self.logger.info(f"[{inputs[i]['qid']}] {text!r}")
            match = re.search(r"[0-3]", text)
            if match:
                entry = {"qid": inputs[i]["qid"], "result": int(match.group())}
                results.append(entry)
                unsaved.append(entry)
            else:
                self.logger.error(f"Could not parse result for {inputs[i]['qid']}: {text!r}")
                errors += 1

            if len(unsaved) >= 10:
                self.write_results_to_json(unsaved)
                unsaved = []

        if unsaved:
            self.write_results_to_json(unsaved)

        self.logger.info(f"Finished processing: {len(inputs)} inputs | errors: {errors}")
        return results
