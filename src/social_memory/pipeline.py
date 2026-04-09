"""Test zero-shot performance of LLMs on transcript-only data"""

import pandas as pd

from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
load_dotenv()

from social_memory.prompts import QA_TEMPLATE_TRANSCRIPT
from social_memory.utils import init_model, load_qa_dataset, load_transcripts


MODEL_NAME = "gpt-5-mini"
MODEL_PROVIDER = "openai"

SPLIT: str = "demo"
MAX_CONCURRENCY: int = 4


async def main():

    # initialize model(s) client
    llm = init_model(
        MODEL_NAME,
        provider=MODEL_PROVIDER,
        api_key_provider="openai",
    )
    # set up concurrency configs 
    config = RunnableConfig(max_concurrency=MAX_CONCURRENCY)

    qa_template = QA_TEMPLATE_TRANSCRIPT

    chain = qa_template | llm

    # load split qa json into dataframe
    dataset = load_qa_dataset(split=SPLIT)
    transcripts = load_transcripts(video_ids=dataset['vid_name'].unique().to_list(), max_workers=MAX_CONCURRENCY)

    inputs = [
        {
            "transcript": transcripts.get(row['video_id']),
            "question": row["q"],
            "options": ",".join([f"{col}: {row[col]}" for col in ["a0", "a1", "a2", "a3"]])
        }
        for i, row in dataset.iterrows()
        if row['video_id'] in transcripts
    ]

    results = []
    # iterate over split qa dataset in batches of "batch_size"
    async for i, res in chain.abatch_as_completed(
        inputs=inputs,
        config=config,
        return_exceptions=True
    ):
        results.append((i, res))
    
    results = pd.DataFrame(results, columns=["index", "result"]).set_index("index")
