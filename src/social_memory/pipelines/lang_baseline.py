"""Test zero-shot performance of LLMs on transcript-only data"""
import logging
import os
import sys
import pandas as pd

from argparse import ArgumentParser
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from tqdm.asyncio import tqdm

from social_memory.constants import DEFAULT_MODEL, DEFAULT_MODEL_PROVIDER
from social_memory.prompts import QA_TEMPLATE_TRANSCRIPT
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


async def main():

    logger.info(f"Running {PIPELINE_NAME} pipeline with model {args.model} on {args.split} split...")
    output_dir = os.path.join("results", PIPELINE_NAME, args.model, args.split)
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "results.csv")

    # initialize model(s) client
    llm_with_retry = init_chat_model(args.model).with_retry(wait_exponential_jitter=True)

    # set up concurrency configs 
    config = RunnableConfig(max_concurrency=args.max_concurrency)

    # set up chain
    prompt_template = QA_TEMPLATE_TRANSCRIPT
    chain = prompt_template | llm_with_retry

    # load split of QA dataset into dataframe
    logger.info("loading dataset...")
    dataset = load_qa_dataset(split=args.split)
    
    logger.info(f"loaded. size: {len(dataset)}")
    transcripts = load_transcripts(
        video_ids=dataset['vid_name'].unique(),
        max_workers=args.max_concurrency
    )

    logger.info("preparing inputs...")
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
        logger.warning(f"{len(dataset) - len(inputs)} were missing a transcript and will be skipped.")

    # TODO: make into function
    errors = 0
    results_df = dataset.assign(result=pd.NA).set_index("qid")
    async for i, res in tqdm(
        chain.abatch_as_completed(inputs=inputs, config=config, return_exceptions=True),
        total=len(inputs),
        desc=f"Processing with model {args.model}..."
    ):
        if isinstance(res, AIMessage):
            result = int(res.content)
            results_df.at[inputs[i]["qid"], "result"] = result
        else:
            # if isinstance(review, (Exception, ValueError, ValidationError)):
            logger.error(f"There was an issue with: {inputs[i]['qid']}, error: {res}")
            errors += 1

        if i > 0 and i % 10 == 0:
            correctness = compute_correctness(results_df)
            logger.info(f"Accuracy: {correctness}")

    correctness = compute_correctness(results_df)
    logger.info(f"Finished processing: {len(inputs)} inputs | errors: {errors} | Accuracy: {correctness}")
    logger.info(f"Saving results to: {results_csv_path}")
    results_df.to_csv(results_csv_path)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
