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
    default=4,
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

    model_provider, model = args.model.split(":")

    # initialize model(s) client
    llm = init_chat_model(f"{model_provider}:{model}")

    # set up concurrency configs 
    config = RunnableConfig(max_concurrency=args.max_concurrency)

    # set up chain
    prompt_template = QA_TEMPLATE_TRANSCRIPT
    chain = prompt_template | llm

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
            "options": ",".join([f"{col}: {row[col]}" for col in ["a0", "a1", "a2", "a3"]])
        }
        for i, row in dataset.iterrows()
        if transcripts.get(row['vid_name']) != ""
    ]

    # TODO: make into function
    errors = 0
    results = []
    async for i, res in tqdm(
        chain.abatch_as_completed(inputs=inputs, config=config, return_exceptions=True),
        total=len(inputs),
        desc=f"Processing with model {args.model}..."
    ):
        if isinstance(res, AIMessage):
            result = int(res.content)

        else:
            # if isinstance(review, (Exception, ValueError, ValidationError)):
            errors += 1
            logger.error(f"Seems like there was an issue: {inputs[i]['qid']}, error: {res}")
            result = "error"

        results.append((inputs[i]["qid"], result))

    logger.info(f"processed: {len(inputs)} queries | errors: {errors}")
    # convert into dataframe
    results_df = pd.DataFrame(results, columns=["qid", "result"])
    dataset_with_results = dataset.merge(results_df, how="left", on="qid")

    correctness = compute_correctness(dataset_with_results)
    logger.info(f"Finished running. Accuracy: {correctness}")

    # Save results under results/<model>/<split>/ with benchmark accuracy in a CSV
    output_dir = os.path.join("results", args.model, args.split)
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "results.csv")
    metrics_path = os.path.join(output_dir, "metrics.txt")

    dataset_with_results.to_csv(results_csv_path, index=False)

    with open(metrics_path, "w") as f:
        f.write(f"accuracy: {correctness}\n")

    logger.info(f"Saving results to: {results_csv_path}")
    logger.info(f"Saving accuracy to: {metrics_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
