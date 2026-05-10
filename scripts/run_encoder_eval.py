

import sys
import json
import pandas as pd
from pathlib import Path


# Repo uses a src/ layout; running `python scripts/run_pipeline.py` does not put `src` on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_src = _REPO_ROOT / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

from argparse import ArgumentParser
from dotenv import load_dotenv
load_dotenv()

from social_memory.encoder import XCLIPEncoder
from social_memory.pipelines.evaluate_encoder import evaluate_encoder
from social_memory.utils import load_qa_dataset

parser = ArgumentParser()
parser.add_argument(
    "--encoder",
    type= str
)
parser.add_argument(
    "--output_path",
    type=str,
)
parser.add_argument(
    "--split",
    type=str
)

args = parser.parse_args()

def main():

    
    df = pd.read_csv("datasets/siq2long/dataset.csv")
    qids = load_qa_dataset(dataset="siq2long", split=args.split)["qid"]
    df = df[df["qid"].isin(qids.unique())]
    df["chunk_ids"] = df["chunk_ids"].apply(json.loads)
    inputs = df.to_dict(orient="records")
    print(f"len of dataset: {len(inputs)}")

    if args.encoder == "xclip":
        encoder = XCLIPEncoder()
    
    results = evaluate_encoder(
        inputs=inputs,
        encoder=encoder,
        output_path=args.output_path,
    )
    print("~Finished!~")

if __name__ == "__main__":
    main()