"""Base pipeline module"""
import os
import sys
import logging

from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, List

from social_memory.constants import RESULTS_DIR
from social_memory.prompts import PROMPT_REGISTRY
from social_memory.utils import load_qa_dataset


class PipelineConfig(BaseModel):
    """Pipeline configurations"""

    model: str  # in the form of {model_provider}:{model}
    split: str
    max_concurrency: int = 4
    debug: bool = False

    def to_yaml(self, path: str) -> None:
        """Writes config object to yaml file"""
        import yaml

        # Convert the config to a dict using Pydantic's model_dump()
        config_dict = self.model_dump()
        # Write to the given YAML path
        with open(path, "x") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)


    @staticmethod
    def from_yaml(path: str) -> "PipelineConfig":
        """Reads config from a yaml file and returns a PipelineConfig object"""
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return PipelineConfig(**config_dict)


class Pipeline(ABC):
    """Initialize the Pipeline with models, dataset, logger, and configurations."""

    NAME: str = ""

    def __init__(self, configs: PipelineConfig) -> None:
        self.configs = configs
        self.logger: logging.Logger = self._create_logger()
        self.run_id = self._generate_run_id()
        self.path_to_output = self._create_output_file_path()
        self.model_runner = None
        self.prompt_template = None

        self._load_prompt_template()  # updates self.prompt_template
        self._load_model_runner()  # updates self.model_runner


    def __repr__(self) -> str:
        """String representation for CLI or logging, similar to _repr_html_."""
        lines = [
            f"{self.__class__.__name__}(",
            f"  Name: {self.NAME or ''}",
            f"  Model: {self.configs.model}",
            f"  Split: {self.configs.split}",
            f"  Run ID: {self.run_id}",
            f"  Output path: {self.path_to_output}",
        ]
        for k, v in self.configs.model_dump().items():
            lines.append(f"    {k}: {v!r}")
        lines.append(")")
        return "\n".join(lines)


    def _repr_html_(self) -> str:
        """Rich HTML repr for Jupyter notebooks."""
        rows = [
            # ("Class", self.__class__.__name__),
            ("Name", self.NAME or ""),
            ("Model", self.configs.model),
            ("Split", self.configs.split),
            ("Run ID", self.run_id),
            ("Output path", self.path_to_output),
        ]

        # Flatten configs nicely
        cfg_html = "<ul>" + "".join(
            f"<li><code>{k}</code>: {v!r}</li>" for k, v in self.configs.model_dump().items()
        ) + "</ul>"
        rows.append(("Configs", cfg_html))

        rows_html = "".join(
            f"<tr><th style='text-align:left;padding-right:1em'>{name}</th>"
            f"<td>{value}</td></tr>"
            for name, value in rows
        )

        return f"""
        <table style="border-collapse:collapse; border:1px solid #ddd; font-family:monospace; font-size:13px;">
          <tbody>
            {rows_html}
          </tbody>
        </table>
        """


    def _generate_run_id(self) -> str:
        """Generate run id for pipeline."""
        return str(datetime.timestamp(datetime.now()))


    def _create_logger(self) -> logging.Logger:
        """Create logger for pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    f'logs/{self.NAME}_{self.configs.model.replace(":", "_")}_{self.configs.split}.log'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        return logger


    def _create_output_file_path(self, create_if_not_existent: bool = True) -> str:
        """Create output file path based on model, split, pipeline name."""        
        path = os.path.join(
            RESULTS_DIR,
            self.NAME,
            self.configs.model,
            self.configs.split,
            f"results.jsonl",
        )
        if create_if_not_existent:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


    def _load_prompt_template(self) -> None:
        """Load prompt template from registry given pipeline"""
        
        self.prompt_template = PROMPT_REGISTRY.get(self.NAME)
        
    def write_results_to_json(self, results: List[dict]) -> None:
        """Write model results to disk as a JSONL file at self.path_to_output"""
        import json
        from tqdm import tqdm

        mode = "a" if os.path.exists(self.path_to_output) else "w"
        with open(self.path_to_output, mode, encoding="utf-8") as f:
            for result in tqdm(results, desc="Writing results"):
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")          

    @abstractmethod
    async def _load_model_runner(self) -> None:
        """
        * SHOULD BE IMPLEMENTED BY INHERITING CLASS *

        Adds a self.model_runner property to the pipeline, 
        which should be a langchain Runnable supporting `abatch_as_completed()`.
        """
        raise NotImplementedError


    @abstractmethod
    async def process_inputs(self, dataset) -> List[Dict[str, str]]:
        """
        * SHOULD BE IMPLEMENTED BY INHERITING CLASS *

        Custom inputs preprocessing. Should return an Iterable of inputs.
        """
        raise NotImplementedError


    @abstractmethod
    async def run_model_on_inputs(self, inputs: List[Dict[str, str]]):
        """Should be implemented by inheriting classes"""
        raise NotImplementedError


    async def run(self) -> None:
        """Run pipeline"""

        self.logger.info("Starting pipeline...")
        start_time = datetime.now()

        # load split of QA dataset into dataframe
        self.logger.info("loading dataset...")
        dataset = load_qa_dataset(split=self.configs.split)
        
        self.logger.info("Preparing inputs...")
        inputs = await self.process_inputs(dataset)

        self.logger.info(f"Processing {len(inputs)} documents.")
        results: List[dict] = await self.run_model_on_inputs(inputs)

        self.logger.info(f"Writing results to {self.path_to_output}")
        self.write_results_to_json(results)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Finished processing {len(inputs)} documents in {elapsed_time:.1f}s.")
    