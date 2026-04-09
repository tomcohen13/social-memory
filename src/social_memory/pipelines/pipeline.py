# adapted from my SciBench project


from datetime import datetime
import logging

from abc import ABC, abstractmethod
import os
from pathlib import Path
import sys
from pydantic import BaseModel, Field

from social_memory.constants import RESULTS_DIR


class PipelineConfig(BaseModel):
    """Pipeline configurations"""

    model: str  # in the form of {model_provider}:{model}
    split: str
    max_concurrency: int = 4
    debug: bool = False

    path_to_sys_prompt: str
    prompt: str = Field(default_factory=lambda data: Path(data['path_to_sys_prompt']).read_text())


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

    def __init__(
        self,
        configs: PipelineConfig,
        logger: logging.Logger = None,
    ) -> None:

        self.configs = configs
        self.logger: logging.Logger = logger or self._create_logger()
        self.run_id = self.generate_run_id()
        self.path_to_output = self._create_output_file_path()

        self.load_model_runner()

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

    def generate_run_id(self) -> str:
        """Generate run id for pipeline."""
        return str(datetime.timestamp(datetime.now()))

    def _create_logger(self) -> logging.Logger:
        """Create logger for pipeline."""
        
        print(self.configs)
        print()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    f'logs/{self.configs.dataset}_{self.NAME}_{self.configs.model.replace(":", "_")}.log'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        return logger
    
    def _create_output_file_path(self, create_if_not_existent: bool = True) -> str:
        """Create output file path based on dataset, kind, and run ID."""
        
        path = os.path.join(
            RESULTS_DIR,
            self.name,
            self.configs.model,
            self.configs.split,
            f"{self.run_id}.csv",
        )
        if create_if_not_existent:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    
    @abstractmethod
    async def load_model_runner(self) -> None:
        """
        * SHOULD BE IMPLEMENTED BY INHERITING CLASS *

        Adds a self.model_runner property to the pipeline, 
        which should be a langchain Runnable supporting `abatch_as_completed()`.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def process_inputs(self):
        """
        * SHOULD BE IMPLEMENTED BY INHERITING CLASS *

        Custom inputs preprocessing. Should return an Iterable of inputs.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def run_model_on_inputs(self, inputs):
        """Should be implemented by inheriting classes"""
        raise NotImplementedError
    
    async def run(self) -> None:
        """Run pipeline"""

        self.logger.info("Starting pipeline...")
        start_time = datetime.now()

        self.logger.info("Preparing inputs...")
        inputs = self.prepare_inputs()

        self.logger.info(f"Processing {len(inputs)} documents.")
        await self.arun_agent_on_inputs(inputs)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Finished processing {len(inputs)} documents in {elapsed_time:.1f}s.")
    