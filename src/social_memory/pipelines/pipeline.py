# adapted from my SciBench project


import logging

from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field



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

        self.load_agent_runner()

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


    @abstractmethod
    def load_agent_runner(self) -> None:
        """Should be implemented by inheriting classes"""
        raise NotImplementedError
    