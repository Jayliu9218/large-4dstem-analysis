from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_workflow_config(path: str | Path = "configs/default_workflow.yaml") -> dict[str, Any]:
    """Load the YAML workflow configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)
