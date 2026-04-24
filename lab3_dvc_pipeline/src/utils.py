import logging
import sys
from typing import Any, Dict

import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log"),
        ],
    )
