import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Шлях до кореня лаби 6
BASE_DIR = Path(__file__).resolve().parent.parent


def setup_logging() -> None:
    """Налаштування логування (Part 5 [Task] Logging)."""
    # Створюємо логер
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Очищуємо старі обробники, щоб не було дублів
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Вивід у консоль
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    # Вивід у файл
    file_handler = logging.FileHandler(
        str(BASE_DIR / "dashboard.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    logging.info("Logging system initialized.")


def load_config(config_path: str = "config/dashboard_config.yaml") -> Dict[str, Any]:
    full_path = BASE_DIR / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def get_absolute_path(relative_path: str) -> str:
    if ":" in relative_path or relative_path.startswith("/"):
        return relative_path
    return str((BASE_DIR / relative_path).resolve())


# def get_absolute_path(relative_path: str) -> str:
#     REPO_ROOT = BASE_DIR.parent
#     path_obj = (BASE_DIR / relative_path).resolve()
#     return str(path_obj)
