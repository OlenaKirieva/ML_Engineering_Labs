import logging
from pathlib import Path
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DESCRIPTIONS = {
    "SimpleCNN": {
        "layers": 2,
        "params": "~50k",
        "description": "Basic convolutional network. Two Conv2D + MaxPooling blocks. Designed to detect simple patterns.",
        "features": ["Dropout 0.3"],
    },
    "ProCNN_8Layers_Fast": {
        "layers": 8,
        "params": "~1.2M",
        "description": "Deep VGG‑style architecture. Four blocks with two convolutions each. High capacity for complex visual features.",
        "features": ["Batch Normalization", "Agressive Augmentation"],
    },
    "ProCNN_8Layers_Optimal": {
        "layers": 8,
        "params": "~1.2M",
        "description": "Deep VGG‑style architecture.  Four blocks with two convolutions each. Utilizes a Scheduler for fine‑tuning weights at later training stages.",
        "features": [
            "Batch Normalization",
            "Learning Rate Scheduler",
            "Early Stopping",
        ],
    },
}


def setup_logging() -> None:
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler = logging.FileHandler(
            str(BASE_DIR / "dashboard.log"), encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config(config_path: str = "config/dashboard_config.yaml") -> Dict[str, Any]:
    with open(BASE_DIR / config_path, "r") as f:
        return yaml.safe_load(f)


def get_absolute_path(relative_path: str) -> str:
    if ":" in relative_path or relative_path.startswith("/"):
        return relative_path
    return str((BASE_DIR / relative_path).resolve())


# import logging
# import sys
# from pathlib import Path
# from typing import Any, Dict

# import yaml

# # Шлях до кореня лаби 6
# BASE_DIR = Path(__file__).resolve().parent.parent


# def setup_logging() -> None:
#     """Налаштування логування (Part 5 [Task] Logging)."""
#     # Створюємо логер
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

#     # Очищуємо старі обробники, щоб не було дублів
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

#     # Вивід у консоль
#     stdout_handler = logging.StreamHandler(sys.stdout)
#     stdout_handler.setFormatter(formatter)

#     # Вивід у файл
#     file_handler = logging.FileHandler(
#         str(BASE_DIR / "dashboard.log"), encoding="utf-8"
#     )
#     file_handler.setFormatter(formatter)

#     logger.addHandler(stdout_handler)
#     logger.addHandler(file_handler)

#     logging.info("Logging system initialized.")


# def load_config(config_path: str = "config/dashboard_config.yaml") -> Dict[str, Any]:
#     full_path = BASE_DIR / config_path
#     with open(full_path, "r") as f:
#         return yaml.safe_load(f)


# def get_absolute_path(relative_path: str) -> str:
#     if ":" in relative_path or relative_path.startswith("/"):
#         return relative_path
#     return str((BASE_DIR / relative_path).resolve())

# MODEL_DESCRIPTIONS = {
#     "SimpleCNN": {
#         "layers": 2,
#         "params": "~50k",
#         "description": "Basic convolutional network. Two Conv2D + MaxPooling blocks. Designed to detect simple patterns.",
#         "features": ["Dropout 0.3"]
#     },
#     "ProCNN_8Layers_Fast": {
#         "layers": 8,
#         "params": "~1.2M",
#         "description": "Deep VGG‑style architecture. Four blocks with two convolutions each. High capacity for complex visual features.",
#         "features": ["Batch Normalization", "Agressive Augmentation"]
#     },
#     "ProCNN_8Layers_Optimal": {
#         "layers": 8,
#         "params": "~1.2M",
#         "description": "Deep VGG‑style architecture.  Four blocks with two convolutions each. Utilizes a Scheduler for fine‑tuning weights at later training stages.",
#         "features": ["Batch Normalization", "Learning Rate Scheduler", "Early Stopping"]
#     }
# }
# MODEL_DESCRIPTIONS = {
#     "SimpleCNN": {
#         "layers": 2,
#         "params": "~50k",
#         "description": "Baseline архітектура. 2 блоки Conv2D. Використана для порівняння ефективності глибини мережі.",
#         "features": ["Dropout 0.3", "ReLU"]
#     },
#     "ProCNN_8Layers_Fast": {
#         "layers": 8,
#         "params": "~1.2M",
#         "description": "Глибока архітектура (8 шарів). Навчена за 15 епох. Демонструє швидку конвергенцію завдячуючи BatchNorm.",
#         "features": ["Batch Normalization", "Global GPU Training"]
#     },
#     "ProCNN_8Layers_Optimal": {
#         "layers": 8,
#         "params": "~1.2M",
#         "description": "Найкраща модель. 100 епох навчання. Використовує Scheduler для тонкої настройки ваг на пізніх етапах.",
#         "features": ["LR Scheduler", "Early Stopping", "Max Accuracy"]
#     }
# }
