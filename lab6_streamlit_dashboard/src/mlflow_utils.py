import logging
from pathlib import Path
from typing import Any, List, Optional, cast

import mlflow  # type: ignore
import pandas as pd
import streamlit as st
import torch

from src.model import SimpleCNN
from src.model_new import CIFAR10ProCNN

logger = logging.getLogger(__name__)


def init_mlflow(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow initialized: {tracking_uri}")


def get_all_experiments() -> List[Any]:
    return mlflow.search_experiments()


@st.cache_resource(show_spinner="Searching for model weights...")
def load_model_smart(
    run_id: str, experiment_id: str, run_name: str, tracking_uri: str
) -> Optional[torch.nn.Module]:
    logger.info(f"--- Global Search for run: {run_name} ---")

    try:
        clean_uri = tracking_uri.replace("sqlite:///", "").replace("file:///", "")
        if clean_uri.startswith("/") and clean_uri[2] == ":":
            clean_uri = clean_uri[1:]
        base_dir = Path(clean_uri).parent

        # НОВА ЛОГІКА ПОШУКУ (враховуючи папку 'models')
        models_dir = base_dir / "mlruns" / "models"

        weights_path = None

        # 1. Шукаємо в папці models (скануємо всі підпапки)
        if models_dir.exists():
            for model_folder in models_dir.iterdir():
                # Шлях до файлу всередині вашої структури
                potential_path = model_folder / "artifacts" / "data" / "model.pth"
                if potential_path.exists():
                    # Перевіряємо meta.yaml у цій папці, чи належить вона нашому run_id
                    meta_path = model_folder / "artifacts" / "MLmodel"
                    if meta_path.exists():
                        with open(meta_path, "r") as f:
                            content = f.read()
                            if run_id in content:
                                weights_path = potential_path
                                logger.info(
                                    f"🎯 Match found in models folder: {weights_path}"
                                )
                                break

        # 2. Якщо не знайшли в models, перевіряємо інші стандартні місця
        if not weights_path:
            fallback_paths = [
                base_dir
                / "mlruns"
                / str(experiment_id)
                / str(run_id)
                / "artifacts"
                / "model"
                / "data"
                / "model.pth",
                base_dir / "artifacts" / "model.pth",
            ]
            for p in fallback_paths:
                if p.exists():
                    weights_path = p
                    break

        if not weights_path:
            logger.error(f"❌ Could not find model.pth for run {run_id}")
            return None

        # Створюємо модель
        model = (
            CIFAR10ProCNN(n_classes=10)
            if "Pro" in str(run_name)
            else SimpleCNN(n_classes=10)
        )

        # Завантаження
        torch.serialization.add_safe_globals([CIFAR10ProCNN, SimpleCNN])
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = cast(Any, checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        logger.info(f"✅ Successfully loaded weights from {weights_path}")
        return model

    except Exception as e:
        logger.error(f"💥 Critical error during smart load: {e}")
        return None


def load_predictions_from_mlflow(
    run_id: str, tracking_uri: str, experiment_id: str
) -> Optional[pd.DataFrame]:
    """Надійний пошук CSV файлу з прогнозами."""
    try:
        # Очищуємо шлях
        clean_uri = tracking_uri.replace("sqlite:///", "").replace("file:///", "")
        if clean_uri.startswith("/") and clean_uri[2] == ":":
            clean_uri = clean_uri[1:]
        base_dir = Path(clean_uri).parent

        # Список можливих папок, де може бути CSV
        search_dirs = [
            base_dir
            / "mlruns"
            / str(experiment_id)
            / str(run_id)
            / "artifacts"
            / "predictions",
            base_dir / "mlruns" / str(experiment_id) / str(run_id) / "artifacts",
            base_dir / "artifacts",  # Резерв (якщо ви скачали файл окремо)
        ]

        for pred_dir in search_dirs:
            if pred_dir.exists():
                # Шукаємо будь-який файл, що закінчується на .csv
                csv_files = list(pred_dir.glob("*.csv"))
                if csv_files:
                    return pd.read_csv(csv_files[0])
        return None
    except Exception:
        return None

def load_artifact_text(
    run_id: str, experiment_id: str, tracking_uri: str, filename: str
) -> Optional[str]:
    """Зчитує текстовий звіт (Part 3)."""
    try:
        clean_uri = tracking_uri.replace("sqlite:///", "").replace("file:///", "")
        if clean_uri.startswith("/") and clean_uri[2] == ":":
            clean_uri = clean_uri[1:]
        file_path = (
            Path(clean_uri).parent
            / "mlruns"
            / str(experiment_id)
            / str(run_id)
            / "artifacts"
            / filename
        )
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    except Exception:
        return None
