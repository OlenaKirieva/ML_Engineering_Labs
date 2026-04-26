import logging
import pickle
import tarfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore
from torchvision.io import read_image  # type: ignore

logger = logging.getLogger(__name__)


def assign_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Ділить датафрейм на N віртуальних батчів."""
    df = labels_df.copy(deep=True)
    df["batch_name"] = "not_set"

    n_batches = config["data"]["n_batches"]
    batch_size = len(df) // n_batches

    for i in range(n_batches):
        start_idx = i * batch_size
        # Для останнього батча беремо все, що залишилося
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else len(df)
        df.iloc[start_idx:end_idx, df.columns.get_loc("batch_name")] = str(i)  # type: ignore

    return df


def select_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Вибирає лише ті батчі, які вказані в конфігурації."""
    selected = config["data"]["batch_names_select"]
    return labels_df[labels_df["batch_name"].isin(selected)].copy()


def download_and_extract(url: str, save_dir: str) -> str:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    file_path = save_path / filename

    if not file_path.exists():
        logger.info(f"Downloading {url}...")
        response = requests.get(url, timeout=30)
        with open(file_path, "wb") as f:
            f.write(response.content)
        if filename.endswith(".tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(save_path)
            file_path.unlink()
    return str(save_path)


def cifar_to_jpg(cifar_dir: str, output_dir: str) -> pd.DataFrame:
    """Конвертує CIFAR у JPG та повертає ПОВНИЙ датафрейм (60к зображень)."""
    cifar_path = Path(cifar_dir) / "cifar-10-batches-py"
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Беремо всі батчі CIFAR для створення загальної бази
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    data_records = []

    for batch_file in batch_files:
        with open(cifar_path / batch_file, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            images = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = entry["labels"]
            filenames = entry["filenames"]
            for i in range(len(images)):
                img_name = f"{batch_file}_{filenames[i]}"
                if not img_name.endswith(".jpg"):
                    img_name += ".jpg"
                full_img_path = img_dir / img_name
                if not full_img_path.exists():
                    Image.fromarray(images[i]).save(full_img_path)
                data_records.append(
                    {"image_path": str(full_img_path.absolute()), "label": labels[i]}
                )
    return pd.DataFrame(data_records)


def train_test_split(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = data.sample(frac=1, random_state=random_state)
    n_test = int(len(data) * test_size)
    return shuffled.iloc[n_test:], shuffled.iloc[:n_test]


def prepare_data_pipelines(
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Реалізація Частини 1: Динамічне об'єднання пакетів."""
    raw_dir = download_and_extract(config["data"]["url"], config["data"]["raw_dir"])
    full_df = cifar_to_jpg(raw_dir, config["data"]["processed_dir"])

    # 1. Відділяємо СТАТИЧНИЙ тест (Part 1 завдання: статичний набір тестів)
    train_val_full, test_df = train_test_split(
        full_df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 2. Ділимо решту даних на ВІРТУАЛЬНІ БАТЧІ
    train_val_batches = assign_batches(train_val_full, config)

    # 3. Вибираємо пакети згідно з конфігурацією (Part 1 завдання: динамічні набори)
    selected_df = select_batches(train_val_batches, config)

    # 4. Розділяємо вибрані дані на Train та Val
    train_df, val_df = train_test_split(
        selected_df,
        test_size=config["data"]["val_size"],
        random_state=config["data"]["random_state"],
    )

    logger.info(f"Batches selected: {config['data']['batch_names_select']}")
    logger.info(
        f"Final sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


class CIFARDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Any) -> None:
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["label"]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, int(label)


def create_data_loader(
    df: pd.DataFrame, config: Dict[str, Any], is_train: bool = True
) -> DataLoader:
    # Вибираємо трансформацію
    if is_train:
        # Аугментація для навчання (використовуємо навички з Lab 1)
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # Лише базова обробка для валідації/тесту
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    dataset = CIFARDataset(df, transform=transform)
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=is_train,
        num_workers=0,
    )
