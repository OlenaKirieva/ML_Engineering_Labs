import logging
import pickle
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # type: ignore
from torchvision.io import read_image # type: ignore

logger = logging.getLogger(__name__)

def download_and_extract(url: str, save_dir: str) -> str:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    file_path = save_path / filename

    if not file_path.exists():
        logger.info(f"Downloading {url}...")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        if filename.endswith(".tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(save_path)
            file_path.unlink()
    return str(save_path)

def cifar_to_jpg(cifar_dir: str, output_dir: str) -> pd.DataFrame:
    cifar_path = Path(cifar_dir) / "cifar-10-batches-py"
    output_path = Path(output_dir)
    img_dir = output_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    data_records = []

    logger.info("Converting CIFAR binary to JPG...")
    for batch_file in batch_files:
        with open(cifar_path / batch_file, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            images = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = entry["labels"]
            filenames = entry["filenames"]
            for i in range(len(images)):
                img_name = filenames[i] if filenames[i].endswith(".jpg") else f"{filenames[i]}.jpg"
                full_img_path = img_dir / f"{batch_file}_{img_name}"
                if not full_img_path.exists():
                    Image.fromarray(images[i]).save(full_img_path)
                data_records.append({"image_path": str(full_img_path.absolute()), "label": labels[i]})
    return pd.DataFrame(data_records)

def train_test_split(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_test = int(len(data) * test_size)
    shuffled = data.sample(frac=1, random_state=42)
    return shuffled.iloc[n_test:], shuffled.iloc[:n_test]

class CIFARImageDataset(Dataset):
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
        return image, label

def get_transforms(is_train: bool = True) -> Any:
    # BASELINE: Тільки базові трансформації
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_data_loader(df: pd.DataFrame, config: Dict[str, Any], is_train: bool = True) -> DataLoader:
    transform = get_transforms(is_train=is_train)
    dataset = CIFARImageDataset(df, transform=transform)
    return DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=is_train, num_workers=0)

def process_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = download_and_extract(config["data"]["url"], config["data"]["raw_dir"])
    full_df = cifar_to_jpg(raw_dir, config["data"]["processed_dir"])
    train_df, test_df = train_test_split(full_df, test_size=config["data"]["test_size"])
    train_df, val_df = train_test_split(train_df, test_size=config["data"]["val_size"])
    return train_df, val_df, test_df