import logging
import pickle
import tarfile
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from src.utils import load_params, setup_logging

setup_logging()
logger = logging.getLogger("data_load")


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
    cifar_path = Path(cifar_dir) / "cifar-10-batches-py"
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    data_records = []

    logger.info("Converting CIFAR to JPG...")
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
                    {
                        "image_path": str(full_img_path.absolute()),
                        "label": labels[i],
                        "batch_origin": batch_file,
                    }
                )
    return pd.DataFrame(data_records)


def main():
    params = load_params()

    # 1. Download & Extract
    raw_dir = download_and_extract(params["data"]["url"], params["data"]["raw_dir"])

    # 2. Convert & Create Registry Table
    full_df = cifar_to_jpg(raw_dir, params["data"]["processed_dir"])

    # 3. Dynamic Splitting (Using Lab 2 logic via params)
    # Тест завжди статичний
    test_df = full_df[full_df["batch_origin"] == "test_batch"]
    train_val_pool = full_df[full_df["batch_origin"] != "test_batch"]

    # Вибір батчів для навчання з params.yaml
    selected_batches = [f"data_batch_{i}" for i in params["data"]["batch_names_select"]]
    train_val_selected = train_val_pool[
        train_val_pool["batch_origin"].isin(selected_batches)
    ]

    # Розподіл Train/Val
    train_df = train_val_selected.sample(
        frac=1 - params["data"]["val_size"], random_state=params["data"]["random_state"]
    )
    val_df = train_val_selected.drop(train_df.index)

    # 4. Save CSV artifacts (Outputs for DVC)
    processed_path = Path(params["data"]["processed_dir"])
    processed_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_path / "train.csv", index=False)
    val_df.to_csv(processed_path / "val.csv", index=False)
    test_df.to_csv(processed_path / "test.csv", index=False)

    logger.info(f"Data ingestion complete. Saved to {processed_path}")


if __name__ == "__main__":
    main()
