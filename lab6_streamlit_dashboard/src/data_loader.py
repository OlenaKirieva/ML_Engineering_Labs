import logging
import os
import pickle
import tarfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)


def prepare_test_data_locally():
    """Завантажує та готує 10,000 тест-картинок."""
    save_dir = Path("data_cache")
    img_dir = save_dir / "test_images"
    registry_path = save_dir / "test_registry.csv"

    if registry_path.exists():
        return pd.read_csv(registry_path)

    st.info("📦 First launch: Preparing 10,000 images...")
    save_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    targz_path = save_dir / "cifar.tar.gz"

    response = requests.get(url, stream=True)
    with open(targz_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with tarfile.open(targz_path, "r:gz") as tar:
        member = tar.getmember("cifar-10-batches-py/test_batch")
        member.name = os.path.basename(member.name)
        tar.extract(member, path=save_dir)

    targz_path.unlink()

    with open(save_dir / "test_batch", "rb") as f:
        entry = pickle.load(f, encoding="latin1")
        images = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        filenames = entry["filenames"]
        labels = entry["labels"]

    data_records = []
    p_bar = st.progress(0)
    for i in range(len(images)):
        img_name = f"{filenames[i]}.jpg"
        full_path = img_dir / img_name
        Image.fromarray(images[i]).save(full_path)
        data_records.append({"image_path": str(full_path), "label": labels[i]})
        if i % 1000 == 0:
            p_bar.progress((i + 1) / 10000)

    df = pd.DataFrame(data_records)
    df.to_csv(registry_path, index=False)
    st.success("✅ 10,000 images ready!")
    return df


def load_image(image_path: str):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None
