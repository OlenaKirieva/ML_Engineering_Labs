import json
from pathlib import Path
from venv import logger

import pandas as pd
import torch
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore
from torchvision.io import read_image  # type: ignore

from src.model import SimpleCNN
from src.utils import load_params, setup_logging

setup_logging()


class CIFARDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = read_image(self.df.iloc[idx]["image_path"])
        if self.transform:
            image = self.transform(image)
        return image, int(self.df.iloc[idx]["label"])


def main():
    params = load_params()
    device = torch.device(params["train"]["device"])

    # 1. Load Model
    model = SimpleCNN(n_classes=params["model"]["n_classes"]).to(device)
    model.load_state_dict(torch.load(params["model"]["save_path"]))
    model.eval()

    # 2. Load Static Test Data
    test_df = pd.read_csv(Path(params["data"]["processed_dir"]) / "test.csv")
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_loader = DataLoader(
        CIFARDataset(test_df, transform), batch_size=params["train"]["batch_size"]
    )

    # 3. Evaluation
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
    }

    # 5. Save Metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("Evaluation complete. Metrics saved to metrics.json") 

if __name__ == "__main__":
    main()
