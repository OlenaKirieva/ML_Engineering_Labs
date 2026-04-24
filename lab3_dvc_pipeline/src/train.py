import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore
from torchvision.io import read_image  # type: ignore

from src.model import SimpleCNN
from src.utils import load_params, setup_logging

setup_logging()
logger = logging.getLogger("train")


class CIFARDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = int(self.df.iloc[idx]["label"])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    params = load_params()
    device = torch.device(params["train"]["device"])

    # Завантаження даних
    train_df = pd.read_csv(Path(params["data"]["processed_dir"]) / "train.csv")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        CIFARDataset(train_df, transform),
        batch_size=params["train"]["batch_size"],
        shuffle=True,
    )

    model = SimpleCNN(n_classes=params["model"]["n_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["train"]["lr"])

    logger.info("Starting training stage...")
    for epoch in range(params["train"]["num_epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        logger.info(f"Epoch {epoch+1}/{params['train']['num_epochs']} finished")

    # Збереження артефакту
    save_path = Path(params["model"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
