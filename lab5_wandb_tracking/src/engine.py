import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (  # type: ignore
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

import wandb

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> Path:
    logger.info(">>> Training started with W&B tracking.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    save_path = Path("artifacts/model_wandb.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Валідація
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = correct / total
        scheduler.step(val_acc)

        # ЛОГУВАННЯ У W&B
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return save_path


def evaluate_and_log_to_wandb(model, test_loader, device, model_path):
    model.eval()
    all_preds = []
    all_labels = []
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Текстовий звіт (classification report)
    report_text = classification_report(
        all_labels, all_preds, target_names=classes, zero_division=0
    )

    # ЛОГУЄМО ЯК ТЕКСТОВИЙ БЛОК (з'явиться в Charts)
    wandb.log({"classification_report": wandb.Html(f"<pre>{report_text}</pre>")})

    # 2. Confusion Matrix як картинка
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    # 3. ЗБЕРЕЖЕННЯ ЗВІТУ ТА МОДЕЛІ ЯК АРТЕФАКТІВ (Без wandb.save)
    # Створюємо один артефакт для результатів
    results_art = wandb.Artifact(name="final_results", type="evaluation")

    # Записуємо текст у файл локально
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # Додаємо файли в артефакт (це просто копіювання, воно працює без прав адміна)
    results_art.add_file(report_path)
    results_art.add_file(str(model_path))

    wandb.log_artifact(results_art)

    print("✅ Done! Report and Model uploaded to W&B Artifacts.")
