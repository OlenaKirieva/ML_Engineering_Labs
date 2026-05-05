import time
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (  # type: ignore
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> Path:
    logger.info(">>> Training process started.")
    criterion = nn.CrossEntropyLoss()

    # Регуляризація
    optimizer = optim.Adam(
        model.parameters(), lr=config["training"]["lr"], weight_decay=1e-4
    )

    # Ініціалізація планувальника
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    save_path = Path("artifacts/model.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    num_epochs = config["training"]["num_epochs"]
    
    # --- НАЛАШТУВАННЯ EARLY STOPPING ---
    early_stop_patience = 10  # Скільки епох чекаємо без покращень
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()  # Початок таймера епохи
        
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        epoch_duration = time.time() - start_time  # Кінець таймера

        # Крок планувальника (зменшує LR, якщо Val Acc не росте)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # ЛОГУВАННЯ В MLFLOW
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        mlflow.log_metric("epoch_duration_sec", epoch_duration, step=epoch) # Зберігаємо час

        msg = f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s"
        print(msg)
        logger.info(msg)

        # ЛОГІКА РАННЬОЇ ЗУПИНКИ ТА ЗБЕРЕЖЕННЯ
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved with Acc: {val_acc:.4f}")
            
            epochs_no_improve = 0  # Скидаємо лічильник
            mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= early_stop_patience:
            stop_msg = f"🛑 Early stopping triggered at epoch {epoch+1}. Model reached optimal state."
            print(stop_msg)
            logger.info(stop_msg)
            break  # Примусово виходимо з циклу

    return save_path

def evaluate_and_log_artifacts(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> None:
    """Оцінка на тестовому наборі та логування Confusion Matrix в MLflow (Part 3)"""
    logger.info(">>> Running final evaluation...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Classification Report в лог
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
    report = classification_report(all_labels, all_preds, target_names=classes)
    logger.info(f"Final Report:\n{report}")

    # Зберігаємо у файл і вантажимо в MLflow як артефакт
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation="vertical")
    plt.title("Confusion Matrix")

    # Зберігаємо тимчасово та вантажимо в MLflow
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close(fig)

    # 3. Реєструємо саму модель
    mlflow.pytorch.log_model(model, "model")
    logger.info("Model and Confusion Matrix logged as artifacts.")
