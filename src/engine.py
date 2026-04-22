import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def save_metrics_plot(train_losses: List[float], val_accs: List[float], save_path: str) -> None:
    """Побудова та збереження графіків втрат та точності."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Графік Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Графік Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Metrics plot saved to {save_path}")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[Path, List[float], List[float]]:
    logger.info(">>> Training process started.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    save_path = Path(config['model']['save_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    for epoch in range(config['training']['num_epochs']):
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

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Валідація
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model artifact saved with accuracy: {best_val_acc:.4f}")

    logger.info(">>> Training process finished successfully.")
    return save_path, train_losses, val_accuracies

def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> None:
    logger.info(">>> Model evaluation on test set started.")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Розрахунок Precision, Recall, F1 через classification_report
    report = classification_report(all_labels, all_preds, target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ])
    
    logger.info(f"Detailed Classification Report:\n{report}")