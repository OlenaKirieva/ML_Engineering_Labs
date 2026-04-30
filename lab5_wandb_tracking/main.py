import torch

import wandb
from src.data import create_data_loader, prepare_data_pipelines
from src.engine import evaluate_and_log_to_wandb, train_model
from src.model_new import CIFAR10ProCNN
from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config()

    # СУВОРЕ ВИЗНАЧЕННЯ ПРИСТРОЮ З КОНФІГУ
    device_name = config["training"]["device"]
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # СКЛАДНА ІЄРАРХІЧНА НАЗВА
    arch = config["model"]["architecture"]
    aug = "WithAug" if config["training"]["use_augmentation"] else "NoAug"
    lr = config["training"]["lr"]
    epochs = config["training"]["num_epochs"]

    run_name = f"{arch}_{aug}_LR{lr}_E{epochs}"

    # Ініціалізація W&B
    wandb.init(project=config["project_name"], name=run_name, config=config)

    # Дані
    train_df, val_df, test_df = prepare_data_pipelines(config)
    train_loader = create_data_loader(
        train_df, config, is_train=config["training"]["use_augmentation"]
    )
    val_loader = create_data_loader(val_df, config, is_train=False)
    test_loader = create_data_loader(test_df, config, is_train=False)

    # Модель
    model = CIFAR10ProCNN(config["model"]["n_classes"]).to(
        device
    )  # Переконайтеся, що назва класу моделі вірна

    # Навчання
    best_model_path = train_model(model, train_loader, val_loader, config, device)

    # Оцінка та артефакти
    model.load_state_dict(torch.load(best_model_path))
    evaluate_and_log_to_wandb(model, test_loader, device, best_model_path)

    wandb.finish()


if __name__ == "__main__":
    main()
