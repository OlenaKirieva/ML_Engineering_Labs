import mlflow
import torch

from src.data import create_data_loader, prepare_data_pipelines
from src.engine import evaluate_and_log_artifacts, train_model

# Імпортуємо обидві моделі
from src.model import SimpleCNN
from src.model_new import CIFAR10ProCNN
from src.utils import load_config, setup_logging


def main() -> None:
    setup_logging()
    config = load_config()

    # Автоматичне визначення пристрою (GPU чи CPU)
    # Якщо в конфігу стоїть "cuda", але вона недоступна, вибереться "cpu"
    if config["training"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 1. Налаштування експерименту MLflow
    mlflow.set_experiment(config["experiment_name"])

    # Створюємо ієрархічну назву запуску (Part 4)
    aug_tag = "Aug" if config["training"]["use_augmentation"] else "NoAug"
    lr = config["training"]["lr"]
    epochs = config["training"]["num_epochs"]
    arch_name = config["model"]["architecture"]

    run_name = f"{arch_name}_{aug_tag}_LR{lr}_E{epochs}"

    with mlflow.start_run(run_name=run_name):
        # 2. Логування параметрів
        mlflow.log_params(config["training"])
        mlflow.log_param("architecture", arch_name)

        # 3. Підготовка даних
        train_df, val_df, test_df = prepare_data_pipelines(config)
        train_loader = create_data_loader(
            train_df, config, is_train=config["training"]["use_augmentation"]
        )
        val_loader = create_data_loader(val_df, config, is_train=False)
        test_loader = create_data_loader(test_df, config, is_train=False)

        # 4. Вибір архітектури моделі
        if arch_name == "ProCNN_8Layers":
            print("Initializing ProCNN (8 Layers)...")
            model = CIFAR10ProCNN(n_classes=config["model"]["n_classes"]).to(device)
        else:
            print("Initializing SimpleCNN...")
            model = SimpleCNN(n_classes=config["model"]["n_classes"]).to(device)

        # 5. Навчання (engine.py тепер містить Scheduler та MLflow logging)
        best_model_path = train_model(model, train_loader, val_loader, config, device)

        # 6. Фінальна оцінка та артефакти (Confusion Matrix, Модель, Текстовий звіт)
        print("Evaluation on test set...")
        model.load_state_dict(torch.load(best_model_path))
        evaluate_and_log_artifacts(model, test_loader, device)


if __name__ == "__main__":
    main()
