import mlflow
import torch
import torch.nn as nn
from src.utils import setup_logging, load_config
from src.data import create_data_loader, prepare_data_pipelines
from src.engine import evaluate_and_log_artifacts, train_model

# Імпортуємо обидві архітектури
from src.model import SimpleCNN
from src.model_new import CIFAR10ProCNN

def main() -> None:
    setup_logging()
    config = load_config()

    # Визначення пристрою
    if config["training"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    # 1. Налаштування експерименту MLflow
    mlflow.set_experiment(config["experiment_name"])

    # Отримуємо дані з конфігу для назви (ВИПРАВЛЕНО: додаємо визначення змінних)
    arch_name = config["model"]["architecture"]
    aug_tag = "Aug_Yes" if config["training"]["use_augmentation"] else "Aug_No"
    lr = config["training"]["lr"]
    epochs = config["training"]["num_epochs"]
    bs = config["training"]["batch_size"]
    
    # Структура: модель – архітектура – аугментація – learning rate – кількість епох
    run_name = f"{arch_name} | {aug_tag} | LR:{lr} | E:{epochs} | BS:{bs}"

    with mlflow.start_run(run_name=run_name):
        # 2. Логування параметрів
        mlflow.log_params(config["training"])
        mlflow.log_params(config["data"])
        mlflow.log_param("architecture", arch_name)

        # 3. Підготовка даних
        train_df, val_df, test_df = prepare_data_pipelines(config)
        train_loader = create_data_loader(
            train_df, config, is_train=config["training"]["use_augmentation"]
        )
        val_loader = create_data_loader(val_df, config, is_train=False)
        test_loader = create_data_loader(test_df, config, is_train=False)

        # 4. Вибір моделі
        if arch_name == "ProCNN_8Layers":
            model = CIFAR10ProCNN(n_classes=config['model']['n_classes']).to(device)
        else:
            model = SimpleCNN(n_classes=config['model']['n_classes']).to(device)

        # 5. Навчання
        best_model_path = train_model(model, train_loader, val_loader, config, device)

        # 6. Оцінка та артефакти
        model.load_state_dict(torch.load(best_model_path))
        evaluate_and_log_artifacts(model, test_loader, device)

if __name__ == "__main__":
    main()
