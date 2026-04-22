import logging

import torch

from src.data_utils import create_data_loader, prepare_data_pipelines
from src.engine import save_metrics_plot, test_model, train_model
from src.model import SimpleCNN
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Setup
    setup_logging()
    config = load_config()
    device = torch.device("cpu")

    logger.info(">>> Starting Lab 2 Experiment")
    logger.info(
        f"Configuration: Selected Batches = {config['data']['batch_names_select']}"
    )

    # 2. Data Preparation (Dynamic batch selection)
    train_df, val_df, test_df = prepare_data_pipelines(config)

    train_loader = create_data_loader(train_df, config, is_train=True)
    val_loader = create_data_loader(val_df, config, is_train=False)
    test_loader = create_data_loader(test_df, config, is_train=False)

    # 3. Model Initialization
    model = SimpleCNN(n_classes=config["model"]["n_classes"]).to(device)

    # 4. Training (Dynamic sets)
    # Зверніть увагу: ми зберігаємо графік з іменем, що залежить від кількості батчів
    n_sel = len(config["data"]["batch_names_select"])
    plot_name = f"artifacts/metrics_{n_sel}_batches.png"

    best_model_path, train_losses, val_accs = train_model(
        model, train_loader, val_loader, config, device
    )

    save_metrics_plot(train_losses, val_accs, plot_name)

    # 5. Final Evaluation on STATIC Test Set
    logger.info("Loading best model for final evaluation on static test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_model(model, test_loader, device)

    logger.info(">>> Experiment finished successfully.")


if __name__ == "__main__":
    main()
