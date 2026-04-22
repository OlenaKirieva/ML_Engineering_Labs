import torch

from src.data import create_data_loader, process_data
from src.engine import save_metrics_plot, test_model, train_model
from src.model import SimpleCNN
from src.utils import load_config, setup_logging


def main() -> None:
    setup_logging()
    config = load_config()
    device = torch.device("cpu")

    # 1. Data
    train_df, val_df, test_df = process_data(config)
    train_loader = create_data_loader(train_df, config, is_train=True)
    val_loader = create_data_loader(val_df, config, is_train=False)
    test_loader = create_data_loader(test_df, config, is_train=False)

    # 2. Model
    model = SimpleCNN(n_classes=config["model"]["n_classes"]).to(device)

    # 3. Train
    best_model_path, train_losses, val_accs = train_model(
        model, train_loader, val_loader, config, device
    )

    # 4. Візуалізація
    save_metrics_plot(train_losses, val_accs, "artifacts/improved_metrics.png")

    # 5. Test
    model.load_state_dict(torch.load(best_model_path))
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
