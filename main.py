import torch
from src.utils import setup_logging, load_config
from src.data import process_data, create_data_loader
from src.model import SimpleCNN
from src.engine import train_model, test_model

def main() -> None:
    setup_logging()
    config = load_config()
    device = torch.device("cpu") # Baseline на CPU

    # 1. Data
    train_df, val_df, test_df = process_data(config)
    train_loader = create_data_loader(train_df, config, is_train=True)
    val_loader = create_data_loader(val_df, config, is_train=False)
    test_loader = create_data_loader(test_df, config, is_train=False)

    # 2. Model
    model = SimpleCNN(n_classes=config['model']['n_classes']).to(device)

    # 3. Train
    best_model_path = train_model(model, train_loader, val_loader, config, device)

    # 4. Test
    model.load_state_dict(torch.load(best_model_path))
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()