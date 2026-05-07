from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_loader import load_image
from src.inference import get_transform
from src.mlflow_utils import init_mlflow, load_model_smart
from src.utils import get_absolute_path, load_config


def generate():
    config = load_config()
    db_uri = config["mlflow"]["tracking_uri"]
    init_mlflow(db_uri)

    # Визначаємо шлях до mlruns (фізичний)
    base_research_dir = Path(get_absolute_path("../lab4.1_colab_research"))
    mlruns_path = base_research_dir / "mlruns"

    # 1. Завантажуємо тест-реєстр
    registry_path = Path("data_cache/test_registry.csv")
    # registry_path = Path(get_absolute_path(config["data"]["registry_path"]))
    df_test = pd.read_csv(registry_path)
    print(f"📦 Тест-реєстр: {len(df_test)} зразків")

    # 2. Отримуємо запуски
    all_exps = mlflow.search_experiments()
    research_exp = [e for e in all_exps if e.name == "CIFAR10_Final_Research"][0]
    runs = mlflow.search_runs(experiment_ids=[research_exp.experiment_id])

    device = torch.device("cpu")
    transform = get_transform()

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", "Unnamed")
        exp_id = research_exp.experiment_id

        if run["status"] != "FINISHED":
            continue

        print(f"\n🚀 Розрахунок для: {run_name}")

        model = load_model_smart(run_id, exp_id, run_name, db_uri)
        if model is None:
            continue

        model.to(device)
        results = []

        with torch.no_grad():
            for _, row in tqdm(
                df_test.iterrows(), total=len(df_test), desc="Predicting"
            ):
                img = load_image(row["image_path"])
                if img is None:
                    continue
                img_t = transform(img).unsqueeze(0).to(device)
                output = model(img_t)
                probs = F.softmax(output, dim=1)[0]
                conf, pred = torch.max(probs, 0)

                results.append(
                    {
                        "image_path": row["image_path"],
                        "true_label": int(row["label"]),
                        "pred_label": int(pred.item()),
                        "confidence": float(conf.item()),
                    }
                )

        # --- МАГІЯ: ФІЗИЧНИЙ ЗАПИС У ПАПКУ ---
        # Створюємо шлях: lab4.1/mlruns/exp_id/run_id/artifacts/predictions/
        target_dir = (
            mlruns_path / str(exp_id) / str(run_id) / "artifacts" / "predictions"
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        preds_df = pd.DataFrame(results)
        final_csv_path = target_dir / f"temp_preds_{run_id[:8]}.csv"
        preds_df.to_csv(final_csv_path, index=False)

        print(f"✅ ФАЙЛ ПРЯМО ЗАПИСАНО: {final_csv_path}")


if __name__ == "__main__":
    generate()
