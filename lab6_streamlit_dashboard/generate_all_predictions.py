from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_loader import load_image
from src.inference import get_transform
from src.mlflow_utils import init_mlflow, load_model_smart
from src.utils import load_config


def generate():
    config = load_config()
    db_uri = config["mlflow"]["tracking_uri"]
    init_mlflow(db_uri)
    client = mlflow.tracking.MlflowClient()

    # 1. Завантажуємо повний тест-реєстр (12 000 картинок)
    # Переконайтеся, що шлях у конфігу веде до повного реєстру, а не до демо
    registry_path = Path("../lab5_wandb_tracking/data/processed/test_registry.csv")
    if not registry_path.exists():
        print(f"❌ Помилка: Реєстр не знайдено за шляхом {registry_path}")
        return

    df_test = pd.read_csv(registry_path)
    print(f"📦 Завантажено тест-реєстр: {len(df_test)} зразків")

    # 2. Отримуємо всі експерименти
    all_exps = mlflow.search_experiments()
    # Вибираємо тільки наш фінальний Research
    research_exp = [e for e in all_exps if e.name == "CIFAR10_Final_Research"]

    if not research_exp:
        print("❌ Експеримент CIFAR10_Final_Research не знайдено в базі!")
        return

    exp_id = research_exp[0].experiment_id
    runs = client.search_runs(experiment_ids=[exp_id])

    device = torch.device("cpu")
    transform = get_transform()

    for run in runs:
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", "Unnamed")

        # Пропускаємо, якщо запуск не завершений
        if run.info.status != "FINISHED":
            continue

        print(f"\n🚀 Оцінюємо модель: {run_name} ({run_id})")

        # Завантажуємо модель за допомогою нашої "розумної" функції
        model = load_model_smart(run_id, exp_id, run_name, db_uri)

        if model is None:
            print(f"⚠️ Не вдалося завантажити ваги для {run_name}. Пропускаємо.")
            continue

        model.to(device)
        model.eval()

        results = []

        # 3. Inference (Прогноз)
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

        # 4. Зберігаємо CSV локально і вантажимо в MLflow як новий артефакт
        preds_df = pd.DataFrame(results)
        temp_csv = f"temp_preds_{run_id[:8]}.csv"
        preds_df.to_csv(temp_csv, index=False)

        # Вантажимо в MLflow до існуючого Run-у
        client.log_artifact(run_id, temp_csv, artifact_path="predictions")

        # Видаляємо тимчасовий файл
        Path(temp_csv).unlink()
        print(f"✅ Файл full_predictions.csv успішно додано до MLflow для {run_name}")


if __name__ == "__main__":
    generate()
