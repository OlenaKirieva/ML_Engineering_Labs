import shutil
from pathlib import Path

import pandas as pd


def create_demo():
    # Шлях до оригінального реєстру з Лаби 5
    src_registry = "../lab5_wandb_tracking/data/processed/test_registry.csv"
    # Папка для демо-картинок всередині Лаби 6
    demo_img_dir = Path("static/demo_images")
    demo_img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_registry)

    # Беремо по 50 випадкових картинок для кожного з 10 класів (разом 500)
    demo_df = df.groupby("label").sample(n=50, random_state=42).copy()

    # Копіюємо файли та оновлюємо шляхи на ВІДНОСНІ
    new_paths = []
    for _, row in demo_df.iterrows():
        old_path = Path(row["image_path"])
        new_filename = old_path.name
        new_path = demo_img_dir / new_filename

        if old_path.exists():
            shutil.copy(old_path, new_path)
            # ВАЖЛИВО: записуємо шлях відносно папки lab6
            new_paths.append(f"static/demo_images/{new_filename}")
        else:
            new_paths.append("file_not_found")

    demo_df["image_path"] = new_paths
    demo_df.to_csv("config/demo_registry.csv", index=False)
    print(f"✅ Demo registry created with {len(demo_df)} images.")


if __name__ == "__main__":
    create_demo()
