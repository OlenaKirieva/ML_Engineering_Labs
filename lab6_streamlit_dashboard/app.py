import os
import sys

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix

# Шлях для модулів
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data_loader import load_dataset_registry, load_image
from src.inference import predict_image, run_gradcam
from src.mlflow_utils import get_all_experiments, init_mlflow, load_model_smart
from src.utils import get_absolute_path, load_config, setup_logging

# Ініціалізація
setup_logging()
st.set_page_config(page_title="CIFAR-10 Analyzer Pro", layout="wide")


def main():

    config = load_config()

    # --- 1. ПІДКЛЮЧЕННЯ ДО MLFLOW (Sidebar) ---
    st.sidebar.header("📦 Model Selection")

    raw_db_path = get_absolute_path(config["mlflow"]["db_path"])
    mlflow_uri = f"sqlite:///{raw_db_path.replace('\\', '/')}"
    init_mlflow(mlflow_uri)

    try:
        all_exps = get_all_experiments()
        # Фільтруємо експерименти: лишаємо тільки фінальне дослідження (Part 5 [Task] Filtering)
        final_exps = [e for e in all_exps if e.name == "CIFAR10_Final_Research"]

        if not final_exps:
            st.sidebar.warning("Final Research experiment not found. Showing all.")
            final_exps = all_exps

        exp_names = [e.name for e in final_exps]
        selected_exp_name = st.sidebar.selectbox("Experiment", exp_names)
        selected_exp = next(e for e in final_exps if e.name == selected_exp_name)

        runs_df = mlflow.search_runs(experiment_ids=[selected_exp.experiment_id])

        if not runs_df.empty:
            val_acc_col = (
                "metrics.val_acc" if "metrics.val_acc" in runs_df.columns else "val_acc"
            )

            # Фільтруємо лише успішні та завершені
            runs_df = runs_df[runs_df["status"] == "FINISHED"].copy()

            # --- РОЗУМНА ФІЛЬТРАЦІЯ МОДЕЛЕЙ ---
            # Розділяємо на Pro та Simple
            is_simple = runs_df["tags.mlflow.runName"].str.contains("Simple", na=False)

            # Лишаємо лише один найкращий Simple (Baseline)
            best_simple = (
                runs_df[is_simple].sort_values(by=val_acc_col, ascending=False).head(1)
            )
            # Лишаємо всі ProCNN для порівняння
            all_pro = runs_df[~is_simple].sort_values(by=val_acc_col, ascending=False)

            final_runs = pd.concat([all_pro, best_simple])

            run_names = final_runs["tags.mlflow.runName"].fillna("Unnamed").tolist()
            # ТІЛЬКИ ОДИН SELECTBOX ДЛЯ ВИБОРУ МОДЕЛІ
            selected_run_name = st.sidebar.selectbox("Select Model Version", run_names)

            run_info = final_runs[
                final_runs["tags.mlflow.runName"] == selected_run_name
            ].iloc[0]
            run_id = run_info["run_id"]

            # Завантаження моделі
            model = load_model_smart(
                run_id, selected_exp.experiment_id, selected_run_name, mlflow_uri
            )
            if model:
                st.sidebar.success("✅ Model Loaded!")
            else:
                st.sidebar.error("❌ Weights not found.")
        else:
            st.sidebar.error("No runs in this experiment.")
            st.stop()

    except Exception as e:
        st.sidebar.error(f"MLflow Error: {e}")
        st.stop()

    # --- 2. Вкладки (Tabs) ---
    t1, t2, t3 = st.tabs(
        ["📊 Dataset Exploration", "🔍 Error Analysis", "🧠 Prediction & Grad-CAM"]
    )

    # --- TAB 1: DATASET EXPLORATION ---
    with t1:
        st.header("Dataset Overview")
        registry_path = get_absolute_path(config["data"]["registry_path"])
        df = load_dataset_registry(registry_path)

        if df is not None:
            # ВИПРАВЛЕННЯ ПОМИЛКИ AttributeError (доступ через індекси [0], [1], [2])
            col_sizes = st.columns(3)
            col_sizes[0].metric("Training Set", "38,400", "80%")
            col_sizes[1].metric("Validation Set", "9,600", "20%")
            col_sizes[2].metric("Test Set (Static)", "12,000", "Static")

            c1, c2 = st.columns([1, 2])

            with c1:
                st.subheader("Statistics")
                st.write(f"**Total Samples:** {len(df)}")
                class_counts = df["label"].value_counts().reset_index()
                class_counts.columns = ["class_id", "count"]
                class_counts["class_name"] = class_counts["class_id"].apply(
                    lambda x: config["classes"][x]
                )
                fig = px.bar(
                    class_counts, x="class_name", y="count", title="Class Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Sample Inspection")
                selected_class = st.selectbox(
                    "Filter by class", ["All"] + config["classes"]
                )
                filtered_df = (
                    df
                    if selected_class == "All"
                    else df[df["label"] == config["classes"].index(selected_class)]
                )

                idx = st.number_input("Sample index", 0, len(filtered_df) - 1, 0)
                sample = filtered_df.iloc[idx]
                img = load_image(sample["image_path"])
                if img:
                    st.image(
                        img,
                        caption=f"Class: {config['classes'][sample['label']]}",
                        width=250,
                    )
                    st.caption(f"Path: {sample['image_path']}")

    # --- TAB 2: ERROR ANALYSIS ---
    with t2:
        st.header("🔍 Model Error Analysis")

        # Скидаємо результати, якщо змінився Run ID у Sidebar
        if (
            "last_run_id" not in st.session_state
            or st.session_state.last_run_id != run_id
        ):
            st.session_state.last_run_id = run_id
            if "analysis_data" in st.session_state:
                del st.session_state["analysis_data"]

        # Кнопка запуску
        if st.button("🚀 Run Analysis on 200 Samples"):
            with st.spinner("Analyzing..."):
                subset = df.sample(n=min(200, len(df)), random_state=42)
                y_true = subset["label"].values

                preds, confs = [], []
                for p in subset["image_path"]:
                    p_idx, p_conf, _, _ = predict_image(model, load_image(p))
                    preds.append(p_idx)
                    confs.append(p_conf)

                y_pred = np.array(preds)
                errors_idx = np.where(y_pred != y_true)[0]

                # ЗБЕРІГАЄМО В ПАМ'ЯТЬ СЕСІЇ (Part 5 [Task] Error Handling/Stability)
                st.session_state["analysis_data"] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "confs": confs,
                    "errors_idx": errors_idx,
                    "subset": subset,
                }

        # Відображаємо результати, якщо вони вже є в пам'яті
        if "analysis_data" in st.session_state:
            res = st.session_state["analysis_data"]

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(res["y_true"], res["y_pred"], labels=list(range(10)))
            fig_cm = ff.create_annotated_heatmap(
                z=cm, x=config["classes"], y=config["classes"], colorscale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.subheader(f"Misclassified Examples ({len(res['errors_idx'])})")
            if len(res["errors_idx"]) > 0:
                # Тепер слайдер не буде скидати додаток!
                err_choice = st.slider(
                    "Browse errors", 0, len(res["errors_idx"]) - 1, 0
                )
                real_idx = res["errors_idx"][err_choice]
                err_sample = res["subset"].iloc[real_idx]

                col_img, col_txt = st.columns(2)
                with col_img:
                    st.image(load_image(err_sample["image_path"]), width=200)
                with col_txt:
                    st.error(
                        f"Predicted: **{config['classes'][res['y_pred'][real_idx]]}**"
                    )
                    st.info(
                        f"True Label: **{config['classes'][res['y_true'][real_idx]]}**"
                    )
                    st.write(f"Confidence: {res['confs'][real_idx]:.2%}")
            else:
                st.success("No errors found in this subset!")

    # --- TAB 3: PREDICTION & EXPLAINABILITY ---
    with t3:
        st.header("🧠 Prediction & Explainability")
        uploaded_file = st.file_uploader(
            "Upload an image...", type=["jpg", "png", "jpeg"]
        )

        if uploaded_file and model:
            img = Image.open(uploaded_file).convert("RGB")

            # Отримуємо результати (тепер тут 4 значення)
            pred_idx, conf, tensor, probs_vector = predict_image(model, img)

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.subheader("Model Decision")
                st.image(img, caption="Uploaded Image", use_container_width=True)
                st.success(
                    f"**Prediction:** {config['classes'][pred_idx]} ({conf:.2%})"
                )

                # Графік розподілу ймовірностей (вимога Part 4)
                prob_df = pd.DataFrame(
                    {
                        "Class": config["classes"],
                        "Probability": probs_vector.cpu().numpy(),
                    }
                ).sort_values(by="Probability", ascending=True)

                fig_probs = px.bar(
                    prob_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    title="Confidence Score per Class",
                    color="Probability",
                    color_continuous_scale="Blues",
                )
                fig_probs.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_probs, use_container_width=True)

            with col_res2:
                st.subheader("Visual Explanation (Grad-CAM)")
                with st.spinner("Generating heatmap..."):
                    heatmap = run_gradcam(model, tensor)
                    if heatmap is not None:
                        st.image(
                            heatmap,
                            caption="Grad-CAM Heatmap",
                            use_container_width=True,
                        )
                        st.info(
                            "💡 Червоні зони показують області, на основі яких модель прийняла рішення."
                        )
                    else:
                        st.error("Grad-CAM generation failed.")


if __name__ == "__main__":
    main()
