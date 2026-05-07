import os
import sys

import mlflow
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from src.data_loader import load_image, prepare_test_data_locally
from src.explainability import run_lime
from src.inference import predict_image, run_gradcam
from src.mlflow_utils import (
    get_all_experiments,
    init_mlflow,
    load_artifact_text,
    load_model_smart,
    load_predictions_from_mlflow,
)
from src.ui_components import (
    create_report_image,
    render_classification_report_mini,
    render_error_matrix,
    render_global_stats,
    render_prediction_box,
    render_probability_chart,
)
from src.utils import MODEL_DESCRIPTIONS, get_absolute_path, load_config, setup_logging

setup_logging()
st.set_page_config(page_title="CIFAR-10 Pro Analyzer", layout="wide")


def main():
    config = load_config()

    # --- 1. ПІДГОТОВКА ДАНИХ (10,000 КАРТИНОК) ---
    df_full = prepare_test_data_locally()

    # --- 2. SIDEBAR: MODEL SELECTION ---
    st.sidebar.header("📦 Model Selection")
    db_path = get_absolute_path(config["mlflow"]["db_path"])
    mlflow_uri = f"sqlite:///{db_path.replace('\\', '/')}"
    init_mlflow(mlflow_uri)

    try:
        all_exps = get_all_experiments()
        selected_exp = next(e for e in all_exps if e.name == "CIFAR10_Final_Research")
        runs_df = mlflow.search_runs(experiment_ids=[selected_exp.experiment_id])
        finished_runs = runs_df[runs_df["status"] == "FINISHED"].copy()

        # Фільтрація моделей
        is_simple = finished_runs["tags.mlflow.runName"].str.contains(
            "Simple", na=False
        )
        best_simple = (
            finished_runs[is_simple]
            .sort_values(by="metrics.val_acc", ascending=False)
            .head(1)
        )
        all_pro = finished_runs[~is_simple].sort_values(
            by="metrics.val_acc", ascending=False
        )
        final_runs = pd.concat([all_pro, best_simple])

        selected_run_name = st.sidebar.selectbox(
            "Select Model Version", final_runs["tags.mlflow.runName"].tolist()
        )
        run_info = final_runs[
            final_runs["tags.mlflow.runName"] == selected_run_name
        ].iloc[0]
        run_id = run_info["run_id"]

        model = load_model_smart(
            run_id, selected_exp.experiment_id, selected_run_name, mlflow_uri
        )

        if model:
            st.sidebar.success("✅ Model Loaded!")

        # ЗАВАНТАЖЕННЯ ТА СИНХРОНІЗАЦІЯ ПРОГНОЗІВ
        df_preds_raw = load_predictions_from_mlflow(
            run_id, mlflow_uri, selected_exp.experiment_id
        )

        if df_preds_raw is not None:
            # Якщо довжини різні (наприклад 500 в MLflow і 10000 локально), робимо зріз
            min_len = min(len(df_preds_raw), len(df_full))
            df_preds = df_preds_raw.iloc[:min_len].copy()
            df_preds["image_path"] = df_full["image_path"].iloc[:min_len].values
            if len(df_preds_raw) < 10000:
                st.sidebar.warning(
                    f"⚠️ Only {len(df_preds_raw)} predictions found for this model."
                )
        else:
            df_preds = None

        # Паспорт моделі
        st.sidebar.divider()
        st.sidebar.subheader("📄 Model Passport")
        m_key = (
            "ProCNN_8Layers_Optimal"
            if "E:100" in selected_run_name
            else (
                "SimpleCNN" if "Simple" in selected_run_name else "ProCNN_8Layers_Fast"
            )
        )
        info = MODEL_DESCRIPTIONS.get(m_key, {})
        st.sidebar.write(f"**Arch:** {info.get('description', 'N/A')}")
        st.sidebar.write(
            f"**Layers:** {info.get('layers')} | **Params:** {info.get('params')}"
        )

    except Exception as e:
        st.sidebar.error(f"MLflow Error: {e}")
        st.stop()

    # --- 3. Вкладки ---
    t1, t2, t3 = st.tabs(
        ["📊 Dataset Exploration", "🔍 Error Explorer", "🧠 Explainability"]
    )

    # --- TAB 1: DATASET ---
    with t1:
        c1, c2 = st.columns([1.5, 1.2])
        with c1:
            render_global_stats(config["classes"])
        with c2:
            st.subheader("Interactive Inspector")
            if df_full is not None:
                sel_class = st.selectbox(
                    "Category", ["All"] + config["classes"], key="t1_cls"
                )
                filtered = (
                    df_full
                    if sel_class == "All"
                    else df_full[df_full["label"] == config["classes"].index(sel_class)]
                )
                idx = st.number_input(
                    f"Sample Index (0 - {len(filtered)-1})", 0, len(filtered) - 1, 0
                )
                sample = filtered.iloc[idx]
                st.image(load_image(sample["image_path"]), width=200)
                # ВИМОГА: Додати правильну категорію
                st.info(f"**True Category:** {config['classes'][sample['label']]}")

    # --- TAB 2: ERROR EXPLORER (ТРИ СТОВПЧИКИ) ---
    with t2:
        # st.header("🔍 Comprehensive Performance Analysis")
        if df_preds is not None:
            col_matrix, col_report = st.columns([1, 1])

            with col_matrix:
                render_error_matrix(df_preds, config["classes"])

            with col_report:
                rep = load_artifact_text(
                    run_id,
                    selected_exp.experiment_id,
                    mlflow_uri,
                    "classification_report.txt",
                )
                render_classification_report_mini(rep)

        # Створюємо два стовпчики для нижнього ряду
        col_filter, col_preview = st.columns([1, 1])

        with col_filter:
            st.write("##### 🛠️ Filters")
            f_col1, _, f_col2 = st.columns([1, 0.2, 1])

            with f_col1:
                # 1. Вибір реального класу (True Label) з опцією "All"
                t_f_name = st.selectbox(
                    "Select Actual Class:",
                    ["All"] + config["classes"],
                    key="t2_filter_true",
                )

                # 2. Вибір прогнозу (Predicted Label)
                p_options = ["All Errors", "Correct Only"] + config["classes"]
                p_f_name = st.selectbox(
                    "Model Predicted as:", p_options, key="t2_filter_pred"
                )

                # --- СКЛАДНА ЛОГІКА ФІЛЬТРАЦІЇ ---
                temp_df = df_preds.copy()

                # Фільтр по входу (Actual)
                if t_f_name != "All":
                    t_idx = config["classes"].index(t_f_name)
                    temp_df = temp_df[temp_df["true_label"] == t_idx]

                # Фільтр по виходу (Predicted)
                if p_f_name == "All Errors":
                    display_df = temp_df[temp_df["true_label"] != temp_df["pred_label"]]
                elif p_f_name == "Correct Only":
                    display_df = temp_df[temp_df["true_label"] == temp_df["pred_label"]]
                else:
                    p_idx = config["classes"].index(p_f_name)
                    display_df = temp_df[temp_df["pred_label"] == p_idx]
                    # Якщо обрано конкретний клас передбачення, а Actual="All",
                    # то зазвичай цікаво бачити помилки (де модель сплутала щось із цим класом)
                    if t_f_name == "All":
                        display_df = display_df[
                            display_df["true_label"] != display_df["pred_label"]
                        ]

            # --- СОРТУВАННЯ ---
            with f_col2:
                sort_mode = st.radio(
                    "Sort results by:", ["Highest Confidence", "Lowest Confidence"]
                )
                display_df = display_df.sort_values(
                    by="confidence", ascending=(sort_mode == "Lowest Confidence")
                )

                st.write(f"🔍 Found **{len(display_df)}** matches.")

            if not display_df.empty:
                idx_in_list = st.slider("Browse matches:", 0, len(display_df) - 1, 0)
                selected_row = display_df.iloc[idx_in_list]
                global_idx = display_df.index[idx_in_list]
            else:
                st.warning("No samples found for this combination.")

            st.caption("💡 Tip: Each class has 1000 samples.")

        with col_preview:
            st.write("##### 🖼️ Sample Preview")
            if not display_df.empty:
                # Вкладені колонки для компактного відображення
                img_col, info_col = st.columns([1, 1])

                with img_col:
                    # Картинка та індекс зліва
                    img = load_image(selected_row["image_path"])
                    if img:
                        # Трохи зменшимо ширину, щоб ідеально вписалося
                        st.image(img, width=200)
                    st.caption(f"**Global Index:** `{global_idx}`")

                with info_col:
                    # Текстові блоки та прогрес-бар праворуч
                    st.info(
                        f"**True Value:** {config['classes'][selected_row['true_label']]}"
                    )

                    if selected_row["true_label"] == selected_row["pred_label"]:
                        st.success(
                            f"**Prediction:** {config['classes'][selected_row['pred_label']]} (Correct)"
                        )
                    else:
                        st.error(
                            f"**Prediction:** {config['classes'][selected_row['pred_label']]} (Incorrect)"
                        )

                    conf_val = selected_row["confidence"]
                    st.write(f"**Confidence:** {conf_val:.2%}")
                    # st.progress(conf_val)

                # Повідомлення-підказка в самому низу під двома колонками
                st.caption(
                    "💡 Tip: Copy the Global Index to Tab 3 for detailed interpretability analysis."
                )

    # --- TAB 3: EXPLAINABILITY ---
    with t3:
        st.header("🧠 Model Interpretation Studio")

        c_mode, c_method = st.columns(2)

        with c_mode:
            src_mode = st.radio(
                "Data Source",
                ["Browse by Category", "Global Index", "Upload File"],
                horizontal=True,
            )

        with c_method:
            method = st.radio("Explain Method", ["Grad-CAM", "LIME"], horizontal=True)

        img, true_label, current_idx = None, None, "upload"

        # ЛОГІКА ВИБОРУ ЗОБРАЖЕННЯ
        f_col1, _ = st.columns([1, 2])
        with f_col1:
            if src_mode == "Browse by Category":
                sel_cat = st.selectbox(
                    "Select Class to explore", config["classes"], key="t3_cat"
                )
                cat_df = df_full[df_full["label"] == config["classes"].index(sel_cat)]
                sub_idx = st.slider(f"Samples in {sel_cat}", 0, len(cat_df) - 1, 0)
                row = cat_df.iloc[sub_idx]
                img = load_image(row["image_path"])
                true_label = sel_cat
                current_idx = cat_df.index[sub_idx]

            elif src_mode == "Global Index":
                g_idx = st.number_input("Enter Index (0-9999)", 0, 9999, 0)
                row = df_full.iloc[g_idx]
                img = load_image(row["image_path"])
                true_label = config["classes"][row["label"]]
                current_idx = g_idx

            elif src_mode == "Upload File":
                f = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])
                if f:
                    img = Image.open(f).convert("RGB")

        # ОСНОВНИЙ БЛОК АНАЛІЗУ
        if img and model:
            # 1. Prediction
            p_idx, p_conf, tens, probs_vec = predict_image(model, img)

            # Створюємо DF для графіка (потрібен і для UI, і для звіту)
            prob_df = pd.DataFrame(
                {"Class": config["classes"], "Prob": probs_vec.cpu().numpy()}
            ).sort_values(by="Prob")

            # 2. UI Layout
            res_c1, res_c2, res_c3 = st.columns([1, 1.2, 1])

            with res_c1:
                st.subheader("Target")
                # Використовуємо 224x224 для UI, щоб було чітко видно
                st.image(img.resize((224, 224)), width=250)
                render_prediction_box(config, p_idx, p_conf, true_label)

            with res_c2:
                st.subheader("Probabilities")
                render_probability_chart(config["classes"], probs_vec)

            with res_c3:
                st.subheader("Explanation")
                with st.spinner(f"Running {method}..."):
                    if method == "Grad-CAM":
                        explanation_data = run_gradcam(model, tens, (224, 224))
                    else:
                        explanation_data = run_lime(model, img)
                    
                    if explanation_data is not None:
                        # МАГІЯ: Перетворюємо масив у об'єкт PIL Image
                        # Це прибере помилку з ".format == GIF"
                        final_img = Image.fromarray(explanation_data)
                        st.image(final_img, width=250, caption=f"{method} View")
                    else:
                        st.error("Explanation failed.")
                        
            # 3. КНОПКА ЗАВАНТАЖЕННЯ ЗВІТУ
            st.divider()
            report_btn_col, _ = st.columns([1, 3])
            with report_btn_col:
                report_bytes = create_report_image(
                    img.resize((224, 224)),  # Фото у звіті буде чітким
                    explanation_data,
                    config["classes"][p_idx],
                    p_conf,
                    selected_run_name,
                    info,
                    method,
                    prob_df,
                )
                st.download_button(
                    label="📥 Download Analytical Report",
                    data=report_bytes,
                    file_name=f"report_idx_{current_idx}_{method}.png",
                    mime="image/png",
                )


if __name__ == "__main__":
    main()
