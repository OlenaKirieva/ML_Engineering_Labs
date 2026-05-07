import logging

logger = logging.getLogger(__name__)

def get_error_metrics(df_preds):
    total = len(df_preds)
    errors = df_preds[df_preds["true_label"] != df_preds["pred_label"]]
    acc = (total - len(errors)) / total
    return len(errors), acc

def get_filtered_errors(df_preds, true_cls, pred_cls, classes, sort_mode):
    logger.info(f"Filtering errors for True:{true_cls} Pred:{pred_cls}")
    errors = df_preds[df_preds["true_label"] != df_preds["pred_label"]].copy()
    if true_cls != "All":
        errors = errors[errors["true_label"] == classes.index(true_cls)]
    if pred_cls != "All":
        errors = errors[errors["pred_label"] == classes.index(pred_cls)]

    ascending = sort_mode == "Lowest Confidence"
    return errors.sort_values(by="confidence", ascending=ascending)
