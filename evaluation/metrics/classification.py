from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = ["FR", "NFR"],
) -> dict:
    """
    Calcula F1, Precision e Recall para classificação FR/NFR.

    Args:
        y_true: labels reais do PROMISE+
        y_pred: labels preditas pelo agente
        labels: classes possíveis

    Returns:
        dict com métricas agregadas e por classe
    """
    return {
        "f1_macro":         f1_score(y_true, y_pred, average="macro",    labels=labels),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted", labels=labels),
        "precision_macro":  precision_score(y_true, y_pred, average="macro",  labels=labels),
        "recall_macro":     recall_score(y_true, y_pred, average="macro",     labels=labels),
        "f1_fr":            f1_score(y_true, y_pred, pos_label="FR",  average="binary"),
        "f1_nfr":           f1_score(y_true, y_pred, pos_label="NFR", average="binary"),
        "report":           classification_report(y_true, y_pred, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "total":            len(y_true),
        "correct":          sum(t == p for t, p in zip(y_true, y_pred)),
    }


def compute_subcategory_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict:
    """
    Calcula F1 por subcategoria NFR.

    Args:
        y_true: subcategorias reais
        y_pred: subcategorias preditas

    Returns:
        dict com F1 por subcategoria
    """
    categories = sorted(set(y_true) - {""})          # ← só categorias reais, ignora ""
    return {
        cat: f1_score(y_true, y_pred, labels=[cat], average="macro", zero_division=0)
        for cat in categories                          # ← one-vs-rest por categoria
    }