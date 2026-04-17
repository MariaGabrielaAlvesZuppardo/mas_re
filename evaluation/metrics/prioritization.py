from collections import Counter

from scipy.stats import kendalltau, spearmanr

def compute_prioritization_metrics(
    y_true_scores: list[float],
    y_pred_scores: list[float],
) -> dict:
    """
    Calcula métricas de qualidade para priorização via MoSCoW.

    Args:
        y_true_scores: priority_score de referência (ground truth)
        y_pred_scores: priority_score predito pelo agente

    Returns:
        dict com métricas de correlação e ordenação
    """
    kendall_tau, kendall_p = kendalltau(y_true_scores, y_pred_scores)
    spearman_r,  spearman_p = spearmanr(y_true_scores, y_pred_scores)

    mae = sum(
        abs(t-p) for t,p in zip (y_true_scores, y_pred_scores)
    )/len(y_true_scores)
    exact_matches = sum(t == p for t, p in zip(y_true_scores, y_pred_scores))

    return {
        "kendall_tau":   kendall_tau,
        "kendall_p":     kendall_p,
        "spearman_r":    spearman_r,
        "spearman_p":    spearman_p,
        "mae":           mae,
        "exact_matches": exact_matches,
        "total":         len(y_true_scores),
    }

def compute_moscow_distribution(scores:list[float])-> dict:
    """
    Calcula a distribuição dos requisitos pelas categorias MoSCoW.

    Args:
        scores: lista de priority_score dos requisitos priorizados

    Returns:
        dict com contagem e percentual por categoria MoSCoW
    """

    moscow_map = {
        1.00: "Must Have",
        0.75: "Should Have",
        0.50: "Could Have",
        0.25: "Won't Have",
    }

    buckets = [moscow_map.get(round(s,2),"Unknown") for s in scores]
    counts = Counter(buckets)
    total = len(scores)

    return {
        label: {
            "count":   counts.get(label, 0),
            "percent": counts.get(label, 0) / total if total else 0.0,
        }
        for label in ["Must Have", "Should Have", "Could Have", "Won't Have"]
    }