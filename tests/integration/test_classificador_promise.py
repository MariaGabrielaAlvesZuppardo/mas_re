import pytest
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv

from agents.classificador import ClassificationAgent
from datasets.promise import PromiseAdapter
from evaluation.metrics.classification import (
compute_classification_metrics,
compute_subcategory_metrics
)
from pipeline.state import Requirement

load_dotenv(Path(__file__).parent.parent.parent / ".env")

SAMPLE_N = 15
MODEL = "llama-3.3-70b-versatile"
RANDOM_SEED = 42

@pytest.fixture(scope="module")
def sample_data():
    adapter = PromiseAdapter()
    return adapter.load(sample_n=SAMPLE_N, random_state=RANDOM_SEED)


@pytest.fixture(scope="module")
def agente():
    return ClassificationAgent(model=MODEL)


@pytest.fixture(scope="module")
def resultados(sample_data, agente):
    """Roda o agente em toda a amostra e coleta resultados."""
    saidas = []
    for i, item in enumerate(sample_data.items, 1):
        req = Requirement(
            id=f"PROMISE-{item.project_id}-{i:03d}",
            text=item.requirement,
            source=item.source,
        )
        resultado = agente.classify(req)
        saidas.append({
            "id":             req.id,
            "text":           item.requirement,
            "y_true":         item.label,
            "y_pred":         resultado.category.value,
            "subcat_true":    item.nfr_category or "",
            "subcat_pred":    resultado.subcategory,
            "confidence":     resultado.confidence,
        })
    return saidas

def test_f1_macro_minimo(resultados):
    """F1 macro deve ser >= 0.60 para ser publicável."""
    y_true = [r["y_true"] for r in resultados]
    y_pred = [r["y_pred"] for r in resultados]
    metricas = compute_classification_metrics(y_true, y_pred)

    print(f"\n{metricas['report']}")
    assert metricas["f1_macro"] >= 0.60, (
        f"F1 macro abaixo do mínimo: {metricas['f1_macro']:.3f}"
    )


def test_f1_nfr(resultados):
    """NFR tende a ser mais difícil — F1 deve ser >= 0.55."""
    y_true = [r["y_true"] for r in resultados]
    y_pred = [r["y_pred"] for r in resultados]
    metricas = compute_classification_metrics(y_true, y_pred)

    print(f"\nF1 NFR: {metricas['f1_nfr']:.3f}")
    assert metricas["f1_nfr"] >= 0.55


def test_acuracia_geral(resultados):
    """Pelo menos 60% dos requisitos classificados corretamente."""
    corretos = sum(1 for r in resultados if r["y_true"] == r["y_pred"])
    acuracia = corretos / len(resultados)

    print(f"\nAcurácia: {acuracia:.2%} ({corretos}/{len(resultados)})")
    assert acuracia >= 0.60


def test_metricas_por_subcategoria(resultados):
    """Calcula F1 por subcategoria NFR e imprime para análise."""
    nfr = [r for r in resultados if r["y_true"] == "NFR"]

    if not nfr:
        pytest.skip("Nenhum NFR na amostra")

    subcat_true = [r["subcat_true"] for r in nfr]
    subcat_pred = [r["subcat_pred"] for r in nfr]
    metricas    = compute_subcategory_metrics(subcat_true, subcat_pred)

    print("\nF1 por subcategoria NFR:")
    for cat, f1 in sorted(metricas.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {f1:.3f}")

    assert isinstance(metricas, dict)


def test_confidence_correlaciona_com_acerto(resultados):
    """
    Documenta a correlação entre confidence e acerto.
    Não falha — é um achado de pesquisa sujeito a variação com amostras pequenas.
    """
    alta_conf  = [r for r in resultados if r["confidence"] >= 0.90]
    baixa_conf = [r for r in resultados if r["confidence"] <  0.90]

    if not alta_conf or not baixa_conf:
        pytest.skip("Amostra insuficiente para comparar confidence")

    acc_alta  = sum(1 for r in alta_conf  if r["y_true"] == r["y_pred"]) / len(alta_conf)
    acc_baixa = sum(1 for r in baixa_conf if r["y_true"] == r["y_pred"]) / len(baixa_conf)

    print(f"\nAcurácia alta confiança  (>= 0.90): {acc_alta:.2%} ({len(alta_conf)} requisitos)")
    print(f"Acurácia baixa confiança (< 0.90):  {acc_baixa:.2%} ({len(baixa_conf)} requisitos)")
    print(f"Correlação positiva: {acc_alta >= acc_baixa}")

    # Documenta — não falha, pois com SAMPLE_N pequeno a correlação pode inverter por acaso
    assert isinstance(acc_alta,  float)
    assert isinstance(acc_baixa, float)

