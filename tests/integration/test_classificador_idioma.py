import pytest
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from agents.classificador import ClassificationAgent
from evaluation.metrics.classification import compute_classification_metrics
from pipeline.state import Requirement

load_dotenv(Path(__file__).parent.parent.parent/".env")
CSV_PT = Path("datasets/promise_nfr/Promise+_pt.csv")
SAMPLE_N = 15
MODEL = "llama-3.3-70b-versatile"

@pytest.fixture(scope="module")
def dataset_biligue():
    if not CSV_PT.exists():
        pytest.skip("Promise+_pt.csv não encontrado — rode translate_promise.py primeiro")
    df = pd.read_csv(CSV_PT)
    return df.head(SAMPLE_N)

@pytest.fixture(scope="module")
def agent():
    return ClassificationAgent(model=MODEL)

def _classificar(agent, textos: list[str], ids: list[str]) -> list[str]:
    """Classifica uma lista de textos e retorna os labels preditos."""
    preds = []
    for req_id, texto in zip(ids, textos):
        req = Requirement(id=req_id, text=texto, source="promise_nfr")
        resultado = agent.classify(req)
        preds.append(resultado.category.value)
    return preds

def test_f1_en_vs_pt(dataset_biligue, agent):
    """
    Compara F1 macro entre classificação em EN e PT.
    Documenta o impacto da língua na qualidade.
    """
    y_true = dataset_biligue["label"].tolist()
    ids    = [f"PROMISE-{row['project_id']}-{i:03d}"
              for i, row in dataset_biligue.iterrows()]

    # Classifica em EN
    preds_en = _classificar(agent, dataset_biligue["requirement_en"].tolist(), ids)
    metricas_en = compute_classification_metrics(y_true, preds_en)

    # Classifica em PT
    preds_pt = _classificar(agent, dataset_biligue["requirement_pt"].tolist(), ids)
    metricas_pt = compute_classification_metrics(y_true, preds_pt)

    print(f"\n{'Métrica':<20} {'EN':>8} {'PT':>8} {'Δ':>8}")
    print("-" * 46)
    for metrica in ["f1_macro", "f1_fr", "f1_nfr", "precision_macro", "recall_macro"]:
        delta = metricas_pt[metrica] - metricas_en[metrica]
        sinal = "+" if delta >= 0 else ""
        print(f"{metrica:<20} {metricas_en[metrica]:>8.3f} {metricas_pt[metrica]:>8.3f} {sinal}{delta:>7.3f}")

    # Documenta a diferença — não falha, pois é um achado de pesquisa
    assert isinstance(metricas_en["f1_macro"], float)
    assert isinstance(metricas_pt["f1_macro"], float)


def test_categorias_mais_afetadas_por_idioma(dataset_biligue, agent):
    """
    Identifica quais subcategorias NFR são mais afetadas pela tradução PT.
    """
    nfr_df = dataset_biligue[dataset_biligue["label"] == "NFR"]

    if nfr_df.empty:
        pytest.skip("Nenhum NFR na amostra")

    ids = [f"PROMISE-{row['project_id']}-{i:03d}"
           for i, row in nfr_df.iterrows()]

    preds_en = _classificar(agent, nfr_df["requirement_en"].tolist(), ids)  # agents → agent
    preds_pt = _classificar(agent, nfr_df["requirement_pt"].tolist(), ids)  # agents → agent
    y_true   = nfr_df["label"].tolist()

    acertos_en = sum(t == p for t, p in zip(y_true, preds_en))
    acertos_pt = sum(t == p for t, p in zip(y_true, preds_pt))

    print(f"\nNFR — acertos EN: {acertos_en}/{len(y_true)}")
    print(f"NFR — acertos PT: {acertos_pt}/{len(y_true)}")

    assert isinstance(acertos_en, int)


def test_requisitos_com_classificacao_divergente(dataset_biligue, agent):  # agentes → agent
    """
    Lista requisitos onde EN e PT produziram classificações diferentes.
    Útil para análise qualitativa na apresentação.
    """
    ids = [f"PROMISE-{row['project_id']}-{i:03d}"
           for i, row in dataset_biligue.iterrows()]

    preds_en = _classificar(agent, dataset_biligue["requirement_en"].tolist(), ids)  # agentes → agent
    preds_pt = _classificar(agent, dataset_biligue["requirement_pt"].tolist(), ids)  # agentes → agent

    divergentes = [
        {
            "id":      req_id,
            "en":      en,
            "pt":      pt,
            "pred_en": pred_en,
            "pred_pt": pred_pt,
            "truth":   truth,
        }
        for req_id, en, pt, pred_en, pred_pt, truth in zip(
            ids,
            dataset_biligue["requirement_en"],
            dataset_biligue["requirement_pt"],
            preds_en,
            preds_pt,
            dataset_biligue["label"],
        )
        if pred_en != pred_pt
    ]

    print(f"\nRequisitos com divergência EN/PT: {len(divergentes)}/{len(ids)}")
    for d in divergentes:
        print(f"\n  {d['id']} | truth: {d['truth']} | EN: {d['pred_en']} | PT: {d['pred_pt']}")
        print(f"  EN: {d['en'][:70]}")
        print(f"  PT: {d['pt'][:70]}")

    # Documenta — não falha
    assert isinstance(divergentes, list)