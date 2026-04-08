"""
datasets/translate_promise.py
──────────────────────────────
Traduz os requisitos do PROMISE NFR+ de inglês para português.

Entrada:  datasets/promise_nfr/Promise+.arff
Saída:    datasets/promise_nfr/Promise+_pt.csv

Uso:
    python datasets/translate_promise.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

from datasets.promise import PromiseAdapter

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
_OUTPUT    = _ROOT / "datasets" / "promise_nfr" / "Promise+_pt.csv"

# ── Configurações ─────────────────────────────────────────────────────────────
_SLEEP_BETWEEN_REQUESTS = 0.1   # evita rate limit do Google Translate
_SOURCE_LANG            = "en"
_TARGET_LANG            = "pt"


def translate_dataset(
    output_path: Path = _OUTPUT,
    sample_n:    int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Carrega o PROMISE+, traduz os requisitos e salva em CSV.

    Args:
        output_path:  caminho do CSV de saída
        sample_n:     traduz apenas N items (None = todos)
        random_state: semente para amostragem reprodutível

    Returns:
        DataFrame com requisitos traduzidos
    """
    # ── Carrega o dataset ──────────────────────────────────────────────────────
    logger.info("Carregando PROMISE NFR+...")
    adapter = PromiseAdapter()
    data    = adapter.load(sample_n=sample_n, random_state=random_state)
    logger.info("Total de requisitos: %d", len(data.items))

    # ── Traduz ────────────────────────────────────────────────────────────────
    translator = GoogleTranslator(source=_SOURCE_LANG, target=_TARGET_LANG)
    rows: list[dict] = []

    for item in tqdm(data.items, desc="Traduzindo para PT"):
        try:
            requirement_pt = translator.translate(item.requirement)
            time.sleep(_SLEEP_BETWEEN_REQUESTS)
        except Exception as exc:
            logger.warning("Falha ao traduzir '%s': %s", item.requirement[:50], exc)
            requirement_pt = item.requirement  # mantém original se falhar

        rows.append({
            "project_id":     item.project_id,
            "requirement_en": item.requirement,
            "requirement_pt": requirement_pt,
            "label":          item.label,
            "nfr_category":   item.nfr_category or "",
            "raw_label":      item.raw_label,
            "source":         item.source,
        })

    # ── Salva ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Salvo em: %s", output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    df = translate_dataset()

    print(f"\nTotal traduzido: {len(df)}")
    print(f"FR:  {(df['label'] == 'FR').sum()}")
    print(f"NFR: {(df['label'] == 'NFR').sum()}")
    print(f"\nExemplo FR:")
    print(df[df["label"] == "FR"].iloc[0][["requirement_en", "requirement_pt"]].to_string())
    print(f"\nExemplo NFR/Performance:")
    print(df[df["raw_label"] == "PE"].iloc[0][["requirement_en", "requirement_pt"]].to_string())
