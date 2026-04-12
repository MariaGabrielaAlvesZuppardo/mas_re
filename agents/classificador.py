"""
Agente de classificação de requisitos de software FR/NFR 

Recebe um Requisito do PipelineState, chama o LLLM via Langchain e retorna um Classificador validado via Pydantic
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from groq import RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from pydantic import BaseModel

from config.models import AVAILABLE_MODELS
from config.settings import settings
from pipeline.state import (
    CategoryEnum,
    ClassifiedRequirement,
    PipelineState,
    Requirement
)
from prompts.classificacao import classification_prompt

logger = logging.getLogger(__name__)

class ClassificationOutput(BaseModel):
    category:    CategoryEnum
    subcategory: str   = ""
    confidence:  float = 0.0


def _build_llm(model_id: str, model_info: dict):
    provider = model_info.get("provider")

    if provider == "anthropic":
        return ChatAnthropic(
            model=model_id,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
        )
    elif provider == "groq":
        return ChatGroq(
            model=model_id,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
        )
    else:
        raise ValueError(f"Provider não suportado: {provider!r}")


class ClassificationAgent:

    def __init__(self, model: str | None = None, max_workers: int = 5):
        model_id   = model or settings.classifier_model
        model_info = AVAILABLE_MODELS.get(model_id, {})

        llm = _build_llm(model_id, model_info)

        self._chain      = classification_prompt | llm.with_structured_output(
            ClassificationOutput
        )
        self._max_workers = max_workers

    def classify(self, requirement: Requirement) -> ClassifiedRequirement:
        """
        Classifica um único requisito.
        Retentar automaticamente em caso de RateLimitError (429) com backoff exponencial.
        """
        logger.debug("Classificando: %s", requirement.id)

        max_attempts = 5
        wait = 60  # segundos iniciais de espera

        for attempt in range(1, max_attempts + 1):
            try:
                output: ClassificationOutput = self._chain.invoke({
                    "requirement": requirement.text,
                })
                return ClassifiedRequirement(
                    **requirement.model_dump(),
                    category=output.category,
                    subcategory=output.subcategory,
                    confidence=output.confidence,
                )
            except RateLimitError as exc:
                if attempt == max_attempts:
                    raise
                logger.warning(
                    "RateLimitError em %s (tentativa %d/%d) — aguardando %ds: %s",
                    requirement.id, attempt, max_attempts, wait, exc,
                )
                time.sleep(wait)
                wait *= 2  # backoff exponencial: 60s, 120s, 240s, 480s

    def classify_batch(
        self, requirements: list[Requirement]
    ) -> tuple[list[ClassifiedRequirement], list[str]]:
        """
        Classifica uma lista de requisitos em paralelo.
        Retorna (classificados, erros) preservando a ordem da lista original.
        """
        classified: list[ClassifiedRequirement | None] = [None] * len(requirements)
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self.classify, req): i
                for i, req in enumerate(requirements)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    classified[idx] = future.result()
                except Exception as exc:
                    msg = f"Erro ao classificar {requirements[idx].id}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)

        return [r for r in classified if r is not None], errors

    def run(self, state: PipelineState) -> dict[str, Any]:
        """
        Nó do grafo LangGraph.
        Classifica todos os requisitos do estado e retorna o delta.
        """
        classified, errors = self.classify_batch(state["requirements"])

        logger.info(
            "Classificados: %d/%d requisitos",
            len(classified),
            len(state["requirements"]),
        )

        return {
            "classified_requirements": classified,
            "errors": errors,
        }
