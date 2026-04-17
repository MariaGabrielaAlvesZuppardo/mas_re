"""
Agente de priorização de requisitos de software via metodo MoSCoW

Recebe um ClassifiedRequirement do PipelineState, chama o LLM via LangChain
e retorna um PrioritizedRequirement validado via Pydantic
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from groq import RateLimitError
from pydantic import BaseModel

from config.models import AVAILABLE_MODELS
from config.settings import settings
from pipeline.state import (
    ClassifiedRequirement,
    PipelineState,
    PrioritizedRequirement,
)
from prompts.priorizacao import prioritization_prompt

logger = logging.getLogger(__name__)


class PrioritizationOutput(BaseModel):
    method:         str   = "MoSCoW"
    priority_score: float = 0.0
    priority_rank:  int   = 0
    justification:  str   = ""


class PriorizationAgent:

    def __init__(self, model: str | None = None, max_workers: int = 5):
        from agents.classificador import _build_llm

        model_id   = model or settings.classifier_model
        model_info = AVAILABLE_MODELS.get(model_id, {})

        llm = _build_llm(model_id, model_info)

        self._chain       = prioritization_prompt | llm.with_structured_output(
            PrioritizationOutput
        )
        self._max_workers = max_workers

    def prioritize(self, requirement: ClassifiedRequirement) -> PrioritizedRequirement:
        """
        Prioriza um único requisito já classificado.
        Retentar automaticamente em caso de RateLimitError (429) com backoff exponencial.
        """
        logger.debug("Priorizando: %s", requirement.id)

        max_attempts = 5
        wait = 60

        for attempt in range(1, max_attempts + 1):
            try:
                output: PrioritizationOutput = self._chain.invoke({
                    "requirement": requirement.text,
                    "category":    requirement.category.value,
                    "subcategory": requirement.subcategory,
                })
                return PrioritizedRequirement(
                    **requirement.model_dump(),
                    priority_score=output.priority_score,
                    priority_rank=output.priority_rank,
                    method=output.method,
                    justification=output.justification,
                )
            except RateLimitError as exc:
                if attempt == max_attempts:
                    raise
                logger.warning(
                    "RateLimitError em %s (tentativa %d/%d) — aguardando %ds: %s",
                    requirement.id, attempt, max_attempts, wait, exc,
                )
                time.sleep(wait)
                wait *= 2

    def prioritize_batch(
        self, requirements: list[ClassifiedRequirement]
    ) -> tuple[list[PrioritizedRequirement], list[str]]:
        """
        Prioriza uma lista de requisitos em paralelo.
        Retorna (priorizados, erros) preservando a ordem da lista original.
        """
        prioritized: list[PrioritizedRequirement | None] = [None] * len(requirements)
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self.prioritize, req): i
                for i, req in enumerate(requirements)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    prioritized[idx] = future.result()
                except Exception as exc:
                    msg = f"Erro ao priorizar {requirements[idx].id}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)

        return [r for r in prioritized if r is not None], errors

    def run(self, state: PipelineState) -> dict[str, Any]:
        """
        Nó do grafo LangGraph.
        Prioriza todos os requisitos classificados e retorna o delta.
        """
        prioritized, errors = self.prioritize_batch(state["classified_requirements"])

        logger.info(
            "Priorizados: %d/%d requisitos",
            len(prioritized),
            len(state["classified_requirements"]),
        )

        return {
            "prioritized_requirements": prioritized,
            "errors": errors,
        }
