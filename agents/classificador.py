"""
Agente de classificação de requisitos de software FR/NFR 

Recebe um Requisito do PipelineState, chama o LLLM via Langchain e retorna um Classificador validado via Pydantic
"""

from __future__ import annotations

import logging
from typing import Any

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

    def __init__(self, model: str | None = None):
        model_id   = model or settings.classifier_model
        model_info = AVAILABLE_MODELS.get(model_id, {})

        llm = _build_llm(model_id, model_info)

        self._chain = classification_prompt | llm.with_structured_output(
            ClassificationOutput
        )

    def classify(self, requirement: Requirement) -> ClassifiedRequirement:
        """
        Classifica um único requisito.

        Args:
            requirement: Requisito a ser classificado

        Returns:
            ClassifiedRequirement com category, subcategory e confidence
        """
        logger.debug("Classificando: %s", requirement.id)

        output: ClassificationOutput = self._chain.invoke({
            "requirement": requirement.text,
        })

        return ClassifiedRequirement(
            **requirement.model_dump(),
            category=output.category,
            subcategory=output.subcategory,
            confidence=output.confidence,
        )

    def run(self, state: PipelineState) -> dict[str, Any]:
        """
        Nó do grafo LangGraph.
        Classifica todos os requisitos do estado e retorna o delta.
        """
        classified = []
        errors     = []

        for req in state["requirements"]:
            try:
                classified.append(self.classify(req))
            except Exception as exc:
                msg = f"Erro ao classificar {req.id}: {exc}"
                logger.warning(msg)
                errors.append(msg)

        logger.info(
            "Classificados: %d/%d requisitos",
            len(classified),
            len(state["requirements"]),
        )

        return {
            "classified_requirements": classified,
            "errors": errors,
        }