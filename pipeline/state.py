from typing import TypedDict, List
from pydantic import BaseModel


class Requirement(BaseModel):
    id: str
    text: str
    source: str = ""


class ClassifiedRequirement(Requirement):
    category: str          # FR ou NFR
    subcategory: str = ""  # ex: Performance, Security, Usability
    confidence: float = 0.0


class PrioritizedRequirement(ClassifiedRequirement):
    priority_score: float = 0.0
    priority_rank: int = 0
    method: str = ""       # ex: MoSCoW, AHP


class PipelineState(TypedDict):
    raw_input: str
    requirements: List[Requirement]
    classified_requirements: List[ClassifiedRequirement]
    prioritized_requirements: List[PrioritizedRequirement]
    errors: List[str]
    metadata: dict
