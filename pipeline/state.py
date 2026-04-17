from typing import TypedDict, List, Annotated
from pydantic import BaseModel
from enum import Enum
import operator


class CategoryEnum(str, Enum):
    FR  = "FR"
    NFR = "NFR"


class Requirement(BaseModel):
    id:     str
    text:   str
    source: str = ""


class ClassifiedRequirement(Requirement):
    category:    CategoryEnum
    subcategory: str   = ""
    confidence:  float = 0.0


class PrioritizedRequirement(ClassifiedRequirement):
    priority_score: float = 0.0
    priority_rank:  int   = 0
    method:         str   = ""
    justification:  str   = ""  # raciocínio do LLM para a prioridade atribuída


class PipelineState(TypedDict):
    raw_input:                str
    requirements:             Annotated[List[Requirement],             operator.add]
    classified_requirements:  Annotated[List[ClassifiedRequirement],  operator.add]
    prioritized_requirements: Annotated[List[PrioritizedRequirement], operator.add]
    errors:                   Annotated[List[str],                    operator.add]
    metadata:                 dict
