import operator
import pytest
from pydantic import ValidationError
from pipeline.state import (
    Requirement,
    ClassifiedRequirement,
    PrioritizedRequirement,
    CategoryEnum,
    PipelineState,
)


def test_requirement_fr_carregado_do_promise(req_fr):
    assert req_fr.id.startswith("PROMISE-")
    assert req_fr.text != ""
    assert "promise_nfr" in req_fr.source

def test_requirement_nfr_performance_carregado(req_nfr_performance):
    assert req_nfr_performance.id.startswith("PROMISE-")
    assert req_nfr_performance.text != ""

def test_requirement_sem_id_falha():
    with pytest.raises(ValidationError):
        Requirement(text="The system shall authenticate users.")

def test_requirement_sem_text_falha():
    with pytest.raises(ValidationError):
        Requirement(id="REQ-001")

def test_requirement_source_padrao():
    req = Requirement(id="REQ-001", text="The system shall allow login.")
    assert req.source == ""


def test_classified_fr(classified_fr):
    assert classified_fr.category == CategoryEnum.FR
    assert classified_fr.subcategory == ""
    assert classified_fr.confidence == 0.97

def test_classified_nfr_performance(classified_nfr_performance):
    assert classified_nfr_performance.category == CategoryEnum.NFR
    assert classified_nfr_performance.subcategory == "Performance"
    assert classified_nfr_performance.confidence == 0.95

def test_classified_nfr_security(classified_nfr_security):
    assert classified_nfr_security.category == CategoryEnum.NFR
    assert classified_nfr_security.subcategory == "Security"

def test_classified_herda_campos_requirement(classified_nfr_performance, req_nfr_performance):
    assert classified_nfr_performance.id    == req_nfr_performance.id
    assert classified_nfr_performance.text  == req_nfr_performance.text
    assert classified_nfr_performance.source == req_nfr_performance.source

def test_classified_category_invalida():
    with pytest.raises(ValidationError):
        ClassifiedRequirement(
            id="REQ-001",
            text="The system shall respond in 2 seconds.",
            category="INVALIDO",
        )


def test_prioritized_herda_classified(prioritized_req, classified_nfr_security):
    assert prioritized_req.id         == classified_nfr_security.id
    assert prioritized_req.category   == classified_nfr_security.category
    assert prioritized_req.subcategory == classified_nfr_security.subcategory

def test_prioritized_moscow(prioritized_req):
    assert prioritized_req.priority_rank  == 1
    assert prioritized_req.method         == "MoSCoW"
    assert prioritized_req.priority_score == 0.95

def test_prioritized_valores_padrao(promise_data):
    item = next(i for i in promise_data.items if i.raw_label == "A")
    req  = Requirement(
        id=f"PROMISE-{item.project_id}-999",
        text=item.requirement,
        source=f"promise_nfr/project_{item.project_id}",
    )
    prioritized = PrioritizedRequirement(
        **req.model_dump(),
        category=CategoryEnum.NFR,
    )
    assert prioritized.priority_score == 0.0
    assert prioritized.priority_rank  == 0
    assert prioritized.method         == ""

def test_pipeline_state_estrutura_inicial():
    state: PipelineState = {
        "raw_input": "Requirements from PROMISE+ project 1",
        "requirements": [],
        "classified_requirements": [],
        "prioritized_requirements": [],
        "errors": [],
        "metadata": {
            "model_used": "claude-sonnet-4-6",
            "dataset": "promise_nfr",
            "run_id": "exp-001",
        },
    }
    assert state["raw_input"] == "Requirements from PROMISE+ project 1"
    assert state["metadata"]["dataset"] == "promise_nfr"

def test_pipeline_state_acumula_requirements(req_fr, req_nfr_performance):
    resultado = operator.add([req_fr], [req_nfr_performance])
    assert len(resultado) == 2
    assert resultado[0].id != resultado[1].id

def test_pipeline_state_acumula_erros():
    erros_a = ["Classificação falhou para PROMISE-1-002"]
    erros_b = ["Priorizador sem output para PROMISE-2-001"]
    assert len(operator.add(erros_a, erros_b)) == 2

def test_pipeline_state_completo(
    req_fr, req_nfr_security,
    classified_fr, classified_nfr_security,
    prioritized_req,
):
    state: PipelineState = {
        "raw_input": "PROMISE+ project 2 — security requirements",
        "requirements":            [req_fr, req_nfr_security],
        "classified_requirements": [classified_fr, classified_nfr_security],
        "prioritized_requirements":[prioritized_req],
        "errors": [],
        "metadata": {
            "model_used": "claude-opus-4-6",
            "dataset":    "promise_nfr",
            "run_id":     "exp-002",
            "total_tokens": 2100,
        },
    }
    assert len(state["requirements"])             == 2
    assert len(state["classified_requirements"])  == 2
    assert len(state["prioritized_requirements"]) == 1
    assert state["metadata"]["total_tokens"]      == 2100
