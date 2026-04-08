from unittest.mock import MagicMock

import pytest

from agents.classificador import ClassificationAgent, ClassificationOutput
from datasets.promise import PROMISE_CATEGORY_MAP
from pipeline import CategoryEnum, ClassifiedRequirement

from pipeline.state import Requirement


@pytest.fixture
def mock_chain():
    return MagicMock()
@pytest.fixture
def classificador(mock_chain):
    agent = ClassificationAgent.__new__(ClassificationAgent)
    agent._chain = mock_chain
    return agent

def _ground_truth_output(item)->ClassificationAgent:
    """Cria o output esperado baseado no label real d Promise +"""
    return ClassificationOutput(
        category=CategoryEnum.FR if item.label == "FR" else CategoryEnum.NFR,
        subcategory=PROMISE_CATEGORY_MAP.get(item.raw_label, ""),
        confidence=1.0,  # ground truth = confiança máxima
    )

def test_classify_fr_com_dado_real(classificador, mock_chain, req_fr, item_fr):
    mock_chain.invoke.return_value = _ground_truth_output(item_fr)

    result = classificador.classify(req_fr)

    assert isinstance(result, ClassifiedRequirement)
    assert result.category    == CategoryEnum.FR
    assert result.subcategory == ""
    assert result.text        == item_fr.requirement
    assert result.id          == req_fr.id

def test_classify_nfr_performance_com_dado_real(
    classificador, mock_chain, req_nfr_performance, item_nfr_performance
):
    mock_chain.invoke.return_value = _ground_truth_output(item_nfr_performance)

    result = classificador.classify(req_nfr_performance)

    assert result.category    == CategoryEnum.NFR
    assert result.subcategory == "Performance"
    assert result.text        == item_nfr_performance.requirement

def test_classify_nfr_security_com_dado_real(
    classificador, mock_chain, req_nfr_security, item_nfr_security
):
    mock_chain.invoke.return_value = _ground_truth_output(item_nfr_security)

    result = classificador.classify(req_nfr_security)

    assert result.category    == CategoryEnum.NFR
    assert result.subcategory == "Security"
    assert result.text        == item_nfr_security.requirement


# ── Testa que campos do Requirement são preservados ───────────────────────────

def test_classify_preserva_campos_do_requirement(
    classificador, mock_chain, req_fr, item_fr
):
    mock_chain.invoke.return_value = _ground_truth_output(item_fr)

    result = classificador.classify(req_fr)

    assert result.id     == req_fr.id
    assert result.text   == req_fr.text
    assert result.source == req_fr.source


# ── Testa category inválida do LLM ────────────────────────────────────────────

def test_classify_category_invalida_nao_passa_validacao(
    classificador, mock_chain, req_fr
):
    from pydantic import ValidationError
    mock_chain.invoke.return_value = ClassificationOutput.__new__(ClassificationOutput)
    mock_chain.invoke.side_effect  = ValidationError.from_exception_data(
        title="ClassificationOutput",
        input_type="python",
        line_errors=[],
    )
    with pytest.raises(Exception):
        classificador.classify(req_fr)


# ── Testa run() com amostra real do PROMISE+ ─────────────────────────────────

def test_run_com_sample_promise(classificador, mock_chain, promise_sample):
    requirements = [
        Requirement(
            id=f"PROMISE-{item.project_id}-{i:03d}",
            text=item.requirement,
            source=f"promise_nfr/project_{item.project_id}",
        )
        for i, item in enumerate(promise_sample.items, 1)
    ]

    # Mock retorna o ground truth de cada item na ordem
    mock_chain.invoke.side_effect = [
        _ground_truth_output(item)
        for item in promise_sample.items
    ]

    state = {
        "raw_input":               "PROMISE+ sample",
        "requirements":            requirements,
        "classified_requirements": [],
        "prioritized_requirements":[],
        "errors":                  [],
        "metadata":                {"dataset": "promise_nfr"},
    }

    delta = classificador.run(state)

    assert len(delta["classified_requirements"]) == len(promise_sample.items)
    assert delta["errors"]                       == []

    # Verifica que cada resultado bate com o ground truth
    for result, item in zip(delta["classified_requirements"], promise_sample.items):
        expected_category = CategoryEnum.FR if item.label == "FR" else CategoryEnum.NFR
        assert result.category == expected_category


# ── Testa run() com erro em um item ──────────────────────────────────────────

def test_run_captura_erro_sem_quebrar(classificador, mock_chain, promise_sample):
    items = promise_sample.items[:2]
    from pipeline import Requirement
    requirements = [
        Requirement(
            id=f"PROMISE-{item.project_id}-{i:03d}",
            text=item.requirement,
            source=f"promise_nfr/project_{item.project_id}",
        )
        for i, item in enumerate(items, 1)
    ]

    mock_chain.invoke.side_effect = [
        _ground_truth_output(items[0]),
        RuntimeError("Timeout na API Anthropic"),
    ]

    state = {
        "raw_input":               "PROMISE+ sample",
        "requirements":            requirements,
        "classified_requirements": [],
        "prioritized_requirements":[],
        "errors":                  [],
        "metadata":                {},
    }

    delta = classificador.run(state)

    assert len(delta["classified_requirements"]) == 1
    assert len(delta["errors"])                  == 1
    assert "Timeout na API Anthropic"            in delta["errors"][0]


def test_run_estado_vazio(classificador, mock_chain):
    state = {
        "raw_input":               "",
        "requirements":            [],
        "classified_requirements": [],
        "prioritized_requirements":[],
        "errors":                  [],
        "metadata":                {},
    }

    delta = classificador.run(state)

    assert delta["classified_requirements"] == []
    assert delta["errors"]                  == []
    mock_chain.invoke.assert_not_called()
