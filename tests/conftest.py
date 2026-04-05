import pytest
from pathlib import Path

from datasets.promise import PromiseAdapter
from pipeline.state import (
    Requirement,
    ClassifiedRequirement,
    PrioritizedRequirement,
    CategoryEnum,
)

# ── Paths do dataset ───────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
_ARFF_PATH  = _ROOT / "datasets" / "promise_nfr" / "Promise+.arff"
_SCHEMA     = _ROOT / "datasets" / "schemas" / "promise.yml"
_XLSX_PATH  = _ROOT / "datasets" / "promise_nfr" / "Lista dos tipos de software.xlsx"

@pytest.fixture(scope="session")
def promise_data():
    adapter = PromiseAdapter(
        arff_path=_ARFF_PATH,
        schema_path=_SCHEMA,
        xlsx_path=_XLSX_PATH,
    )
    return adapter.load()

def _to_requirement(item, idx: int) -> Requirement:
    return Requirement(
        id=f"PROMISE-{item.project_id}-{idx:03d}",
        text=item.requirement,
        source=f"promise_nfr/project_{item.project_id}",
    )

def _find(items, raw_label: str, offset: int = 0):
    matches = [i for i in items if i.raw_label == raw_label]
    return matches[offset]

@pytest.fixture
def req_fr(promise_data):
    item = _find(promise_data.items, raw_label="F")
    return _to_requirement(item, idx=1)

@pytest.fixture
def req_nfr_performance(promise_data):
    item = _find(promise_data.items, raw_label="PE")
    return _to_requirement(item, idx=2)

@pytest.fixture
def req_nfr_security(promise_data):
    item = _find(promise_data.items, raw_label="SE")
    return _to_requirement(item, idx=3)

@pytest.fixture
def req_nfr_scalability(promise_data):
    item = _find(promise_data.items, raw_label="SC")
    return _to_requirement(item, idx=4)

@pytest.fixture
def req_nfr_usability(promise_data):
    item = _find(promise_data.items, raw_label="US")
    return _to_requirement(item, idx=5)

@pytest.fixture
def classified_fr(req_fr):
    return ClassifiedRequirement(
        **req_fr.model_dump(),
        category=CategoryEnum.FR,
        subcategory="",
        confidence=0.97,
    )

@pytest.fixture
def classified_nfr_performance(req_nfr_performance):
    return ClassifiedRequirement(
        **req_nfr_performance.model_dump(),
        category=CategoryEnum.NFR,
        subcategory="Performance",
        confidence=0.95,
    )

@pytest.fixture
def classified_nfr_security(req_nfr_security):
    return ClassifiedRequirement(
        **req_nfr_security.model_dump(),
        category=CategoryEnum.NFR,
        subcategory="Security",
        confidence=0.98,
    )


@pytest.fixture
def prioritized_req(classified_nfr_security):
    return PrioritizedRequirement(
        **classified_nfr_security.model_dump(),
        priority_score=0.95,
        priority_rank=1,
        method="MoSCoW",
    )
