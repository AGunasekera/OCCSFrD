import pytest
from occsfrd import wick


@pytest.fixture
def first():
    return wick.index.Index("i", True)

@pytest.fixture
def second():
    return wick.index.Index("i", True)

@pytest.fixture
def different():
    return wick.index.Index("a", False)

@pytest.fixture
def general():
    return wick.index.Index("p", False)

@pytest.fixture
def specific(general):
    return wick.index.SpecificOrbitalIndex("a", contractedFrom=general)

@pytest.fixture
def copied(general, specific):
    return specific.contractedCopy(general)

@pytest.fixture
def index_p0():
    return wick.index.Index("p0", False)

@pytest.fixture
def index_p1():
    return wick.index.Index("p1", False)

@pytest.fixture
def basic_cre_p0a(index_p0):
    return wick.operator.BasicOperator(index_p0, True, True)

@pytest.fixture
def basic_ann_p0a(index_p0):
    return wick.operator.BasicOperator(index_p0, False, True)

@pytest.fixture
def basic_cre_p0b(index_p0):
    return wick.operator.BasicOperator(index_p0, True, False)

@pytest.fixture
def basic_ann_p0b(index_p0):
    return wick.operator.BasicOperator(index_p0, False, False)

@pytest.fixture
def basic_cre_p1a(index_p1):
    return wick.operator.BasicOperator(index_p1, True, True)

@pytest.fixture
def basic_ann_p1a(index_p1):
    return wick.operator.BasicOperator(index_p1, False, True)

@pytest.fixture
def basic_cre_p1b(index_p1):
    return wick.operator.BasicOperator(index_p1, True, False)

@pytest.fixture
def basic_ann_p1b(index_p1):
    return wick.operator.BasicOperator(index_p1, False, False)

