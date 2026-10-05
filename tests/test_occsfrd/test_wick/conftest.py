import pytest
from occsfrd import wick

#Pytest fixtures for index objects
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

#Pytest fixtures for operator objects
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

#Pytest fixtures for tensor objects
@pytest.fixture
def ATensor():
    A = wick.tensor.Tensor("A", ["g", "p"], ["g", "h"])
    A.getShape([1,1,0])
    return A

@pytest.fixture
def BTensor():
    B = wick.tensor.Tensor("B", ["g", "v"], ["p", "a"])
    B.getShapeActive((3,2), 6)
    return B

@pytest.fixture
def twoBody_hhpp():
    g = wick.tensor.Tensor("g", ["h", "h"], ["p", "p"])
    return g

@pytest.fixture
def amplitude_pphh():
    t = wick.tensor.Tensor("t", ["p", "p"], ["h", "h"])
    return t

@pytest.fixture
def ccd_gt_uncontracted(twoBody_hhpp, amplitude_pphh):
    return twoBody_hhpp * amplitude_pphh

@pytest.fixture
def ccd_directTerm(ccd_gt_uncontracted):
    lower = ccd_gt_uncontracted.lowerIndices
    upper = ccd_gt_uncontracted.upperIndices

    directContractions = [(lower[0], upper[2]), (lower[1], upper[3]), (lower[2], upper[0]), (lower[3], upper[1])]
    return wick.tensor.TensorProduct(ccd_gt_uncontracted.TensorList, contractions=directContractions)

@pytest.fixture
def ccd_exchangeTerm(ccd_gt_uncontracted):
    lower = ccd_gt_uncontracted.lowerIndices
    upper = ccd_gt_uncontracted.upperIndices

    directContractions = [(lower[0], upper[3]), (lower[1], upper[2]), (lower[2], upper[0]), (lower[3], upper[1])]
    return wick.tensor.TensorProduct(ccd_gt_uncontracted.TensorList, contractions=directContractions)