import numpy as np
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
    g = wick.tensor.Tensor("g", ["h", "h"], ["p", "p"], distinguishableParticles=False)
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
    lower = ccd_gt_uncontracted.freeLowerIndices
    upper = ccd_gt_uncontracted.freeUpperIndices

    directContractions = [(lower[0], upper[2]), (lower[1], upper[3]), (lower[2], upper[0]), (lower[3], upper[1])]
    return wick.tensor.TensorProduct(ccd_gt_uncontracted.tensorList, contractionsList=directContractions)

@pytest.fixture
def ccd_exchangeTerm(ccd_gt_uncontracted):
    lower = ccd_gt_uncontracted.freeLowerIndices
    upper = ccd_gt_uncontracted.freeUpperIndices

    exchangeContractions = [(lower[0], upper[3]), (lower[1], upper[2]), (lower[2], upper[0]), (lower[3], upper[1])]
    return wick.tensor.TensorProduct(ccd_gt_uncontracted.tensorList, contractionsList=exchangeContractions)

@pytest.fixture
def ccd_exchangeTerm1(ccd_gt_uncontracted):
    lower = ccd_gt_uncontracted.freeLowerIndices
    upper = ccd_gt_uncontracted.freeUpperIndices

    exchangeContractions = [(lower[0], upper[2]), (lower[1], upper[3]), (lower[2], upper[1]), (lower[3], upper[0])]
    return wick.tensor.TensorProduct(ccd_gt_uncontracted.tensorList, contractionsList=exchangeContractions)

# Fixtures for contraction tests
@pytest.fixture
def o1(basic_ann_p0a):
    return basic_ann_p0a

@pytest.fixture
def o2(basic_cre_p0a):
    return basic_cre_p0a

@pytest.fixture
def operatorList(basic_ann_p0a, basic_cre_p0a, basic_ann_p1a, basic_cre_p1a):
    return [basic_ann_p0a, basic_cre_p0a, basic_ann_p1a, basic_cre_p1a]

@pytest.fixture
def operatorList_(operatorList):
    return operatorList

@pytest.fixture
def prefactor():
    return 1.0

@pytest.fixture
def existingContractions():
    return []

@pytest.fixture
def normalOrderedStartPoints():
    return []

@pytest.fixture
def operatorProduct(operatorList):
    return wick.operator.OperatorProduct(operatorList)

@pytest.fixture
def operatorProduct_(operatorProduct):
    return operatorProduct

@pytest.fixture
def operator_(operatorProduct):
    return operatorProduct

@pytest.fixture
def contractionsListFortran(index_p0, index_p1):
    return [2, 0, 4, 0]

@pytest.fixture
def contractionsArray():
    return np.array([[2], [0], [4], [0]], dtype=int)

@pytest.fixture
def signFlips():
    return np.array([0], dtype=int)

@pytest.fixture
def freeIndexTypes():
    return (['p'], ['p'])

@pytest.fixture
def tensorProduct():
    tensor_ = wick.tensor.Tensor('A', ['g'], ['g'])
    tensor_.getShape([1, 0])
    tensor_.setArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    return wick.tensor.TensorProduct([tensor_])

@pytest.fixture
def tensorProduct_(tensorProduct):
    return tensorProduct

@pytest.fixture
def term(tensorProduct):
    return tensorProduct

@pytest.fixture
def vertex(tensorProduct):
    return tensorProduct.vertexList[0]

@pytest.fixture
def index(vertex):
    return vertex.lowerIndices[0]

@pytest.fixture
def lowerIndex(tensorProduct):
    return tensorProduct.freeLowerIndices[0]

@pytest.fixture
def upperIndex(tensorProduct):
    return tensorProduct.freeUpperIndices[0]

@pytest.fixture
def lowerIndexList(tensorProduct):
    return list(tensorProduct.freeLowerIndices)

@pytest.fixture
def upperIndexList(tensorProduct):
    return list(tensorProduct.freeUpperIndices)

@pytest.fixture
def contractionsList(lowerIndex, upperIndex):
    return [(lowerIndex, upperIndex)]

@pytest.fixture
def array():
    return np.arange(4).reshape(2, 2)

@pytest.fixture
def slice():
    return np.s_[0:1, :]

@pytest.fixture
def tensorSum_(tensorProduct):
    return wick.tensor.TensorSum([tensorProduct])