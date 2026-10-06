import numpy as np
import pytest
from occsfrd.solve import diis

@pytest.fixture
def oldErrorVecs():
    return [[np.array([[1.0, 2.0]])], [np.array([[3.0, 4.0]])]]

def test_overlapMatrix(oldErrorVecs):
    # errors = [
    #     [np.array([[1.0, 2.0]])],
    #     [np.array([[3.0, 4.0]])],
    # ]

    result = diis.overlapMatrix(oldErrorVecs)

    expected = np.array([
        [5.0, 11.0],
        [11.0, 25.0],
    ])
    assert np.allclose(result, expected)

def test_LagrangianMatrix(oldErrorVecs):
    # errors = [[np.array([[1.0]])], [np.array([[2.0]])]]

    result = diis.LagrangianMatrix(oldErrorVecs)

    expected = np.array([
        [5.0, 11.0, 1.0],
        [11.0, 25.0, 1.0],
        [1.0, 1.0, 0.0]
    ])
    assert np.allclose(result, expected)

def test_getDIISWeights(oldErrorVecs):
    weights = diis.getDIISWeights(oldErrorVecs)
    augmentedWeights = np.array(list(weights) + [-0.5])

    LMatrix = diis.LagrangianMatrix(oldErrorVecs)

    result = np.matmul(LMatrix, augmentedWeights)
    expected = np.array([0.0, 0.0, 1.0])
    assert np.allclose(result, expected)

def test_updateAmpsDIIS(oldErrorVecs):
    weights = np.array([0.5, 0.5])
    oldAmplitudes = [np.array([[1.0, 3.0]]), np.array([[2.0, 5.0]])]
    result = diis.updateAmpsDIIS(weights, oldAmplitudes, oldErrorVecs)
    assert np.allclose(
        result,
        np.array([3.5, 7.0])
    )

    weights = np.array([0.0, 1.0])
    result = diis.updateAmpsDIIS(weights, oldAmplitudes, oldErrorVecs)
    assert np.allclose(
        result,
        np.array([5.0, 9.0])
    )