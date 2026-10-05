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

    assert np.allclose(
        result,
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 1.0],
         [1.0, 1.0, 0.0]],
    )

def test_getDIISWeights(oldErrorVecs):
    result = diis.getDIISWeights(oldErrorVecs)

    assert np.allclose(
        result,
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 1.0],
         [1.0, 1.0, 0.0]],
    )

def test_updateAmpsDIIS(weights, oldAmplitudes, oldErrorVecs):
    result = diis.updateAmpsDIIS(oldErrorVecs)

    assert np.allclose(
        result,
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 1.0],
         [1.0, 1.0, 0.0]],
    )