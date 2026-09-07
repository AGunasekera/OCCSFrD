import numpy as np
from occsfrd.solve import diis

def test_overlapMatrix(oldErrorVecs):
    errors = [
        [np.array([[1.0, 2.0]])],
        [np.array([[3.0, 4.0]])],
    ]

    result = diis.overlapMatrix(errors)

    expected = np.array([
        [5.0, 11.0],
        [11.0, 25.0],
    ])
    np.testing.assert_allclose(result, expected)

def test_LagrangianMatrix(oldErrorVecs):
    errors = [[np.array([[1.0]])], [np.array([[2.0]])]]

    result = diis.LagrangianMatrix(errors)

    np.testing.assert_array_equal(
        result,
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 1.0],
         [1.0, 1.0, 0.0]],
    )

def test_getDIISWeights(oldErrorVecs):
    return

def test_updateAmpsDIIS(weights, oldAmplitudes, oldErrorVecs):
    return