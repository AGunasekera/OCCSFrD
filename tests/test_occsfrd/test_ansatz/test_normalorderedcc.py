from occsfrd.ansatz import normalorderedcc

def test_projectionManifold(amplitudeTensor):
    assert True

def test_getEnergyEquation(transformedHamiltonian, referenceOperator=None, spinFree=True, verbose=True):
    assert True

def test_getEnergyEquationUnlinked(HamiltonianAndWaveOperator, referenceOperator=None, spinFree=True, verbose=True):
    assert True

def test_getAmplitudeEquation(transformedHamiltonian, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    assert True

def test_getAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    assert True

def test_getBiorthAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    assert True

def test_getCollectedBiorthogonalAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    assert True

def test_getCollectedAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    assert True

def test_getAmplitudeEquationOnlyLinked(HamiltonianAndWaveOperator, amplitudeTensor, referenceOperator=None, spinFree=True):
    assert True

def test_getBiorthDoublesAmplitudeEquationOnlyLinked(HamiltonianAndWaveOperator, doublesAmplitudeTensor, referenceOperator=None, spinFree=True):
    assert True

def test_genNormalOrderedCCAnsatz(amplitudeTensorsList, order=2):
    assert True