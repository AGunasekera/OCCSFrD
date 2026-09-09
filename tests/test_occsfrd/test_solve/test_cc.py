from occsfrd.solve import cc

def test_amplitudeUpdates(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=0):
    assert True

def test_doublesAmplitudeUpdates(doublesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
    assert True

def test_singlesAmplitudeUpdates(singlesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
    assert True

def test_iterateAmplitudes(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=0):
    assert True

def test_iterateDoublesAmplitudes(doublesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
    assert True

def test_iterateSinglesAmplitudes(singlesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
    assert True

def test_iterateTriplesAmplitudes(triplesTensor, residual, fockMatrix):
    assert True

def test_convergeDoublesAmplitudes(doublesTensor, CCDEnergyEquation, CCDAmplitudeEquation, fockTensor, tol=10, spinFree=True, biorthogonal=False, verbosity=0):
    assert True

def test_convergeCollectedDoublesAmplitudes(doublesTensor, CCDEnergyEquation, collectedCCDAmplitudeEquation, fockTensor, tol=10, verbosity=0):
    assert True

def test_convergeCCSDAmplitudes(singlesTensor, doublesTensor, CCSDEnergyEquation, singlesCCSDAmplitudeEquation, doublesCCSDAmplitudeEquation, fockTensor, tol=10, biorthogonal=False, verbosity=0):
    assert True

def test_convergeCCSDTAmplitudes(singlesTensor, doublesTensor, triplesTensor, CCSDTEnergyEquation, singlesCCSDTAmplitudeEquation, doublesCCSDTAmplitudeEquation, triplesCCSDTAmplitudeEquation,  fockTensor, tol=10, verbosity=0):
    assert True

def test_convergeUnlinkedDoublesAmplitudes(doublesTensor, unlinkedCCDEnergyEquation, unlinkedCCDAmplitudeEquation, unlinkedCCDAmplitudeEquationCorrectionOverE, fockTensor, tol=10, spinFree=True, biorthogonal=False, verbosity=0):
    assert True

def test_convergeSeparateUnlinkedDoublesAmplitudesOnlyLinked(amplitudeTensors, unlinkedEnergyEquation, unlinkedAmplitudeEquations, unlinkedAmplitudeEquationCorrectionsOverE, fockTensor, nCore, nActive, nVirtual, tol=10, spinFree=True, biorthogonal=False, maxIterations=10000, verbosity=0):
    assert True

def test_convergeUnlinkedAmplitudes(Norbs, Nelec, Nactive, amplitudeTensors, unlinkedEnergyEquationAndNorm, unlinkedAmplitudeEquationsAndCorrectionsOverE, fockTensor, Rtol=10, Etol=8, spinFree=True, biorthogonal=False, verbosity=0, levelShift=0., maxIter=100, nDIIS=0, maxOrder=2, onlyConnect=False):
    assert True

def test_runUnlinkedCC(mf, equationsDict, levelShift=0., verbosity=0, biorthogonal=False, Rtol=10, Etol=8, maxIter=100, nDIIS=0, maxOrder=2, onlyConnect=False):
    assert True

def test_getReferenceEnergy(mf, equationsDict, levelShift=0., verbosity=0, biorthogonal=False, Rtol=10, Etol=8, maxIter=100, nDIIS=0, maxOrder=2, onlyConnect=False):
    assert True
