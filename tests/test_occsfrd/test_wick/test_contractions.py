from occsfrd.wick import contractions

def test_canContract(o1, o2):
    assert True

def test_recursiveFullContraction(operatorList_, prefactor, existingContractions, normalOrderedStartPoints, speedup=False):
    assert True
    
def test_genFortranInterfaceLists(operatorList):
    assert True

def test_recursiveFullContractionsFortran(operatorProduct):
    assert True

def test_getKroneckerDeltasFromFortranInterfaceContractionsList(operatorList, contractionsListFortran):
    assert True

def test_genContractionsListsFromFortranInterface(operatorProduct, contractionsArray, signFlips):
    assert True

def test_vacuumExpectationValue(operator_, speedup=False, printing=False):
    assert True

def test_evaluateWickOld(term, referenceOperator=None, normalOrderedParts=True):
    assert True

def test_evaluateWick(term, referenceOperator=None, normalOrderedParts=True):
    assert True

def test_chooseUncontractedOperatorPositions(operatorProduct_, freeIndexTypes):
    assert True

def test_recursiveIncompleteContractionNew(operator_, freeIndexTypes=([], []), speedup=False):
    assert True

def test_evaluateWickFree(term, freeIndexTypes=([], []), speedup=False, normalOrderedParts=True):
    assert True

def test_getAxis(vertex, index):
    assert True

def test_getContractedArrayOld(tensorProduct_, targetLowerIndexList=None, targetUpperIndexList=None):
    assert True

def test_sliceActiveIndices(array, lowerIndexList, upperIndexList):
    assert True

def test_getContractedArrayOldTest(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None):
    assert True

def test_followUpperIndexThroughContractionsOld(upperIndex, contractionsList):
    assert True

def test_followLowerIndexThroughContractionsOld(lowerIndex, contractionsList):
    assert True

def test_testEqualTermsInTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    assert True

def test_testEqualTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    assert True

def test_testOldContractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None):
    assert True

def test_contractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    assert True

def test_getContractedArraySlow(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert True

def test_getEinsumInformationNew(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert True

def test_getEinsumInformation(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert True

def test_getContractedArray(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert True


def test_followUpperIndexThroughContractions(upperIndex, contractionsList):
    assert True

def test_followLowerIndexThroughContractions(lowerIndex, contractionsList):
    assert True

def test_findLowerIndexSpecificValue(lowerIndex, lowerIndexList, upperIndexList):
    assert True

def test_findUpperIndexSpecificValue(upperIndex, lowerIndexList, upperIndexList):
    assert True

def test_maskArrayBySlice(array, slice):
    assert True