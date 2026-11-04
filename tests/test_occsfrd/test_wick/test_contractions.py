from occsfrd.wick import contractions

def test_canContract(o1, o2):
    assert False

def test_recursiveFullContraction(operatorList_, prefactor, existingContractions, normalOrderedStartPoints, speedup=False):
    assert False
    
def test_genFortranInterfaceLists(operatorList):
    assert False

def test_recursiveFullContractionsFortran(operatorProduct):
    assert False

def test_getKroneckerDeltasFromFortranInterfaceContractionsList(operatorList, contractionsListFortran):
    assert False

def test_genContractionsListsFromFortranInterface(operatorProduct, contractionsArray, signFlips):
    assert False

def test_vacuumExpectationValue(operator_, speedup=False, printing=False):
    assert False

def test_evaluateWickOld(term, referenceOperator=None, normalOrderedParts=True):
    assert False

def test_evaluateWick(term, referenceOperator=None, normalOrderedParts=True):
    assert False

def test_chooseUncontractedOperatorPositions(operatorProduct_, freeIndexTypes):
    assert False

def test_recursiveIncompleteContractionNew(operator_, freeIndexTypes=([], []), speedup=False):
    assert False

def test_evaluateWickFree(term, freeIndexTypes=([], []), speedup=False, normalOrderedParts=True):
    assert False

def test_getAxis(vertex, index):
    assert False

def test_getContractedArrayOld(tensorProduct_, targetLowerIndexList=None, targetUpperIndexList=None):
    assert False

def test_sliceActiveIndices(array, lowerIndexList, upperIndexList):
    assert False

def test_getContractedArrayOldTest(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None):
    assert False

def test_followUpperIndexThroughContractionsOld(upperIndex, contractionsList):
    assert False

def test_followLowerIndexThroughContractionsOld(lowerIndex, contractionsList):
    assert False

def test_testEqualTermsInTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    assert False

def test_testEqualTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    assert False

def test_testOldContractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None):
    assert False

def test_contractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    assert False

def test_getContractedArraySlow(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert False

def test_getEinsumInformationNew(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert False

def test_getEinsumInformation(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert False

def test_getContractedArray(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    assert False

def test_followUpperIndexThroughContractions(upperIndex, contractionsList):
    assert False

def test_followLowerIndexThroughContractions(lowerIndex, contractionsList):
    assert False

def test_findLowerIndexSpecificValue(lowerIndex, lowerIndexList, upperIndexList):
    assert False

def test_findUpperIndexSpecificValue(upperIndex, lowerIndexList, upperIndexList):
    assert False

def test_maskArrayBySlice(array, slice):
    assert False