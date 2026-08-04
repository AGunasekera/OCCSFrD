from occsfrd.wick import contractions

def test_canContract(o1, o2):
    return

def test_recursiveFullContraction(operatorList_, prefactor, existingContractions, normalOrderedStartPoints, speedup=False):
    return
    
def test_genFortranInterfaceLists(operatorList):
    return

def test_recursiveFullContractionsFortran(operatorProduct):
    return

def test_getKroneckerDeltasFromFortranInterfaceContractionsList(operatorList, contractionsListFortran):
    return

def test_genContractionsListsFromFortranInterface(operatorProduct, contractionsArray, signFlips):
    return

def test_vacuumExpectationValue(operator_, speedup=False, printing=False):
    return

def test_evaluateWickOld(term, referenceOperator=None, normalOrderedParts=True):
    return

def test_evaluateWick(term, referenceOperator=None, normalOrderedParts=True):
    return

def test_chooseUncontractedOperatorPositions(operatorProduct_, freeIndexTypes):
    return

def test_recursiveIncompleteContractionNew(operator_, freeIndexTypes=([], []), speedup=False):
    return

def test_evaluateWickFree(term, freeIndexTypes=([], []), speedup=False, normalOrderedParts=True):
    return

def test_getAxis(vertex, index):
    return

def test_getContractedArrayOld(tensorProduct_, targetLowerIndexList=None, targetUpperIndexList=None):
    return

def test_sliceActiveIndices(array, lowerIndexList, upperIndexList):
    return

def test_getContractedArrayOldTest(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None):
    return

def test_followUpperIndexThroughContractionsOld(upperIndex, contractionsList):
    return

def test_followLowerIndexThroughContractionsOld(lowerIndex, contractionsList):
    return

def test_testEqualTermsInTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    return

def test_testEqualTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    return

def test_testOldContractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None):
    return

def test_contractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    return

def test_getContractedArraySlow(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    return

def test_getEinsumInformationNew(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    return

def test_getEinsumInformation(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    return

def test_getContractedArray(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    return


def test_followUpperIndexThroughContractions(upperIndex, contractionsList):
    return

def test_followLowerIndexThroughContractions(lowerIndex, contractionsList):
    return

def test_findLowerIndexSpecificValue(lowerIndex, lowerIndexList, upperIndexList):
    return

def test_findUpperIndexSpecificValue(upperIndex, lowerIndexList, upperIndexList):
    return

def test_maskArrayBySlice(array, slice):
    return