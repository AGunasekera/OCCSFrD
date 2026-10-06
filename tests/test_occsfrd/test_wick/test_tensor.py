import numpy as np
from occsfrd.wick import tensor

def test_getShape(ATensor):
    #Tensor is A_gp^gh, in 2 occupied and 1 virtual orbital

    assert ATensor.array.shape == (3,1,3,2)
    assert np.all(ATensor.array == 0)

def test_getShapeActive(BTensor):
    #Tensor is B_gv^pa, in 2 doubly occupied, 1 singly occpied, and 3 virtual orbitals

    assert BTensor.array.shape == (6,3,4,1)
    assert np.all(BTensor.array == 0)

def test_getAndSetArray():
    fTensor = tensor.Tensor("f", ["g"], ["g"])
    assert fTensor.getArray() is None

    fTensor.getShape([1, 0, 0])
    assert np.all(fTensor.getArray() == np.array([0, 0, 0], [0, 0, 0], [0, 0, 0]))

    fTensor.setArray(np.array([1, 1], [1, 1]))
    assert np.all(fTensor.getArray() == np.array([0, 0, 0], [0, 0, 0], [0, 0, 0]))

    fTensor.setArray(np.array([1, 2, 3], [4, 5, 6], [7, 8, 9]))
    assert np.all(fTensor.getArray() == np.array([1, 2, 3], [4, 5, 6], [7, 8, 9]))


def test_getOperator():
    assert True

def test_getDiagrams():
    assert True

def test_getAllDiagrams():
    assert True 

def test_getAllDiagramsGeneral():
    assert True

def test_getAllDiagramsActive():
    assert True
    
def test_assignDiagramArrays():
    assert True
    
def test_assignDiagramArraysActive():
    assert True

def test_conjugate():
    assert True

def test_setSlices():
    assert True

def test_getArraySubDiagram():
    assert True

def test_setArraySubDiagram():
    assert True  

def test_calculateArray():
    assert True

def test_getOperatorVertex():
    assert True

def test_applyContraction():
    assert True

def test_addNewIndex():
    assert True 

def test_getVertexList():
    assert True

def test_getOperatorTensorProduct():
    assert True

def test_getVacuumExpectationValue():
    assert True

def test_getGraph():
    assert True

def test_getGraphOld():
    assert True

def test_drawGraph():
    assert True

def test_nodeMatch():
    assert True

def test_edgeMatch():
    assert True

def test_isProportional(ccd_directTerm, ccd_exchangeTerm, ccd_exchangeTerm1):
    assert not ccd_directTerm.isProportional(ccd_exchangeTerm)
    assert ccd_exchangeTerm.isProportional(ccd_exchangeTerm1)

def test_followPropagation():
    assert True

def test_getFreeIndexPairs():
    assert True

def test_isProportional1():
    assert True

def test_isConnected():
    assert True

def getOperatorTensorSum():
    assert True

def test_collectIsomorphicTerms():
    assert True

def test_getConnectedTerms():
    assert True
    
def test_collectConnectedIsomorphicTerms():
    assert True