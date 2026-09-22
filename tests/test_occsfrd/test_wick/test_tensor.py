import numpy as np
from occsfrd.wick import tensor

def test_getShape():
    ATensor = tensor.Tensor("A", ["g", "p"], ["g", "h"])
    ATensor.getShape([1,1,0])

    assert ATensor.array.shape == (3,1,3,2)
    assert np.all(ATensor.array == 0)

def test_getShapeActive():
    BTensor = tensor.Tensor("B", ["g", "v"], ["p", "a"])
    BTensor.getShapeActive((3,2), 6)

    assert BTensor.array.shape == (6,3,4,1)
    assert np.all(BTensor.array == 0)

def test_getArray():
    assert True

def test_setArray():
    assert True

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

def getVertexList():
    assert True

def getOperatorTensorProduct():
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

def test_isProportional():
    assert True

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