import numpy as np
from math import factorial
from copy import copy
from pyscf import ao2mo

from occsfrd.wick import tensor, contractions

def isParticleHoleExcitation(tensor):
    return all(iType == 'p' for iType in tensor.lowerIndexTypes) and all(iType == 'h' for iType in tensor.upperIndexTypes)

def operatorExponential(operator, trunc=4, normalOrdered=False):
    result = tensor.TensorSum([tensor.TensorProduct([])])
    for k in range(trunc):
        term = operator
        for i in range(k):
            term = term * operator
        if normalOrdered:
            if isinstance(term, tensor.TensorProduct):
                term.normalOrderedSlices = [slice(0, len(term.tensorList))]
            elif isinstance(term, tensor.TensorSum):
                for summand in term.summandList:
                    summand.normalOrderedSlices = [slice(0, len(summand.tensorList))]
        result = result + (1 / factorial(k + 1)) * term
    return result

def fullRankDoubleTensorProduct(tensor1, tensor2):
    product = tensor.Tensor(tensor1.name + tensor2.name, tensor1.lowerIndexTypes + tensor2.lowerIndexTypes, tensor1.upperIndexTypes + tensor2.upperIndexTypes)
    permutation = tuple([range(tensor1.excitationRank)] + [range(2 * tensor1.excitationRank, 2 * tensor1.excitationRank + tensor2.excitationRank)] + [range(tensor1.excitationRank, 2 * tensor1.excitationRank)] + [range(2 * tensor1.excitationRank + tensor2.excitationRank, 2 * tensor1.excitationRank + 2 * tensor2.excitationRank)])
    try:
        product.array = np.tensordot(tensor1.array, tensor2.array, axes=0).transpose(permutation)
    except AttributeError:
        pass
    return product

def fullRankTensorProduct(tensorProduct):
    if len(tensorProduct.tensorList) == 0:
        return tensor.Tensor(str(tensorProduct.prefactor), [], [])
    product = tensorProduct.tensorList[0]
    i = 1
    while i < len(tensorProduct.tensorList):
        product = fullRankDoubleTensorProduct(product, tensorProduct.tensorList[i])
        i += 1
    return product

def commutator(operator1, operator2):
    return operator1 * operator2 + (-1) * operator2 * operator1

def BCHSimilarityTransform(H, T, order):
    result = H
    for k in range(order):
        nestedCommutator = H
        for i in range(k + 1):
            nestedCommutator = commutator(nestedCommutator, T)
        result += (1 / factorial(k + 1)) * nestedCommutator
    return result

def deProjectEquation(equation):
#    copiedEquation = copy(equation)
#    for term in copiedEquation.summandList:
    for term in equation.summandList:
        term.tensorList.pop(0)
        projectionVertex = term.vertexList.pop(0)
        term.freeLowerIndices = projectionVertex.upperIndices
        term.freeUpperIndices = projectionVertex.lowerIndices
        term.freeIndexNodes = [tensor.node(lowerIndex, term.freeUpperIndices[lI]) for lI, lowerIndex in enumerate(term.freeLowerIndices)]
#    return copiedEquation

def roundEquation(equation, prec=5):
    roundedEquation = copy(equation)
    for term in roundedEquation.summandList:
        term.prefactor = round(term.prefactor, prec)
    roundedEquation = tensor.TensorSum([term for term in roundedEquation.summandList if term.prefactor != 0])
    return roundedEquation

def cleanEquation(equation, prec=5):
    return tensor.TensorSum([term for term in equation.summandList if abs(term.prefactor) > pow(10, -prec)])

def normSquared(tensor_):
    try:
        # return np.sum(np.square(tensor_.getArray()))
        return np.sum(np.square(tensor_.getArray()))
    except AttributeError:
        print("Norm unavailable for tensor with no array")

def electronicEnergy(mf, referenceOperator=1., correlationTensor=1.):
    h1Tensor = tensor.Tensor("h1", ["g"], ["g"])
    h1Tensor.getAllDiagramsActive()
    h1Tensor.getShapeActive(mf.nelec, mf.mol.nao)
    h1Tensor.setArray(mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff))
    h1Tensor.assignDiagramArraysActive(mf.nelec[1], mf.nelec[0]-mf.nelec[1], mf.mol.nao-mf.nelec[0])

    h2Tensor = tensor.Tensor("h2", ["g", "g"], ["g", "g"])
    h2Tensor.getAllDiagramsActive()
    h2Tensor.getShapeActive(mf.nelec, mf.mol.nao)
    h2Tensor.setArray(ao2mo.kernel(mf.mol, mf.mo_coeff, compact=False).reshape((mf.mol.nao, mf.mol.nao, mf.mol.nao, mf.mol.nao)).swapaxes(2,3).swapaxes(1,2))
    h2Tensor.assignDiagramArraysActive(mf.nelec[1], mf.nelec[0]-mf.nelec[1], mf.mol.nao-mf.nelec[0])

    return contractions.contractTensorSum(contractions.evaluateWick(correlationTensor.conjugate() * (sum(h1Tensor.diagrams) + 0.5 * sum(h2Tensor.diagrams)) * correlationTensor, normalOrderedParts=False, referenceOperator=referenceOperator), lowerIndexList=[], upperIndexList=[])