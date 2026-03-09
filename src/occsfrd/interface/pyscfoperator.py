import pyscf
import math
import numpy as np
from occsfrd.wick import operator, tensor
from copy import copy

from occsfrd import wick

def pyscf_to_vec(pyscfWavefunction, nOrbs, nElec):
    '''
    Take a wavefunction in the pyscf representation and return it as a (Hilbert-space-sized dimension) vector

    Inputs:
    pyscfWavefunction (np.ndarray): wavefunction in the Knowles--Handy representation used by pyscf
    norbs               (int): number of orbitals
    nelec             (tuple): (number of alpha electrons, number of beta electrons)

    Returns:
    (vector): wavefunction as a vector

    Side-effects:
    None
    '''
    nAlpha  = nElec[0]
    nBeta = nElec[1]

    assert pyscfWavefunction.shape == (math.comb(nOrbs, nAlpha), math.comb(nOrbs, nBeta))

    return pyscfWavefunction.flatten()

def vec_to_pyscf(waveVec, nOrbs, nElec):
    '''
    Take a wavefunction as a (Hilbert-space-sized dimension) vector and return it in the pyscf representation

    Inputs:
    waveVec (np.ndarray): wavefunction as a vector
    norbs          (int): number of orbitals
    nelec        (tuple): (number of alpha electrons, number of beta electrons)

    Returns:
    (np.ndarray): wavefunction in the Knowles--Handy representation used by pyscf

    Side-effects:
    None
    '''
    nAlpha  = nElec[0]
    nBeta = nElec[1]

    assert waveVec.shape == (math.comb(nOrbs, nAlpha) * math.comb(nOrbs, nBeta),)

    return waveVec.reshape((math.comb(nOrbs, nAlpha), math.comb(nOrbs, nBeta)))

def apply_OperatorProduct(wavefunction, nOrbs, nElec, operatorProduct, indexValueTuple):
    '''
    Apply a product of creation and annihilation operators to a wavefunction (in the pyscf format)

    Inputs:
    wavefunction                  (np.ndarray): wavefunction in the Knowles--Handy representation used by pyscf
    norbs                                (int): number of orbitals
    nelec                              (tuple): (number of alpha electrons, number of beta electrons)
    operatorProduct (operator.OperatorProduct): operator product being applied
    indexValueTuple                    (tuple): tuple of the values to be assigned to each index of the basic operators

    Returns:
    (np.ndarray): new wavefunction in pyscf's Knowles--Handy representation
    (tuple)     : (number of alpha electrons, number of beta electrons) after operator product applied
    '''
    newWavefunction = copy(wavefunction)
    nAlpha  = nElec[0]
    nBeta = nElec[1]

    for o, operator in enumerate(reversed(operatorProduct.operatorList)):
        indexValue = indexValueTuple[-o]
        if operator.creation_annihilation:
            if operator.spin:
                newWavefunction = pyscf.fci.addons.cre_a(newWavefunction, nOrbs, (nAlpha, nBeta), indexValue)
                nAlpha += 1
            else:
                newWavefunction = pyscf.fci.addons.cre_b(newWavefunction, nOrbs, (nAlpha, nBeta), indexValue)
                nBeta += 1
        else:
            if operator.spin:
                newWavefunction = pyscf.fci.addons.des_a(newWavefunction, nOrbs, (nAlpha, nBeta), indexValue)
                nAlpha -= 1
            else:
                newWavefunction = pyscf.fci.addons.des_b(newWavefunction, nOrbs, (nAlpha, nBeta), indexValue)
                nBeta -= 1

    return operatorProduct.prefactor * newWavefunction, (nAlpha, nBeta)

def apply_OperatorSum(wavefunction, nOrbs, nElec, operatorSum, indexValueTuples):
    '''
    Apply a sum of products of creation and annihilation operators to a wavefunction (in the pyscf format)

    Inputs:
    wavefunction          (np.ndarray): wavefunction in the Knowles--Handy representation used by pyscf
    norbs                        (int): number of orbitals
    nelec                      (tuple): (number of alpha electrons, number of beta electrons)
    operatorSum (operator.OperatorSum): operator sum being applied
    indexValueTuples  (list of tuples): list of tuple of the values to be assigned to each index of the basic operators

    Returns:
    (np.ndarray): new wavefunction in pyscf's Knowles--Handy representation
    or (list)   : in case operator summands do not all leave the same electron numbers, return each separate result in a list
    '''
    resultList = []
#    nElecList = []
    for s, summand in enumerate(operatorSum.summandList):
        partialResult, nElecFinal = apply_OperatorProduct(wavefunction, nOrbs, nElec, summand, indexValueTuples[s])
        resultList.append((partialResult, nElecFinal))
#        nElecList.append(nElecFinal)
    if all(nE == resultList[0,1] for nE in resultList[:,1]):
        return sum(resultList[:,0])
    else:
        return resultList

def nElec_change(operatorProduct):
    '''
    Change in number of (alpha, beta) electrons when applying a second quantised operator product
    '''
    alphaChange = sum((o.creation_annihilation and o.spin) for o in operatorProduct.operatorList) - sum(((not o.creation_annihilation) and o.spin) for o in operatorProduct.operatorList)
    betaChange = sum((o.creation_annihilation and not o.spin) for o in operatorProduct.operatorList) - sum(((not o.creation_annihilation) and not o.spin) for o in operatorProduct.operatorList)
    return (alphaChange, betaChange)

def operator_to_matrix(operator_, nOrbs, nElec):
    '''
    Convert an operator (operator.OperatorSum) to a N x M matrix in a basis of Slater determinants (assumed orthogonal)
    where N is the dimension of the Sz-adapted Hilbert space of bra states
    and M is the dimension of the Sz-adapted Hilbert space of ket states
    
    Inputs:
    operator (operator.OperatorSum): the operator whose matrix representation is sought
    nOrbs                     (int): number of orbitals
    nElec                   (tuple): (number of alpha electrons, number of beta electrons)

    Returns:
    (numpy.ndarray): matrix representation of operator
    '''
    nAlpha  = nElec[0]
    nBeta = nElec[1]
    nElecChange = (0, 0)
    if isinstance(operator_, operator.OperatorSum):
        nElecChange = nElec_change(operator_.summandList[0])
        assert all(nElec_change(s) == nElecChange for s in operator_.summandList)
    elif isinstance(operator_, operator.OperatorProduct):
        nElecChange = nElec_change(operator_)
    dim = (math.comb(nOrbs, nAlpha) * math.comb(nOrbs, nBeta), math.comb(nOrbs, nAlpha + nElecChange[0]) * math.comb(nOrbs, nBeta + nElecChange[1]))
    result = np.zeros(dim)
    for p in range(dim[0]):
        result