import numpy as np
from numbers import Number
from copy import copy
import itertools
import string
from math import factorial
from occsfrd.wick import index, operator, tensor
from occsfrd.wick.contract import contractEXT

def canContract(o1, o2):
    """
    Check if two basic creation/annihilation operators can contract
    
    Parameters
    ----------
    o1: operator.BasicOperator
        Operator to the left in the contraction
    o2: operator.BasicOperator
        Operator to the right in the contraction

    Returns
    -------
    bool
        If operators in specified order can contract
    """
    if o1.quasi_cre_ann:
        return False
    elif not o2.quasi_cre_ann:
        return False
    elif o1.index.occupiedInVacuum != o2.index.occupiedInVacuum:
        return False
    elif o1.index.active != o2.index.active:
        return False
    elif (isinstance(o1.index, index.SpecificOrbitalIndex) and isinstance(o2.index, index.SpecificOrbitalIndex)) and (o1.index.value != o2.index.value):
        return False
    else:
        return o1.spin == o2.spin

#def recursiveFullContraction(operatorProduct_, speedup=False):
def recursiveFullContraction(operatorList_, prefactor, existingContractions, normalOrderedStartPoints, speedup=False):
    """
    Recursively finds the possible complete combinations of contractions of an operator product

    Parameters
    ----------
    operatorList_: list of operator.BasicOperator
        List of the elementary operators being contracted
    prefactor: int
        Prefactor
    existingContractions: list of (operator.BasicOperator, operator.BasicOperator)
        List of contractions already applied (if any)
    normalOrderedStartPoints: list of int
        Positions in operatorList_ at which normal ordered blocks start
    speedup: bool
        Whether to use speedup routines (overall quasiparticle number conservation)

    Returns
    -------
    operator.operatorSum
        Sum of all fully contracted (scalar) terms.
    """
#    operatorList_ = operatorProduct_.operatorList
    if speedup:
        if not sum(o.quasi_cre_ann for o in operatorList_) == sum(not o.quasi_cre_ann for o in operatorList_):
#            return operator.OperatorSum([])
            return 0
        elif not sum(((o.creation_annihilation and o.spin) or (not o.creation_annihilation and not o.spin)) for o in operatorList_) == sum(((not o.creation_annihilation and o.spin) or (o.creation_annihilation and not o.spin)) for o in operatorList_):
            return 0
#    existingContractions = operatorProduct_.contractionsList
    if len(operatorList_) == 0:
#        return operator.OperatorSum([operatorProduct_])
        return operator.OperatorSum([operator.OperatorProduct([], prefactor, existingContractions)])
    elif len(operatorList_) == 2:
        if canContract(operatorList_[0], operatorList_[1]):
            contractionTuple = tuple()
            if operatorList_[0].creation_annihilation:
                contractionTuple = (operatorList_[0].index, operatorList_[1].index)
            else:
                contractionTuple = (operatorList_[1].index, operatorList_[0].index)
            return operator.OperatorSum([operator.OperatorProduct([], prefactor, existingContractions + [contractionTuple])])
        else:
#            return operator.OperatorSum([])
            return 0
    elif len(operatorList_) % 2 == 0:
        i1 = 1
        if len(normalOrderedStartPoints) > 1 and normalOrderedStartPoints[0] == 0:
            i1 = normalOrderedStartPoints[1]
#        elif len(operatorProduct_.normalOrderedStartPoints) == 1:
#            if operatorProduct_.normalOrderedStartPoints[0] != 0:
#                i1 = operatorProduct_.normalOrderedStartPoints[0]
        elif normalOrderedStartPoints == [0]:
#            return operator.OperatorSum([])
            return 0
        result = operator.OperatorSum([])
        for i in range(i1, len(operatorList_) - 1):
            if canContract(operatorList_[0], operatorList_[i]):
                contractionTuple = tuple()
                if operatorList_[0].creation_annihilation:
                    contractionTuple = (operatorList_[0].index, operatorList_[i].index)
                else:
                    contractionTuple = (operatorList_[i].index, operatorList_[0].index)
                result += pow(-1, i-1) * recursiveFullContraction(operatorList_[1:i] + operatorList_[i+1:], prefactor, existingContractions + [contractionTuple], [p - 1 - (p > i) if p > 0 else 0 for p in normalOrderedStartPoints])
        if canContract(operatorList_[0], operatorList_[-1]):
            contractionTuple = tuple()
            if operatorList_[0].creation_annihilation:
                contractionTuple = (operatorList_[0].index, operatorList_[-1].index)
            else:
                contractionTuple = (operatorList_[-1].index, operatorList_[0].index)
            result += recursiveFullContraction(operatorList_[1:-1], prefactor, existingContractions + [contractionTuple], [p - 1 if p > 0 else 0 for p in normalOrderedStartPoints])
        return result
    else:
#        return operator.OperatorSum([])
        return 0
    
def genFortranInterfaceLists(operatorList):
    '''
    For n-operator-long list,
    Generate lists 

    nList (number of operators with which operator n can contract) -- n elements long

    indexList -- n*n: which operator can connect to which other operator
    '''
    n = len(operatorList)
    # indexConnectionsArray = np.zeros((n, n))
    indexConnectionsArray = np.zeros((n, n))
    nList = np.zeros(n)
    for i, iOperator in enumerate(operatorList):
        # iConnectionsList = [i + 1] + [j for j in range(i, n) if canContract(iOperator, operatorList[j])]
        iConnectionsList = [j+1 for j in range(i, n) if canContract(iOperator, operatorList[j])]
        # indexConnectionsArray[i,:len(iConnectionsList)] = np.array(iConnectionsList)
        indexConnectionsArray[:len(iConnectionsList),i] = np.array(iConnectionsList)
        nList[i] = len(iConnectionsList)
    return indexConnectionsArray, nList

def recursiveFullContractionsFortran(operatorProduct):
    if operatorProduct.operatorList == []:
        return operator.OperatorSum([operatorProduct])
    indexConnectionsArray, nList = genFortranInterfaceLists(operatorProduct.operatorList)
    # print(operatorProduct)
    # print("n", len(operatorProduct.operatorList), "nlist", nList, "ncmax", len(operatorProduct.operatorList) / 2, "list", indexConnectionsArray)
    # contractionsFromFortran = contractEXT.contract(n=len(operatorProduct.operatorList), nlist=nList, ncmax=len(operatorProduct.operatorList) / 2, list=indexConnectionsArray)
    n = len(operatorProduct.operatorList)
    # ncmax = int(factorial(n)/(factorial(int(n/2)) * pow(2,int(n/2))))
    ncmax = 100000
    # print(n, ncmax)
    contractionsFromFortran = contractEXT.contract(n=n, nlist=nList, ncmax=ncmax, list=indexConnectionsArray)
    # if contractionsFromFortran[1] and len(operatorProduct.operatorList) == 8:
    #     print(operatorProduct)
    #     print(indexConnectionsArray)
    #     print(contractionsFromFortran[0].T[:contractionsFromFortran[1],:], contractionsFromFortran[2][:contractionsFromFortran[1]])
    #     # problemconnectionsArray = np.zeros_like(indexConnectionsArray)
    #     # problemconnectionsArray[0,:4] = [7,8,5,5]
    #     # problemconnectionsArray[1,:4] = [0,0,6,6]
    #     # problemNList = np.array([1,1,2,2,0,0,0,0])
    #     # problemContractionsFromFortran = contractEXT.contract(n=len(operatorProduct.operatorList), nlist=problemNList, ncmax=2, list=problemconnectionsArray)
    #     # print(problemContractionsFromFortran[0], problemContractionsFromFortran[2])
    #     pass
    # print(contractionsFromFortran)
    contractionsLists = contractionsFromFortran[0][:,:contractionsFromFortran[1]]
    signFlips = contractionsFromFortran[2][:contractionsFromFortran[1]]
    # print(signFlips)
    return genContractionsListsFromFortranInterface(operatorProduct, contractionsLists, signFlips)

def getKroneckerDeltasFromFortranInterfaceContractionsList(operatorList, contractionsListFortran):
    result = []
    for c, contraction in enumerate(contractionsListFortran):
        o0 = operatorList[c]
        o1 = operatorList[contraction-1]
        if contraction == 0:
            pass
        elif o0.creation_annihilation and not o1.creation_annihilation:
            result.append((o0.index, o1.index))
        elif not o0.creation_annihilation and o1.creation_annihilation:
            result.append((o1.index, o0.index))
        else:
            print("invalid contraction")
            return NotImplementedError
    return result

def genContractionsListsFromFortranInterface(operatorProduct, contractionsArray, signFlips):
    '''
    Take list of sets of contractions and form operatorSum object for scalar result
    '''
    # print(contractionsArray)
    # summandList = [operator.OperatorProduct([], operatorProduct.prefactor * pow(-1,signFlips[c]), contractionsList_=getKroneckerDeltasFromFortranInterfaceContractionsList(operatorProduct.operatorList, contractionsList)) for c, contractionsList in enumerate(contractionsArray.T)]
    summandList = [operator.OperatorProduct([], operatorProduct.prefactor * (-1) if signFlips[c] else operatorProduct.prefactor, contractionsList_=getKroneckerDeltasFromFortranInterfaceContractionsList(operatorProduct.operatorList, contractionsList)) for c, contractionsList in enumerate(contractionsArray.T)]
    # summandList = [operator.OperatorProduct([], operatorProduct.prefactor, contractionsList_=getKroneckerDeltasFromFortranInterfaceContractionsList(operatorProduct.operatorList, contractionsList)) for c, contractionsList in enumerate(contractionsArray.T)]
    return operator.OperatorSum(summandList)

def vacuumExpectationValue(operator_, speedup=False, printing=False):
    if isinstance(operator_, operator.OperatorProduct):
#        return recursiveFullContraction(operator_, speedup)
        # return recursiveFullContraction(operator_.operatorList, operator_.prefactor, operator_.contractionsList, operator_.normalOrderedStartPoints, speedup)
        return recursiveFullContractionsFortran(operator_)
    elif isinstance(operator_, operator.OperatorSum):
        result = operator.OperatorSum([])
        for p, product in enumerate(operator_.summandList):
#            term = recursiveFullContraction(product, speedup)
            # term = recursiveFullContraction(product.operatorList, product.prefactor, product.contractionsList, product.normalOrderedStartPoints, speedup)
            term = recursiveFullContractionsFortran(product)
            if printing and term != 0.:
                print(product, term)
            result += term
        return result
    elif isinstance(operator_, Number):
        return operator.OperatorSum([operator.OperatorProduct([], operator_)])
    else:
#        return operator.OperatorSum([])
        return 0

def evaluateWickOld(term, referenceOperator=None, normalOrderedParts=True):
    '''
    Wick's theorem applied to a term

    input: term (TensorProduct)
    output: sum of fully contracted terms (TensorSum)
    '''
    if isinstance(term, tensor.TensorSum):
#        return sum([evaluateWick(summand) for summand in term.summandList])
        result = 0.
        l = len(term.summandList)
        for s, summand in enumerate(term.summandList):
            if s % 1000 == 0:
                print(s, l)
            result = result + (evaluateWickOld(summand, referenceOperator, normalOrderedParts))
        return result
#        return sum([evaluateWick(summand, spinFree) for summand in term.summandList])

    summandList = []
    if referenceOperator is None:
        fullContractions = vacuumExpectationValue(term.getOperator(normalOrderedParts), speedup=True)
    else:
        fullContractions = vacuumExpectationValue(referenceOperator.conjugate() * term.getOperator(normalOrderedParts) * referenceOperator, speedup=True)
#    fullContractions = vacuumExpectationValue(term.getOperator(normalOrderedParts), speedup=True)
    for topology in fullContractions.summandList:
        contractionsList = topology.contractionsList
        contractedTerm = copy(term)
        contractedTerm.prefactor = topology.prefactor
        for c, contraction in enumerate(contractionsList):
            contractedTerm.applyContraction(contraction)
            # for v, vertex in reversed(list(enumerate(term.vertexList))):
            #     if contraction[0] in vertex.lowerIndices:
            #         contractedTerm.vertexList[v].lowerIndices[vertex.lowerIndices.index(contraction[0])] = contraction[1]
            #         break
            #     elif contraction[1] in vertex.upperIndices:
            #         contractedTerm.vertexList[v].upperIndices[vertex.upperIndices.index(contraction[1])] = contraction[0]
            #         break
        summandList.append(contractedTerm)
    return tensor.TensorSum(summandList)

def evaluateWick(term, referenceOperator=None, normalOrderedParts=True):
    '''
    Wick's theorem applied to a term

    input: term (TensorProduct)
    output: sum of fully contracted terms (TensorSum)
    '''
    if isinstance(term, tensor.TensorSum):
#        return sum([evaluateWick(summand) for summand in term.summandList])
        result = 0.
        l = len(term.summandList)
        for s, summand in enumerate(term.summandList):
            if s == 66:
                pass
            if s % 1000 == 0:
                print(s, l)
            result = result + (evaluateWick(summand, referenceOperator, normalOrderedParts))
        return result
#        return sum([evaluateWick(summand, spinFree) for summand in term.summandList])

    summandList = []
    if referenceOperator is None:
        fullContractions = vacuumExpectationValue(term.getOperator(normalOrderedParts), speedup=True)
    else:
        fullContractions = vacuumExpectationValue(referenceOperator.conjugate() * term.getOperator(normalOrderedParts) * referenceOperator, speedup=True)
#    fullContractions = vacuumExpectationValue(term.getOperator(normalOrderedParts), speedup=True)
    if isinstance(fullContractions, Number):
        return fullContractions
    for t, topology in enumerate(fullContractions.summandList):
        contractionsList = topology.contractionsList
        contractedTerm = copy(term)
        contractedTerm.prefactor = topology.prefactor
#        contractedTerm.contractionsList = contractionsList
        for c, contraction in enumerate(contractionsList):
            # if contraction[0] in contractedTerm.freeLowerIndices:
            #     contractedTerm.freeLowerIndices.remove(contraction[0])
            # if contraction[1] in contractedTerm.freeUpperIndices:
            #     contractedTerm.freeUpperIndices.remove(contraction[1])
            contractedTerm.applyContraction(contraction)
            # for v, vertex in reversed(list(enumerate(term.vertexList))):
            #     if contraction[0] in vertex.lowerIndices:
            #         contractedTerm.vertexList[v].lowerIndices[vertex.lowerIndices.index(contraction[0])] = contraction[1]
            #         break
            #     elif contraction[1] in vertex.upperIndices:
            #         contractedTerm.vertexList[v].upperIndices[vertex.upperIndices.index(contraction[1])] = contraction[0]
            #         break
        summandList.append(contractedTerm)
    return tensor.TensorSum(summandList)

def chooseUncontractedOperatorPositions(operatorProduct_, freeIndexTypes):
    operatorList_ = operatorProduct_.operatorList
    lowerIndexTypes, upperIndexTypes = freeIndexTypes[0], freeIndexTypes[1]
    possiblechoices = itertools.combinations([*range(len(operatorList_))], len(lowerIndexTypes) + len(upperIndexTypes))
    for possiblechoice in possiblechoices:
        if sum([operatorList_[o].creation_annihilation for o in possiblechoice]) == len(lowerIndexTypes):
            if sum([operatorList_[o].creation_annihilation and operatorList_[o].quasi_cre_ann for o in possiblechoice]) == sum([lowerIndexType == "p" for lowerIndexType in lowerIndexTypes]) and sum([operatorList_[o].creation_annihilation and not operatorList_[o].quasi_cre_ann for o in possiblechoice]) == sum([lowerIndexType == "c" for lowerIndexType in lowerIndexTypes]):
                if sum([not operatorList_[o].creation_annihilation and not operatorList_[o].quasi_cre_ann for o in possiblechoice]) == sum([upperIndexType == "p" for upperIndexType in upperIndexTypes]) and sum([not operatorList_[o].creation_annihilation and operatorList_[o].quasi_cre_ann for o in possiblechoice]) == sum([upperIndexType == "c" for upperIndexType in upperIndexTypes]):
                    yield possiblechoice

def recursiveIncompleteContractionNew(operator_, freeIndexTypes=([], []), speedup=False):
    if isinstance(operator_, operator.OperatorSum):
        return sum([recursiveIncompleteContractionNew(summand, freeIndexTypes, speedup)for summand in operator_.summandList])
    operatorList_ = operator_.operatorList
    total = operator.OperatorSum([])
    for choice in chooseUncontractedOperatorPositions(operator_, freeIndexTypes):
        parity = sum([p for p in choice]) % 2
        newOperatorList = [operatorList_[p] for p in choice]
        contractedOperatorList = [operator_ for o, operator_ in enumerate(operatorList_) if o not in choice]
        contractions = vacuumExpectationValue(operator.OperatorProduct(contractedOperatorList), speedup)
        for contraction in contractions.summandList:
            contraction.operatorList = newOperatorList
        contractions *= pow(-1, parity)
        total += contractions
    return total

def evaluateWickFree(term, freeIndexTypes=([], []), speedup=False, normalOrderedParts=True):
    '''
    Wick's theorem applied to a term

    input: term (TensorProduct)
    output: sum of partially contracted terms (TensorSum)
    '''
    if isinstance(term, tensor.TensorSum):
        return sum([evaluateWickFree(summand, freeIndexTypes, speedup) for summand in term.summandList])
    summandList = []
    if freeIndexTypes == ([], []):
        contractions = vacuumExpectationValue(term.getOperator(normalOrderedParts), speedup)
    else:
        contractions = recursiveIncompleteContractionNew(term.getOperator(normalOrderedParts), freeIndexTypes, speedup)
    for topology in contractions.summandList:
        contractionsList = topology.contractionsList
        contractedTerm = copy(term)
        contractedTerm.prefactor = topology.prefactor
        contractedTerm.prefactor /= pow(2, len(freeIndexTypes[0]))
        for c, contraction in enumerate(contractionsList):
            for v, vertex in reversed(list(enumerate(term.vertexList))):
                if contraction[0] in vertex.lowerIndices:
                    contractedTerm.vertexList[v].lowerIndices[vertex.lowerIndices.index(contraction[0])] = contraction[1]
                    break
                elif contraction[1] in vertex.upperIndices:
                    contractedTerm.vertexList[v].upperIndices[vertex.upperIndices.index(contraction[1])] = contraction[0]
                    break
        summandList.append(contractedTerm)
    return tensor.TensorSum(summandList)
#    return TensorSum(summandList).collectIsomorphicTerms()

def getAxis(vertex, index):
    for a in range(vertex.excitationRank):
        if vertex.lowerIndices[a] == index:
            return a
        elif vertex.upperIndices[a] == index:
            return vertex.excitationRank + a

def getContractedArrayOld(tensorProduct_, targetLowerIndexList=None, targetUpperIndexList=None):
    lowerIndexList = list(itertools.chain.from_iterable([vertex.lowerIndices for vertex in tensorProduct_.vertexList]))
    upperIndexList = list(itertools.chain.from_iterable([vertex.upperIndices for vertex in tensorProduct_.vertexList]))
    lowerIndexLetters = string.ascii_lowercase[:len(lowerIndexList)]
    upperIndexLetters = ""
    freeLowerIndexMask = np.ones(len(lowerIndexList))
    freeUpperIndexMask = np.ones(len(upperIndexList))
    nFreeUpperIndices = 0
    for uI, upperIndex in enumerate(upperIndexList):
        free = True
        if not isinstance(upperIndex, index.SpecificOrbitalIndex):
            for lI, lowerIndex in enumerate(lowerIndexList):
                if upperIndex == lowerIndex:
                    upperIndexLetters += lowerIndexLetters[lI]
                    freeLowerIndexMask[lI] = 0
                    freeUpperIndexMask[uI] = 0
                    free = False
        if free:
            upperIndexLetters += string.ascii_lowercase[len(lowerIndexList) + nFreeUpperIndices]
            nFreeUpperIndices += 1
    freeLowerIndexList = [lowerIndex for lI, lowerIndex in enumerate(lowerIndexList) if freeLowerIndexMask[lI]]
    freeUpperIndexList = [upperIndex for uI, upperIndex in enumerate(upperIndexList) if freeUpperIndexMask[uI]]
    summandZero = False

    lowerIndexListContractedFrom = [lowerIndex if not isinstance(lowerIndex, index.SpecificOrbitalIndex) else lowerIndex.contractedFrom for lowerIndex in lowerIndexList]
    upperIndexListContractedFrom = [upperIndex if not isinstance(upperIndex, index.SpecificOrbitalIndex) else upperIndex.contractedFrom for upperIndex in upperIndexList]
    if targetLowerIndexList == None and targetUpperIndexList == None:
#        targetLowerIndexList = freeLowerIndexList
        targetLowerIndexList = [lowerIndex for lowerIndex in freeLowerIndexList if not isinstance(lowerIndex, index.SpecificOrbitalIndex)]
#        targetLowerIndexList = [lowerIndex if not isinstance(lowerIndex, index.SpecificOrbitalIndex) else lowerIndex.contractedFrom for lowerIndex in freeLowerIndexList]
#        targetUpperIndexList = freeUpperIndexList
        targetUpperIndexList = [upperIndex for upperIndex in freeUpperIndexList if not isinstance(upperIndex, index.SpecificOrbitalIndex)]
#        targetUpperIndexList = [upperIndex if not isinstance(upperIndex, index.SpecificOrbitalIndex) else upperIndex.contractedFrom for upperIndex in freeUpperIndexList]
        summandZero = True
#    print("tl", *targetLowerIndexList)
#    print("l", *lowerIndexList)
#    print("tu", *targetUpperIndexList)
#    print("u", *upperIndexList)
#    freeLowerIndexLetters = "".join([lowerIndexLetters[lowerIndexList.index(lowerIndex)] for lowerIndex in freeLowerIndexList])
#    freeUpperIndexLetters = "".join([upperIndexLetters[upperIndexList.index(upperIndex)] for upperIndex in freeUpperIndexList])
#    targetLowerIndexLetters = "".join([lowerIndexLetters[lowerIndexList.index(lowerIndex)] for lowerIndex in targetLowerIndexList])
#    targetUpperIndexLetters = "".join([upperIndexLetters[upperIndexList.index(upperIndex)] for upperIndex in targetUpperIndexList])
#    freeLowerIndexLetters = "".join([(lowerIndexLetters[lowerIndexList.index(lowerIndex.contractedFrom)] if isinstance(lowerIndex, specificOrbitalIndex) else lowerIndexLetters[lowerIndexList.index(lowerIndex)]) for lowerIndex in targetLowerIndexList])
#    freeUpperIndexLetters = "".join([(upperIndexLetters[upperIndexList.index(upperIndex.contractedFrom)] if isinstance(upperIndex, specificOrbitalIndex) else upperIndexLetters[upperIndexList.index(upperIndex)]) for upperIndex in targetUpperIndexList])
#    freeLowerIndexLettersNoSpecific = "".join([lowerIndexLetters[lowerIndexList.index(lowerIndex)] for lowerIndex in freeLowerIndexList if not isinstance(lowerIndex, specificOrbitalIndex)])
#    freeUpperIndexLettersNoSpecific = "".join([upperIndexLetters[upperIndexList.index(upperIndex)] for upperIndex in freeUpperIndexList if not isinstance(upperIndex, specificOrbitalIndex)])
    freeLowerIndexLetters = ""
    freeLowerIndexLettersNoSpecific = ""
    targetLowerIndexLetters = ""
    for lowerIndex in freeLowerIndexList:
        freeLowerIndexLetters += lowerIndexLetters[lowerIndexList.index(lowerIndex)]
        if not isinstance(lowerIndex, index.SpecificOrbitalIndex):
            freeLowerIndexLettersNoSpecific += lowerIndexLetters[lowerIndexList.index(lowerIndex)]
    for lowerIndex in targetLowerIndexList:
#        targetLowerIndexLetters += lowerIndexLetters[lowerIndexListContractedFrom.index(lowerIndex)]
        targetLowerIndexLetters += lowerIndexLetters[lowerIndexList.index(lowerIndex)]
        # if isinstance(lowerIndex, specificOrbitalIndex):
        #     try:
        #         targetLowerIndexLetters += lowerIndexLetters[lowerIndexList.index(lowerIndex.contractedFrom)]
        #     except ValueError:
        #         pass
        # else:
        #     targetLowerIndexLetters += lowerIndexLetters[lowerIndexList.index(lowerIndex)]
        # else:
        #     freeLowerIndexLetters += lowerIndexLetters[lowerIndexList.index(lowerIndex)]
        #     freeLowerIndexLetters += lowerIndexLetters[np.where(lowerIndexList == lowerIndex)]
    freeUpperIndexLetters = ""
    freeUpperIndexLettersNoSpecific = ""
    targetUpperIndexLetters = ""
    for upperIndex in freeUpperIndexList:
        freeUpperIndexLetters += upperIndexLetters[upperIndexList.index(upperIndex)]
        if not isinstance(upperIndex, index.SpecificOrbitalIndex):
            freeUpperIndexLettersNoSpecific += upperIndexLetters[upperIndexList.index(upperIndex)]
    for upperIndex in targetUpperIndexList:
#        targetUpperIndexLetters += upperIndexLetters[upperIndexListContractedFrom.index(upperIndex)]
        targetUpperIndexLetters += upperIndexLetters[upperIndexList.index(upperIndex)]
        # if isinstance(upperIndex, specificOrbitalIndex):
        #     try:
        #         targetUpperIndexLetters += upperIndexLetters[upperIndexList.index(upperIndex.contractedFrom)]
        #     except ValueError:
        #         pass
        # else:
        #     targetUpperIndexLetters += upperIndexLetters[upperIndexList.index(upperIndex)]
        # else:
        #     freeUpperIndexLetters += upperIndexLetters[upperIndexList.index(upperIndex)]
        #     freeUpperIndexLetters += upperIndexLetters[np.where(upperIndexList == upperIndex)]
#.join([(lowerIndexLetters[]  else try lowerIndexLetters[lowerIndexList.index(lowerIndex)]) except ValueError pass ])
#    freeUpperIndexLetters = ""
# .join([(upperIndexLetters[upperIndexList.index(upperIndex.contractedFrom)] if isinstance(upperIndex, specificOrbitalIndex) else try upperIndexLetters[upperIndexList.index(upperIndex)]) except ValueError pass for upperIndex in targetUpperIndexList])
    einsumSubstrings = []
    start = 0
    for vertex in tensorProduct_.vertexList:
        end = start + vertex.excitationRank
        einsumSubstring = lowerIndexLetters[start:end] + upperIndexLetters[start:end]
        einsumSubstrings.append(einsumSubstring)
        start = end
    einsumString = ",".join(einsumSubstrings)
    einsumString += '->' + freeLowerIndexLetters + freeUpperIndexLetters
#    einsumString += '->' + targetLowerIndexLetters + targetUpperIndexLetters
    contribution = tensorProduct_.prefactor * np.einsum(einsumString, *[vertex.tensor.getArray() for vertex in tensorProduct_.vertexList])
    slicedContribution = sliceActiveIndices(contribution, freeLowerIndexList, freeUpperIndexList)
    if summandZero:
#        return slicedContribution, freeLowerIndexList, freeUpperIndexList
        return slicedContribution, targetLowerIndexList, targetUpperIndexList
    return slicedContribution
    # transposedContribution = np.einsum(freeLowerIndexLettersNoSpecific + freeUpperIndexLettersNoSpecific + '->' + targetLowerIndexLetters + targetUpperIndexLetters, slicedContribution)
    # if summandZero:
    #     return transposedContribution, freeLowerIndexList, freeUpperIndexList
    # return transposedContribution

def sliceActiveIndices(array, lowerIndexList, upperIndexList):
    if isinstance(array, Number):
        return array
    slices = [slice(None)] * len(array.shape)
    for lI, lowerIndex in enumerate(lowerIndexList):
#        print(lI, lowerIndex)
        if isinstance(lowerIndex, index.SpecificOrbitalIndex):
#            print(lowerIndex.value)
            if isinstance(lowerIndex.value, int):
                slices[lI] = lowerIndex.value
            else:
                slices[lI] = 0
                print("No value found for lower index")
    for uI, upperIndex in enumerate(upperIndexList):
#        print(uI, upperIndex)
        if isinstance(upperIndex, index.SpecificOrbitalIndex):
#            print(upperIndex.value)
            if isinstance(upperIndex.value, int):
                slices[uI + len(lowerIndexList)] = upperIndex.value
            else:
                slices[uI + len(lowerIndexList)] = 0
                print("No value found for upper index")
#    print(slices)
#    newArray = np.zeros_like(array)
#    newArray[tuple(slices)] = array[tuple(slices)]
#    return newArray[tuple(slices)]
#    return array[tuple(slices)]
    return copy(array[tuple(slices)])

def getContractedArrayOldTest(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None):
    '''
    Take a tensor product (uncontracted) and a list of contractions (as pairs of indices)
    and return the array corresponding to the contracted tensor with the target indices as specified

    Args:
    tensorProduct (tensor.TensorProduct): the uncontracted tensor product being contracted
    contractionsList (list) of (tuple): list of pairs of index.Index objects corresponding to contracted indices
    prefactor (float): prefactor if applicable
    targetLowerIndices (list) of (index.Index): target lower indices of resultant tensor
    targetUpperIndices (list) of (index.Index): target upper indices of resultant tensor

    Structure:
    identify each index with a letter (character) to biuld einsum strings: use dictionary?

    get einsum string for 
    '''
    if targetLowerIndices is None:
        targetLowerIndices = tensorProduct.freeLowerIndices
    if targetUpperIndices is None:
        targetUpperIndices = tensorProduct.freeUpperIndices
    lowerIndexList = list(itertools.chain.from_iterable([vertex.lowerIndices for vertex in tensorProduct.vertexList]))
    upperIndexList = list(itertools.chain.from_iterable([vertex.upperIndices for vertex in tensorProduct.vertexList]))
    lowerIndexLetters = string.ascii_lowercase[:len(lowerIndexList)]
    upperIndexLetters = string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)]
#    upperIndexLettersList = []
    lowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    upperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    newLowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    newUpperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    targetLowerIndexLettersList = []
    targetUpperIndexLettersList = []

    # indexLettersDictionary = {}
    # for i, ind in enumerate(lowerIndexList+upperIndexList):
    #     indexLettersDictionary[ind] = string.ascii_lowercase[i]

    contractionsList = tensorProduct.contractionsList + contractionsList_

    # for c, contraction in enumerate(contractionsList):
    #     if not contraction[0] in indexLettersDictionary.keys():
    #         indexLettersDictionary[contraction[0]] = string.ascii_lowercase[len(indexLettersDictionary)]
    #     if not contraction[1] in indexLettersDictionary.keys():
    #         indexLettersDictionary[contraction[1]] = string.ascii_lowercase[len(indexLettersDictionary)]
    #     lowerIndexList.append(contraction[1])
    #     upperIndexList.append(contraction[0])

    # targetLowerIndices, targetUpperIndices = targetLowerIndices_, targetUpperIndices_
    # if targetLowerIndices is None:
    #     targetLowerIndices = [lowerIndex for lI, lowerIndex in enumerate(lowerIndexList) if lowerIndex not in upperIndexList]
    # if targetUpperIndices is None:
    #     targetUpperIndices = [upperIndex for uI, upperIndex in enumerate(upperIndexList) if upperIndex not in lowerIndexList]

    specificContractionIndicesList = []
#     nonSpecificContractionsList = []
    contractedList = [False] * len(contractionsList)
    for c, contraction in enumerate(contractionsList):
        if not isinstance(contraction[0], index.SpecificOrbitalIndex) and not isinstance(contraction[1], index.SpecificOrbitalIndex):
#        contracted = False
            for lI, lowerIndex in enumerate(lowerIndexList):
                if contraction[0] == lowerIndex:
                    contractedList[c] = True
                    try:
                        newUpperIndexLettersList[upperIndexList.index(contraction[1])] = lowerIndexLettersList[lI]
                    except ValueError:
                        pass
                        # print("contraction", *contraction)
                        # print("upper Index List", *upperIndexList)
            if not contractedList[c]:
                for uI, upperIndex in enumerate(upperIndexList):
                    if contraction[1] == upperIndex:
                        contractedList[c] = True
                        try:
                            newLowerIndexLettersList[lowerIndexList.index(contraction[0])] = upperIndexLettersList[uI]
                        except ValueError:
                            pass
                            # print("contraction", *contraction)
                            # print("lower Index List", *lowerIndexList)
        elif isinstance(contraction[0], index.SpecificOrbitalIndex) and isinstance(contraction[1], index.SpecificOrbitalIndex):
            specificContractionIndicesList.append(c)
    if tensorProduct.tensorList == [] and len(specificContractionIndicesList) == len(contractionsList):
        result = tensorProduct.prefactor * prefactor * int(np.all([contraction[0].value == contraction[1].value for c, contraction in enumerate (contractionsList)]))
        return result, [], []
    if targetLowerIndices is None:
        targetLowerIndexLettersList = [lIL for lIL in lowerIndexLettersList if lIL not in newUpperIndexLettersList]
    else:
#         print("tarL", *targetLowerIndices)
        targetLowerIndexLettersList = []
        for targetLowerIndex in targetLowerIndices:
            traced = False
            tLI = targetLowerIndex
            while not traced:
                if tLI in lowerIndexList:
                    targetLowerIndexLettersList.append(lowerIndexLettersList[lowerIndexList.index(tLI)])
                    traced = True
                else:
                    for contraction in contractionsList:
                        if contraction[1] == tLI:
                            # print("trace", *contraction)
                            tLI = contraction[0]
    if targetUpperIndices is None:
        targetUpperIndexLettersList = [uIL for uIL in newUpperIndexLettersList if uIL not in lowerIndexLettersList]
    else:
        # print("tarU", *targetUpperIndices)
        targetUpperIndexLettersList = []
        for targetUpperIndex in targetUpperIndices:
            traced = False
            tUI = targetUpperIndex
            while not traced:
                if tUI in upperIndexList:
                    targetUpperIndexLettersList.append(upperIndexLettersList[upperIndexList.index(tUI)])
                    traced = True
                else:
                    for contraction in contractionsList:
                        if contraction[0] == tUI:
                            tUI = contraction[1]
    lowerIndexLetters = "".join(newLowerIndexLettersList)
    upperIndexLetters = "".join(newUpperIndexLettersList)
    resultLowerIndexLetters = "".join(targetLowerIndexLettersList)
    resultUpperIndexLetters = "".join(targetUpperIndexLettersList)
#    upperIndexLetters = ""
#    resultLowerIndexLetters = ""
#    resultUpperIndexLetters = ""
#     for c, contraction in enumerate(contractionsList):
#         if contraction[0] in targetUpperIndices:
#             targetUpperIndices[targetUpperIndices.index(contraction[0])] = contraction[1]
#             contractionsList.pop(c)
#         if contraction[1] in targetLowerIndices:
#             targetLowerIndices[targetLowerIndices.index(contraction[1])] = contraction[0]
#             contractionsList.pop(c)
#     for uI, upperIndex in enumerate(upperIndexList):
#         contracted = False
#         for c, contraction in enumerate(contractionsList):
#             if not isinstance(contraction[0], index.SpecificOrbitalIndex) and contraction[0] not in targetUpperIndices:
#                 if not isinstance(contraction[1], index.SpecificOrbitalIndex) and contraction[1] not in targetLowerIndices:
#                     if contraction[1] == upperIndex:
#                         upperIndexLetters += lowerIndexLetters[lowerIndexList.index(contraction[0])]
#                         contracted = True
#         if not contracted:
#             upperIndexLetters += string.ascii_lowercase[len(lowerIndexList)+len(upperIndexLetters)]
# #            resultUpperIndexLetters += string.ascii_lowercase[len(lowerIndexList)+len(upperIndexLetters)]
#             resultUpperIndexLetters += upperIndexLetters[-1]
#     for lI, lowerIndex in enumerate(lowerIndexList):
#         contracted = False
#         for c, contraction in enumerate(contractionsList):
#             if not isinstance(contraction[0], index.SpecificOrbitalIndex) and contraction[0] not in targetUpperIndices:
#                 if not isinstance(contraction[1], index.SpecificOrbitalIndex) and contraction[1] not in targetLowerIndices:
#                     if contraction[0] == lowerIndex:
#                         contracted = True
#         if not contracted:
#             resultLowerIndexLetters += lowerIndexLetters[lI]
# #            resultLowerIndexLetters += lowerIndexLetters[lowerIndexList.index(contraction[0])]
# #                upperIndexLetters[]
#     if targetLowerIndices is not None:
#         resultLowerIndexLetters = "".join([lowerIndexLetters[lowerIndexList.index(lowerIndex)] for lI, lowerIndex in enumerate(targetLowerIndices)])
#     if targetUpperIndices is not None:
#         resultUpperIndexLetters = "".join([upperIndexLetters[upperIndexList.index(upperIndex)] for uI, upperIndex in enumerate(targetUpperIndices)])
    einsumSubstrings = []
    vertexSlices = []
    start = 0
    for vertex in tensorProduct.vertexList:
        end = start + vertex.excitationRank
        einsumSubstring = lowerIndexLetters[start:end] + upperIndexLetters[start:end]
        einsumSubstrings.append(einsumSubstring)
        start = end
        vertexSlice = []
        for lI, lowerIndex in enumerate(vertex.lowerIndices):
            followedIndex, specificValues = followLowerIndexThroughContractionsOld(lowerIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        for uI, upperIndex in enumerate(vertex.upperIndices):
            followedIndex, specificValues = followUpperIndexThroughContractionsOld(upperIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        vertexSlices.append(tuple(vertexSlice))
    # for vertex in tensorProduct.vertexList:
    #     einsumSubstrings.append("".join([indexLettersDictionary[ind] for ind in vertex.lowerIndices]) + "".join([indexLettersDictionary[followUpperIndexThroughContractions(ind, contractionsList)] for ind in vertex.upperIndices]))
    einsumString = ",".join(einsumSubstrings)
    einsumString += "->"
    # einsumString += "".join([indexLettersDictionary[ind] for ind in targetLowerIndices]) + "".join([indexLettersDictionary[ind] for ind in targetUpperIndices])
    einsumString += resultLowerIndexLetters
    einsumString += resultUpperIndexLetters
    #print(einsumString)
    #print(*tensorProduct.vertexList)
    #print(vertexSlices)
    #print(*[vertex.tensor.array.shape for v, vertex in enumerate(tensorProduct.vertexList)])
    result = prefactor * tensorProduct.prefactor * np.einsum(einsumString, *[vertex.tensor.getArray()[vertexSlices[v]] for v, vertex in enumerate(tensorProduct.vertexList)], optimize='optimal') # optimised einsum ordering 16/01/2023
#    slicedResult = np.zeros_like(result)
#     nonzeroSlices = [slice(None)] * len(result.shape)
#     for c, contraction in enumerate(contractionsList):
#         if isinstance(contraction[1], index.SpecificOrbitalIndex):
#             if isinstance(contraction[0], index.SpecificOrbitalIndex):
#                 if contraction[0].value != contraction[1].value:
#                     return slicedResult
#             else:
#                 if contraction[0] in lowerIndexList:
#                     axis = resultLowerIndexLetters.index(lowerIndexLetters[lowerIndexList.index(contraction[0])])
# #                if contraction[0] in targetLowerIndices:
# #                    axis = targetLowerIndices.index(contraction[0])
#                     nonzeroSlices[axis] = slice(contraction[1].value, contraction[1].value + 1)
#         elif isinstance(contraction[0], index.SpecificOrbitalIndex):
#             if contraction[1] in upperIndexList:
#                 axis = len(resultLowerIndexLetters) + resultUpperIndexLetters.index(upperIndexLetters[upperIndexList.index(contraction[1])])
# #            if contraction[1] in targetUpperIndices:
# #                axis = len(targetLowerIndices) + targetUpperIndices.index(contraction[1])
#                 nonzeroSlices[axis] = slice(contraction[0].value, contraction[0].value + 1)
#     slicedResult[tuple(nonzeroSlices)] = result[tuple(nonzeroSlices)]
    if targetLowerIndices is None and targetUpperIndices is None:
        return result, [followLowerIndexThroughContractionsOld(lowerIndexList[lowerIndexLettersList.index(lowerIndexLetter)], contractionsList)[0] for lIL, lowerIndexLetter in enumerate(resultLowerIndexLetters)], [followUpperIndexThroughContractionsOld(upperIndexList[upperIndexLettersList.index(upperIndexLetter)], contractionsList)[0] for uIL, upperIndexLetter in enumerate(resultUpperIndexLetters)]
#        return slicedResult, [lowerIndexList[lowerIndexLettersList.index(lowerIndexLetter)] for lIL, lowerIndexLetter in enumerate(resultLowerIndexLetters)], [upperIndexList[upperIndexLettersList.index(upperIndexLetter)] for uIL, upperIndexLetter in enumerate(resultUpperIndexLetters)]
#        return slicedResult, targetLowerIndices, targetUpperIndices
#    return slicedResult
    return result

def followUpperIndexThroughContractionsOld(upperIndex, contractionsList):
    '''
    Take an upper index and follow it through a list of contractions
    '''
    currentIndex = upperIndex
    specificValues = []
    while True:
        if isinstance(currentIndex, index.SpecificOrbitalIndex):
            specificValues.append(currentIndex.value)
        found = False
        for c, contraction in enumerate(contractionsList):
            if contraction[1] == currentIndex:
                found = True
                currentIndex = contraction[0]
        if not found:
            break
    return currentIndex, specificValues

def followLowerIndexThroughContractionsOld(lowerIndex, contractionsList):
    '''
    Take a lower index and follow it through a list of contractions
    '''
    currentIndex = lowerIndex
    specificValues = []
    while True:
        if isinstance(currentIndex, index.SpecificOrbitalIndex):
            specificValues.append(currentIndex.value)
        found = False
        for c, contraction in enumerate(contractionsList):
            if contraction[0] == currentIndex:
                found = True
                currentIndex = contraction[1]
            if isinstance(currentIndex, index.SpecificOrbitalIndex):
                break
        if not found:
            break
    return currentIndex, specificValues

def testEqualTermsInTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    for t, term in enumerate(tensorSum_.summandList):
        oldResult = getContractedArrayOldTest(term, targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList)
        newResult = getContractedArray(term, targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList, resultShape=resultShape)
        if not np.all(oldResult == newResult):
            print(t, term)

def testEqualTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    oldResult = testOldContractTensorSum(tensorSum_, lowerIndexList=lowerIndexList, upperIndexList=upperIndexList)
    newResult = contractTensorSum(tensorSum_, lowerIndexList=lowerIndexList, upperIndexList=upperIndexList, resultShape=resultShape)
    print(np.all(oldResult == newResult))

def testOldContractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None):
    if len(tensorSum_.summandList) == 0:
        return 0
    contractedArray = getContractedArrayOldTest(tensorSum_.summandList[0], targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList)
    i = 1
    while i < len(tensorSum_.summandList):
        contractedArray += getContractedArrayOldTest(tensorSum_.summandList[i], targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList)
        i += 1
    return contractedArray

def contractTensorSum(tensorSum_, lowerIndexList=None, upperIndexList=None, resultShape=None):
    if len(tensorSum_.summandList) == 0:
        return 0
    # if lowerIndexList is not None and upperIndexList is not None:
    #     contractedArray = getContractedArray(tensorSum_.summandList[0], targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList)
    # else:
    #     contractedArray, lowerIndexList, upperIndexList = getContractedArray(tensorSum_.summandList[0])
    contractedArray = getContractedArray(tensorSum_.summandList[0], targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList, resultShape=resultShape)
    # if lowerIndexList is None:
    #     lowerIndexList = tensorSum_.summandList[0].freeLowerIndices
    # if upperIndexList is None:
    #     upperIndexList = tensorSum_.summandList[0].freeUpperIndices
    i = 1
    while i < len(tensorSum_.summandList):
        contractedArray += getContractedArray(tensorSum_.summandList[i], targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList, resultShape=resultShape)
        # if i == 1:
        #     print(tensorSum_.summandList[i])
        i += 1
    return contractedArray
    #return sliceActiveIndices(contractedArray, lowerIndexList, upperIndexList)

# def contractTensorSumNew(tensorSum_, lowerIndexList=None, upperIndexList=None):
#     if len(tensorSum_.summandList) == 0:
#         return 0
#     if lowerIndexList is not None and upperIndexList is not None:
#         contractedArray = getContractedArrayNew(tensorSum_.summandList[0], targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList)
#     else:
#         contractedArray = getContractedArrayNew(tensorSum_.summandList[0])
#     i = 1
#     while i < len(tensorSum_.summandList):
#         contractedArray += getContractedArrayNew(tensorSum_.summandList[i], targetLowerIndices=lowerIndexList, targetUpperIndices=upperIndexList)
#         i += 1
#     return contractedArray

def getContractedArraySlow(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    '''
    Take a tensor product (uncontracted) and a list of contractions (as pairs of indices)
    and return the array corresponding to the contracted tensor with the target indices as specified

    Args:
    tensorProduct (tensor.TensorProduct): the uncontracted tensor product being contracted
    contractionsList (list) of (tuple): list of pairs of index.Index objects corresponding to contracted indices
    prefactor (float): prefactor if applicable
    targetLowerIndices (list) of (index.Index): target lower indices of resultant tensor
    targetUpperIndices (list) of (index.Index): target upper indices of resultant tensor

    Structure:
    identify each index with a letter (character) to build einsum strings: use dictionary?

    get einsum string for 
    '''
    if targetLowerIndices is None:
        targetLowerIndices = tensorProduct.freeLowerIndices
    if targetUpperIndices is None:
        targetUpperIndices = tensorProduct.freeUpperIndices
    lowerIndexList = list(itertools.chain.from_iterable([vertex.lowerIndices for vertex in tensorProduct.vertexList]))
    upperIndexList = list(itertools.chain.from_iterable([vertex.upperIndices for vertex in tensorProduct.vertexList]))
    lowerIndexLetters = string.ascii_lowercase[:len(lowerIndexList)]
    upperIndexLetters = string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)]
#    upperIndexLettersList = []
    lowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    upperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    newLowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    newUpperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    targetLowerIndexLettersList = []
    targetUpperIndexLettersList = []

    # indexLettersDictionary = {}
    # for i, ind in enumerate(lowerIndexList+upperIndexList):
    #     indexLettersDictionary[ind] = string.ascii_lowercase[i]

    contractionsList = tensorProduct.contractionsList + contractionsList_
    contractionsLowerIndices = []
    contractionsUpperIndices = []
    contractionLowerIndicesLettersList = []
    contractionUpperIndicesLettersList = []
    nNewIndices = 0
    for c, contraction in enumerate(contractionsList):
        contractionsLowerIndices.append(contraction[1])
        contractionsUpperIndices.append(contraction[0])
        try:
            contractionUpperIndicesLettersList.append(lowerIndexLetters[lowerIndexList.index(contraction[0])])
        except ValueError:
            try:
                contractionUpperIndicesLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(contraction[0])])
            except (ValueError, IndexError):
                contractionUpperIndicesLettersList.append(string.ascii_lowercase[len(lowerIndexList) + len(upperIndexList) + nNewIndices])
                nNewIndices += 1
        try:
            contractionLowerIndicesLettersList.append(upperIndexLetters[upperIndexList.index(contraction[1])])
        except ValueError:
            try:
                contractionLowerIndicesLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(contraction[1])])
            except ValueError:
                contractionLowerIndicesLettersList.append(string.ascii_lowercase[len(lowerIndexList) + len(upperIndexList) + nNewIndices])
                nNewIndices += 1

    # for c, contraction in enumerate(contractionsList):
    #     if not contraction[0] in indexLettersDictionary.keys():
    #         indexLettersDictionary[contraction[0]] = string.ascii_lowercase[len(indexLettersDictionary)]
    #     if not contraction[1] in indexLettersDictionary.keys():
    #         indexLettersDictionary[contraction[1]] = string.ascii_lowercase[len(indexLettersDictionary)]
    #     lowerIndexList.append(contraction[1])
    #     upperIndexList.append(contraction[0])

    # targetLowerIndices, targetUpperIndices = targetLowerIndices_, targetUpperIndices_
    # if targetLowerIndices is None:
    #     targetLowerIndices = [lowerIndex for lI, lowerIndex in enumerate(lowerIndexList) if lowerIndex not in upperIndexList]
    # if targetUpperIndices is None:
    #     targetUpperIndices = [upperIndex for uI, upperIndex in enumerate(upperIndexList) if upperIndex not in lowerIndexList]

    specificContractionIndicesList = []
    extraContractionsList = []
    extraContractionsEinsumSubstrings = []
    extraContractionsMatrices = []
#     nonSpecificContractionsList = []
    contractedList = [False] * len(contractionsList)
    for c, contraction in enumerate(contractionsList):
        if not isinstance(contraction[0], index.SpecificOrbitalIndex) and not isinstance(contraction[1], index.SpecificOrbitalIndex):
#        contracted = False
            for lI, lowerIndex in enumerate(lowerIndexList):
                if contraction[0] == lowerIndex:
                    contractedList[c] = True
                    try:
                        newUpperIndexLettersList[upperIndexList.index(contraction[1])] = lowerIndexLettersList[lI]
                    except ValueError:
                        pass
                        # print("contraction", *contraction)
                        # print("upper Index List", *upperIndexList)
            if not contractedList[c]:
                for uI, upperIndex in enumerate(upperIndexList):
                    if contraction[1] == upperIndex:
                        contractedList[c] = True
                        try:
                            newLowerIndexLettersList[lowerIndexList.index(contraction[0])] = upperIndexLettersList[uI]
                        except ValueError:
                            pass
                            # print("contraction", *contraction)
                            # print("lower Index List", *lowerIndexList)
        elif isinstance(contraction[0], index.SpecificOrbitalIndex) and isinstance(contraction[1], index.SpecificOrbitalIndex):
            specificContractionIndicesList.append(c)
    if tensorProduct.tensorList == [] and len(specificContractionIndicesList) == len(contractionsList):
        result = tensorProduct.prefactor * prefactor * int(np.all([contraction[0].value == contraction[1].value for c, contraction in enumerate (contractionsList)]))
        return result, [], []
    if targetLowerIndices is None:
        targetLowerIndexLettersList = [lIL for lIL in lowerIndexLettersList if lIL not in newUpperIndexLettersList]
    else:
#         print("tarL", *targetLowerIndices)
        targetLowerIndexLettersList = []
        for targetLowerIndex in targetLowerIndices:
            traced = False
            tLI = targetLowerIndex
            while not traced:
                if tLI in lowerIndexList:
                    targetLowerIndexLettersList.append(lowerIndexLettersList[lowerIndexList.index(tLI)])
                    traced = True
                elif tLI in targetUpperIndices:
                    targetLowerIndexLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)])
                    extraContractionsList.append((tLI, targetLowerIndex))
                    extraContractionsEinsumSubstrings.append(str(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)] + contractionUpperIndicesLettersList[contractionsUpperIndices.index(tLI)]))
                    extraContractionsMatrices.append(np.eye(max(resultShape[targetLowerIndices.index(targetLowerIndex)], resultShape[len(targetLowerIndices) + targetUpperIndices.index(tLI)])))
                    traced = True
                else:
                    for contraction in contractionsList:
                        if contraction[1] == tLI:
                            # print("trace", *contraction)
                            tLI = contraction[0]
            # try:
            #     targetLowerIndexLettersList.append(lowerIndexLettersList[lowerIndexList.index(targetLowerIndex)])
            # except ValueError:
            #     traced = False
            #     currentLetter = contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)]
            #     while not traced:
            #         if currentLetter in lowerIndexLettersList:
            #             targetLowerIndexLettersList.append(currentLetter)
            #             traced = True
            #         else:
            #             currentLetter = contractionUpperIndicesLettersList[contractionLowerIndicesLettersList.index(currentLetter)]
    if targetUpperIndices is None:
        targetUpperIndexLettersList = [uIL for uIL in newUpperIndexLettersList if uIL not in lowerIndexLettersList]
    else:
        # print("tarU", *targetUpperIndices)
        targetUpperIndexLettersList = []
        for targetUpperIndex in targetUpperIndices:
            traced = False
            tUI = targetUpperIndex
            while not traced:
                if tUI in upperIndexList:
                    targetUpperIndexLettersList.append(upperIndexLettersList[upperIndexList.index(tUI)])
                    traced = True
                elif tUI in targetLowerIndices:
                    targetUpperIndexLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)])
                    traced = True
                else:
                    for contraction in contractionsList:
                        if contraction[0] == tUI:
                            tUI = contraction[1]
            # try:
            #     targetUpperIndexLettersList.append(upperIndexLettersList[upperIndexList.index(targetUpperIndex)])
            # except ValueError:
            #     targetUpperIndexLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)])
            #     traced = False
            #     tUI = targetUpperIndex
            #     while not traced:
            #         if tUI in upperIndexList:
            #             targetUpperIndexLettersList.append(upperIndexLettersList[upperIndexList.index(tUI)])
            #             traced = True
            #         else:
            #             for contraction in contractionsList:
            #                 if contraction[0] == tUI:
            #                     tUI = contraction[1]
    lowerIndexLetters = "".join(newLowerIndexLettersList)
    upperIndexLetters = "".join(newUpperIndexLettersList)
    resultLowerIndexLetters = "".join(targetLowerIndexLettersList)
    resultUpperIndexLetters = "".join(targetUpperIndexLettersList)
#    upperIndexLetters = ""
#    resultLowerIndexLetters = ""
#    resultUpperIndexLetters = ""
#     for c, contraction in enumerate(contractionsList):
#         if contraction[0] in targetUpperIndices:
#             targetUpperIndices[targetUpperIndices.index(contraction[0])] = contraction[1]
#             contractionsList.pop(c)
#         if contraction[1] in targetLowerIndices:
#             targetLowerIndices[targetLowerIndices.index(contraction[1])] = contraction[0]
#             contractionsList.pop(c)
#     for uI, upperIndex in enumerate(upperIndexList):
#         contracted = False
#         for c, contraction in enumerate(contractionsList):
#             if not isinstance(contraction[0], index.SpecificOrbitalIndex) and contraction[0] not in targetUpperIndices:
#                 if not isinstance(contraction[1], index.SpecificOrbitalIndex) and contraction[1] not in targetLowerIndices:
#                     if contraction[1] == upperIndex:
#                         upperIndexLetters += lowerIndexLetters[lowerIndexList.index(contraction[0])]
#                         contracted = True
#         if not contracted:
#             upperIndexLetters += string.ascii_lowercase[len(lowerIndexList)+len(upperIndexLetters)]
# #            resultUpperIndexLetters += string.ascii_lowercase[len(lowerIndexList)+len(upperIndexLetters)]
#             resultUpperIndexLetters += upperIndexLetters[-1]
#     for lI, lowerIndex in enumerate(lowerIndexList):
#         contracted = False
#         for c, contraction in enumerate(contractionsList):
#             if not isinstance(contraction[0], index.SpecificOrbitalIndex) and contraction[0] not in targetUpperIndices:
#                 if not isinstance(contraction[1], index.SpecificOrbitalIndex) and contraction[1] not in targetLowerIndices:
#                     if contraction[0] == lowerIndex:
#                         contracted = True
#         if not contracted:
#             resultLowerIndexLetters += lowerIndexLetters[lI]
# #            resultLowerIndexLetters += lowerIndexLetters[lowerIndexList.index(contraction[0])]
# #                upperIndexLetters[]
#     if targetLowerIndices is not None:
#         resultLowerIndexLetters = "".join([lowerIndexLetters[lowerIndexList.index(lowerIndex)] for lI, lowerIndex in enumerate(targetLowerIndices)])
#     if targetUpperIndices is not None:
#         resultUpperIndexLetters = "".join([upperIndexLetters[upperIndexList.index(upperIndex)] for uI, upperIndex in enumerate(targetUpperIndices)])
    einsumSubstrings = []
    vertexSlices = []
    start = 0
    for vertex in tensorProduct.vertexList:
        end = start + vertex.excitationRank
        einsumSubstring = lowerIndexLetters[start:end] + upperIndexLetters[start:end]
        einsumSubstrings.append(einsumSubstring)
        start = end
        vertexSlice = []
        for lI, lowerIndex in enumerate(vertex.lowerIndices):
#            followedIndex, specificValues = followLowerIndexThroughContractions(lowerIndex, contractionsList)
            followedIndex = followLowerIndexThroughContractions(lowerIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        for uI, upperIndex in enumerate(vertex.upperIndices):
#            followedIndex, specificValues = followUpperIndexThroughContractions(upperIndex, contractionsList)
            followedIndex = followUpperIndexThroughContractions(upperIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        vertexSlices.append(tuple(vertexSlice))
    # for vertex in tensorProduct.vertexList:
    #     einsumSubstrings.append("".join([indexLettersDictionary[ind] for ind in vertex.lowerIndices]) + "".join([indexLettersDictionary[followUpperIndexThroughContractions(ind, contractionsList)] for ind in vertex.upperIndices]))
    einsumString = ",".join(einsumSubstrings + extraContractionsEinsumSubstrings)
    einsumString += "->"
    # einsumString += "".join([indexLettersDictionary[ind] for ind in targetLowerIndices]) + "".join([indexLettersDictionary[ind] for ind in targetUpperIndices])
    einsumString += resultLowerIndexLetters
    einsumString += resultUpperIndexLetters
    #print(einsumString)
    #print(*tensorProduct.vertexList)
    #print(vertexSlices)
    #print(*[vertex.tensor.array.shape for v, vertex in enumerate(tensorProduct.vertexList)])
#    result = prefactor * tensorProduct.prefactor * np.einsum(einsumString, *[vertex.tensor.getArray()[vertexSlices[v]] for v, vertex in enumerate(tensorProduct.vertexList)], optimize='optimal') # optimised einsum ordering 16/01/2023
    result = prefactor * tensorProduct.prefactor * np.einsum(einsumString, *[maskArrayBySlice(vertex.tensor.getArray(), vertexSlices[v]) for v, vertex in enumerate(tensorProduct.vertexList)], *extraContractionsMatrices, optimize='optimal') # optimised einsum ordering 16/01/2023
#    slicedResult = np.zeros_like(result)
#     nonzeroSlices = [slice(None)] * len(result.shape)
#     for c, contraction in enumerate(contractionsList):
#         if isinstance(contraction[1], index.SpecificOrbitalIndex):
#             if isinstance(contraction[0], index.SpecificOrbitalIndex):
#                 if contraction[0].value != contraction[1].value:
#                     return slicedResult
#             else:
#                 if contraction[0] in lowerIndexList:
#                     axis = resultLowerIndexLetters.index(lowerIndexLetters[lowerIndexList.index(contraction[0])])
# #                if contraction[0] in targetLowerIndices:
# #                    axis = targetLowerIndices.index(contraction[0])
#                     nonzeroSlices[axis] = slice(contraction[1].value, contraction[1].value + 1)
#         elif isinstance(contraction[0], index.SpecificOrbitalIndex):
#             if contraction[1] in upperIndexList:
#                 axis = len(resultLowerIndexLetters) + resultUpperIndexLetters.index(upperIndexLetters[upperIndexList.index(contraction[1])])
# #            if contraction[1] in targetUpperIndices:
# #                axis = len(targetLowerIndices) + targetUpperIndices.index(contraction[1])
#                 nonzeroSlices[axis] = slice(contraction[0].value, contraction[0].value + 1)
#     slicedResult[tuple(nonzeroSlices)] = result[tuple(nonzeroSlices)]
    if not np.all([contractionsList[c][0].value == contractionsList[c][1].value for c in specificContractionIndicesList]):
        result = np.zeros_like(result)
    # finalLowerIndices = [followLowerIndexThroughContractions(lowerIndexList[lowerIndexLettersList.index(lowerIndexLetter)], contractionsList) for lIL, lowerIndexLetter in enumerate(resultLowerIndexLetters)]
    # finalUpperIndices = [followUpperIndexThroughContractions(upperIndexList[upperIndexLettersList.index(upperIndexLetter)], contractionsList) for uIL, upperIndexLetter in enumerate(resultUpperIndexLetters)]
    specificIndexSlices = []
    for fI, finalIndex in enumerate(targetLowerIndices):
        # finalIndexSpecificValue = findLowerIndexSpecificValue(finalIndex, lowerIndexList + contractionsLowerIndices, upperIndexList + contractionsUpperIndices)
        finalIndexSpecificValue = findLowerIndexSpecificValue(finalIndex, contractionsLowerIndices, contractionsUpperIndices)
        # if isinstance(finalIndex, index.SpecificOrbitalIndex):
        #     specificIndexSlices.append(slice(finalIndex.value, finalIndex.value+1))
        if isinstance(finalIndexSpecificValue, int):
            specificIndexSlices.append(slice(finalIndexSpecificValue, finalIndexSpecificValue+1))
        else:
            specificIndexSlices.append(slice(None))
    for fI, finalIndex in enumerate(targetUpperIndices):
        # finalIndexSpecificValue = findUpperIndexSpecificValue(finalIndex, lowerIndexList + contractionsLowerIndices, upperIndexList + contractionsUpperIndices)
        finalIndexSpecificValue = findUpperIndexSpecificValue(finalIndex, contractionsLowerIndices, contractionsUpperIndices)
        # if isinstance(finalIndex, index.SpecificOrbitalIndex):
        #     specificIndexSlices.append(slice(finalIndex.value, finalIndex.value+1))
        if isinstance(finalIndexSpecificValue, int):
            specificIndexSlices.append(slice(finalIndexSpecificValue, finalIndexSpecificValue+1))
        else:
            specificIndexSlices.append(slice(None))
    return maskArrayBySlice(result, tuple(specificIndexSlices))
#     if targetLowerIndices is None and targetUpperIndices is None:
#         return result, finalLowerIndices, finalUpperIndices
#     else:
#         specificIndexSlices = []
#         for fI, finalIndex in enumerate(targetLowerIndices + targetUpperIndices):
# #        for fI, finalIndex in enumerate(finalLowerIndices + finalUpperIndices):
#             if isinstance(finalIndex, index.SpecificOrbitalIndex):
#                 specificIndexSlices.append(slice(finalIndex.value, finalIndex.value+1))
#             else:
#                 specificIndexSlices.append(slice(None))
#         return result[tuple(specificIndexSlices)]
#        return slicedResult, [lowerIndexList[lowerIndexLettersList.index(lowerIndexLetter)] for lIL, lowerIndexLetter in enumerate(resultLowerIndexLetters)], [upperIndexList[upperIndexLettersList.index(upperIndexLetter)] for uIL, upperIndexLetter in enumerate(resultUpperIndexLetters)]
#        return slicedResult, targetLowerIndices, targetUpperIndices
#    return slicedResult
#    return result

def getEinsumInformationNew(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    if targetLowerIndices is None:
        targetLowerIndices = tensorProduct.freeLowerIndices
    if targetUpperIndices is None:
        targetUpperIndices = tensorProduct.freeUpperIndices
    lowerIndexList = list(itertools.chain.from_iterable([vertex.lowerIndices for vertex in tensorProduct.vertexList]))
    upperIndexList = list(itertools.chain.from_iterable([vertex.upperIndices for vertex in tensorProduct.vertexList]))
    lowerIndexLetters = string.ascii_lowercase[:len(lowerIndexList)]
    upperIndexLetters = string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)]
    lowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    upperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    newLowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    newUpperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    targetLowerIndexLettersList = []
    targetUpperIndexLettersList = []

    contractionsList = tensorProduct.contractionsList + contractionsList_
    contractionsLowerIndices = []
    contractionsUpperIndices = []
    contractionLowerIndicesLettersList = []
    contractionUpperIndicesLettersList = []
    nNewIndices = 0
    for c, contraction in enumerate(contractionsList):
        contractionsLowerIndices.append(contraction[1])
        contractionsUpperIndices.append(contraction[0])
        try:
            contractionUpperIndicesLettersList.append(lowerIndexLetters[lowerIndexList.index(contraction[0])])
        except ValueError:
            try:
                contractionUpperIndicesLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(contraction[0])])
            except (ValueError, IndexError):
                contractionUpperIndicesLettersList.append(string.ascii_lowercase[len(lowerIndexList) + len(upperIndexList) + nNewIndices])
                nNewIndices += 1
        try:
            contractionLowerIndicesLettersList.append(upperIndexLetters[upperIndexList.index(contraction[1])])
        except ValueError:
            try:
                contractionLowerIndicesLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(contraction[1])])
            except ValueError:
                contractionLowerIndicesLettersList.append(string.ascii_lowercase[len(lowerIndexList) + len(upperIndexList) + nNewIndices])
                nNewIndices += 1

    specificContractionIndicesList = []
    extraContractionsList = []
    extraContractionsEinsumSubstrings = []
    extraContractionsMatrices = []
    contractedList = [False] * len(contractionsList)
    for c, contraction in enumerate(contractionsList):
        if not isinstance(contraction[0], index.SpecificOrbitalIndex) and not isinstance(contraction[1], index.SpecificOrbitalIndex):
            for lI, lowerIndex in enumerate(lowerIndexList):
                if contraction[0] == lowerIndex:
                    contractedList[c] = True
                    try:
                        newUpperIndexLettersList[upperIndexList.index(contraction[1])] = lowerIndexLettersList[lI]
                    except ValueError:
                        pass
            if not contractedList[c]:
                for uI, upperIndex in enumerate(upperIndexList):
                    if contraction[1] == upperIndex:
                        contractedList[c] = True
                        try:
                            newLowerIndexLettersList[lowerIndexList.index(contraction[0])] = upperIndexLettersList[uI]
                        except ValueError:
                            pass
        elif isinstance(contraction[0], index.SpecificOrbitalIndex) and isinstance(contraction[1], index.SpecificOrbitalIndex):
            specificContractionIndicesList.append(c)
    # if tensorProduct.tensorList == [] and len(specificContractionIndicesList) == len(contractionsList):
    #     result = tensorProduct.prefactor * prefactor * int(np.all([contraction[0].value == contraction[1].value for c, contraction in enumerate (contractionsList)]))
    #     return result, [], []
    # if targetLowerIndices is None:
    #     targetLowerIndexLettersList = [lIL for lIL in lowerIndexLettersList if lIL not in newUpperIndexLettersList]
    # else:
    #     targetLowerIndexLettersList = []
        # for targetLowerIndex in targetLowerIndices:
        #     traced = False
        #     tLI = targetLowerIndex
        #     while not traced:
        #         if tLI in lowerIndexList:
        #             targetLowerIndexLettersList.append(lowerIndexLettersList[lowerIndexList.index(tLI)])
        #             traced = True
        #         elif tLI in targetUpperIndices:
        #             targetLowerIndexLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)])
        #             extraContractionsList.append((tLI, targetLowerIndex))
        #             extraContractionsEinsumSubstrings.append(str(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)] + contractionUpperIndicesLettersList[contractionsUpperIndices.index(tLI)]))
        #             extraContractionsMatrices.append(np.eye(max(resultShape[targetLowerIndices.index(targetLowerIndex)], resultShape[len(targetLowerIndices) + targetUpperIndices.index(tLI)])))
        #             traced = True
        #         else:
        #             # found = False
        #             for contraction in contractionsList:
        #                 if contraction[1] == tLI:
        #                     found = True
        #                     tLI = contraction[0]
        #                     break
                    # if not found:
                    #     if isinstance(tLI, index.SpecificOrbitalIndex):
                    #         targetLowerIndexLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)])
                    #         extraContractionsList.append((tLI, targetLowerIndex))
                    #         extraContractionsEinsumSubstrings.append(str(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)] + contractionUpperIndicesLettersList[contractionsUpperIndices.index(tLI)]))
                    #         extraContractionsMatrices.append(np.eye(max(resultShape[targetLowerIndices.index(targetLowerIndex)], resultShape[len(targetLowerIndices) + targetUpperIndices.index(tLI)])))
                    #         traced = True
    # if targetUpperIndices is None:
    #     targetUpperIndexLettersList = [uIL for uIL in newUpperIndexLettersList if uIL not in lowerIndexLettersList]
    # else:
    #     targetUpperIndexLettersList = []
        # for targetUpperIndex in targetUpperIndices:
        #     traced = False
        #     tUI = targetUpperIndex
        #     while not traced:
        #         if tUI in upperIndexList:
        #             targetUpperIndexLettersList.append(upperIndexLettersList[upperIndexList.index(tUI)])
        #             traced = True
        #         elif tUI in targetLowerIndices:
        #             targetUpperIndexLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)])
        #             traced = True
        #         else:
        #             # found = False
        #             for contraction in contractionsList:
        #                 if contraction[0] == tUI:
        #                     tUI = contraction[1]
        #                     break
                    # if not found:
                    #     if isinstance(tUI, index.SpecificOrbitalIndex):
                    #         targetUpperIndexLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)])
                    #         extraContractionsList.append((targetUpperIndex, tUI))
                    #         extraContractionsEinsumSubstrings.append(str(contractionLowerIndicesLettersList[contractionsLowerIndices.index(tUI)] + contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)]))
                    #         extraContractionsMatrices.append(np.eye(max(resultShape[targetLowerIndices.index(tUI)], resultShape[len(targetLowerIndices) + targetUpperIndices.index(targetUpperIndex)])))
                    #         traced = True
    lowerIndexLetters = "".join(newLowerIndexLettersList)
    upperIndexLetters = "".join(newUpperIndexLettersList)

    for tLI, targetLowerIndex in enumerate(targetLowerIndices):
        try:
            targetLowerIndexLettersList.append(lowerIndexLetters[lowerIndexList.index(targetLowerIndex)])
        except ValueError:
            targetLowerIndexLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)])
    for tUI, targetUpperIndex in enumerate(targetUpperIndices):
        try:
            targetUpperIndexLettersList.append(upperIndexLetters[upperIndexList.index(targetUpperIndex)])
        except ValueError:
            targetUpperIndexLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)])

    resultLowerIndexLetters = "".join(targetLowerIndexLettersList)
    resultUpperIndexLetters = "".join(targetUpperIndexLettersList)

    einsumSubstrings = []
    vertexSlices = []
    start = 0
    for vertex in tensorProduct.vertexList:
        end = start + vertex.excitationRank
        einsumSubstring = lowerIndexLetters[start:end] + upperIndexLetters[start:end]
        einsumSubstrings.append(einsumSubstring)
        start = end
        vertexSlice = []
        for lI, lowerIndex in enumerate(vertex.lowerIndices):
            followedIndex = followLowerIndexThroughContractions(lowerIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        for uI, upperIndex in enumerate(vertex.upperIndices):
            followedIndex = followUpperIndexThroughContractions(upperIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        vertexSlices.append(tuple(vertexSlice))
    einsumString = ",".join(einsumSubstrings + extraContractionsEinsumSubstrings)
    einsumString += "->"
    einsumString += resultLowerIndexLetters
    einsumString += resultUpperIndexLetters

    specificIndexSlices = []
    for fI, finalIndex in enumerate(targetLowerIndices):
        finalIndexSpecificValue = findLowerIndexSpecificValue(finalIndex, contractionsLowerIndices, contractionsUpperIndices)
        if isinstance(finalIndexSpecificValue, int):
            specificIndexSlices.append(slice(finalIndexSpecificValue, finalIndexSpecificValue+1))
        else:
            specificIndexSlices.append(slice(None))
    for fI, finalIndex in enumerate(targetUpperIndices):
        finalIndexSpecificValue = findUpperIndexSpecificValue(finalIndex, contractionsLowerIndices, contractionsUpperIndices)
        if isinstance(finalIndexSpecificValue, int):
            specificIndexSlices.append(slice(finalIndexSpecificValue, finalIndexSpecificValue+1))
        else:
            specificIndexSlices.append(slice(None))
    finalSlices = tuple(specificIndexSlices)
    return (einsumString, vertexSlices, finalSlices, extraContractionsMatrices)

def getEinsumInformation(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    if targetLowerIndices is None:
        targetLowerIndices = tensorProduct.freeLowerIndices
    if targetUpperIndices is None:
        targetUpperIndices = tensorProduct.freeUpperIndices
    # targetLowerIndices = tensorProduct.freeLowerIndices
    # targetUpperIndices = tensorProduct.freeUpperIndices
    lowerIndexList = list(itertools.chain.from_iterable([vertex.lowerIndices for vertex in tensorProduct.vertexList]))
    upperIndexList = list(itertools.chain.from_iterable([vertex.upperIndices for vertex in tensorProduct.vertexList]))
    lowerIndexLetters = string.ascii_lowercase[:len(lowerIndexList)]
    upperIndexLetters = string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)]
    lowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    upperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    newLowerIndexLettersList = list(string.ascii_lowercase[:len(lowerIndexList)])
    newUpperIndexLettersList = list(string.ascii_lowercase[len(lowerIndexList):len(lowerIndexList)+len(upperIndexList)])
    targetLowerIndexLettersList = []
    targetUpperIndexLettersList = []

    contractionsList = tensorProduct.contractionsList + contractionsList_
    contractionsLowerIndices = []
    contractionsUpperIndices = []
    contractionLowerIndicesLettersList = []
    contractionUpperIndicesLettersList = []
    nNewIndices = 0
    for c, contraction in enumerate(contractionsList):
        contractionsLowerIndices.append(contraction[1])
        contractionsUpperIndices.append(contraction[0])
        try:
            contractionUpperIndicesLettersList.append(lowerIndexLetters[lowerIndexList.index(contraction[0])])
        except ValueError:
            try:
                contractionUpperIndicesLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(contraction[0])])
            except (ValueError, IndexError):
                contractionUpperIndicesLettersList.append(string.ascii_lowercase[len(lowerIndexList) + len(upperIndexList) + nNewIndices])
                nNewIndices += 1
        try:
            contractionLowerIndicesLettersList.append(upperIndexLetters[upperIndexList.index(contraction[1])])
        except ValueError:
            try:
                contractionLowerIndicesLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(contraction[1])])
            except ValueError:
                contractionLowerIndicesLettersList.append(string.ascii_lowercase[len(lowerIndexList) + len(upperIndexList) + nNewIndices])
                nNewIndices += 1

    specificContractionIndicesList = []
    specificContractionFactor = 1.
    extraContractionsList = []
    extraContractionsEinsumSubstrings = []
    extraContractionsMatrices = []
    contractedList = [False] * len(contractionsList)
    for c, contraction in enumerate(contractionsList):
        if not isinstance(contraction[0], index.SpecificOrbitalIndex) and not isinstance(contraction[1], index.SpecificOrbitalIndex):
            for lI, lowerIndex in enumerate(lowerIndexList):
                if contraction[0] == lowerIndex:
                    contractedList[c] = True
                    try:
                        newUpperIndexLettersList[upperIndexList.index(contraction[1])] = lowerIndexLettersList[lI]
                    except ValueError:
                        pass
            if not contractedList[c]:
                for uI, upperIndex in enumerate(upperIndexList):
                    if contraction[1] == upperIndex:
                        contractedList[c] = True
                        try:
                            newLowerIndexLettersList[lowerIndexList.index(contraction[0])] = upperIndexLettersList[uI]
                        except ValueError:
                            pass
        elif isinstance(contraction[0], index.SpecificOrbitalIndex) and isinstance(contraction[1], index.SpecificOrbitalIndex):
            specificContractionIndicesList.append(c)
            specificContractionFactor = specificContractionFactor * int(contraction[0].value == contraction[1].value)
    # if tensorProduct.tensorList == [] and len(specificContractionIndicesList) == len(contractionsList):
    #     result = tensorProduct.prefactor * prefactor * int(np.all([contraction[0].value == contraction[1].value for c, contraction in enumerate (contractionsList)]))
    #     return result, [], []
    if targetLowerIndices is None:
        targetLowerIndexLettersList = [lIL for lIL in lowerIndexLettersList if lIL not in newUpperIndexLettersList]
    else:
        targetLowerIndexLettersList = []
        for targetLowerIndex in targetLowerIndices:
            traced = False
            tLI = targetLowerIndex
            while not traced:
                if tLI in lowerIndexList:
                    targetLowerIndexLettersList.append(lowerIndexLettersList[lowerIndexList.index(tLI)])
                    traced = True
                elif tLI in targetUpperIndices:
                    targetLowerIndexLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)])
                    extraContractionsList.append((tLI, targetLowerIndex))
                    extraContractionsEinsumSubstrings.append(str(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)] + contractionUpperIndicesLettersList[contractionsUpperIndices.index(tLI)]))
                    extraContractionsMatrices.append(np.eye(max(resultShape[targetLowerIndices.index(targetLowerIndex)], resultShape[len(targetLowerIndices) + targetUpperIndices.index(tLI)])))
                    traced = True
                elif isinstance(tLI, index.SpecificOrbitalIndex):
                    targetLowerIndexLettersList.append(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)])
                    extraContractionsList.append((tLI, targetLowerIndex))
                    extraContractionsEinsumSubstrings.append(str(contractionLowerIndicesLettersList[contractionsLowerIndices.index(targetLowerIndex)] + contractionUpperIndicesLettersList[contractionsUpperIndices.index(tLI)]))
                    # extraContractionsMatrices.append(np.eye(max(resultShape[targetLowerIndices.index(targetLowerIndex)], resultShape[len(targetLowerIndices) + targetUpperIndices.index(tLI)])))
                    extraContractionsMatrices.append(np.eye(resultShape[targetLowerIndices.index(targetLowerIndex)]))
                    traced = True
                else:
                    # found = False
                    for contraction in contractionsList:
                        if contraction[1] == tLI:
                            found = True
                            tLI = contraction[0]
                            break
    if targetUpperIndices is None:
        targetUpperIndexLettersList = [uIL for uIL in newUpperIndexLettersList if uIL not in lowerIndexLettersList]
    else:
        targetUpperIndexLettersList = []
        for targetUpperIndex in targetUpperIndices:
            traced = False
            tUI = targetUpperIndex
            while not traced:
                if tUI in upperIndexList:
                    targetUpperIndexLettersList.append(upperIndexLettersList[upperIndexList.index(tUI)])
                    traced = True
                elif tUI in targetLowerIndices:
                    targetUpperIndexLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)])
                    traced = True
                elif isinstance(tUI, index.SpecificOrbitalIndex):
                    targetUpperIndexLettersList.append(contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)])
                    extraContractionsList.append((targetUpperIndex, tUI))
                    extraContractionsEinsumSubstrings.append(str(contractionLowerIndicesLettersList[contractionsLowerIndices.index(tUI)] + contractionUpperIndicesLettersList[contractionsUpperIndices.index(targetUpperIndex)]))
                    # extraContractionsMatrices.append(np.eye(max(resultShape[targetLowerIndices.index(tUI)], resultShape[len(targetLowerIndices) + targetUpperIndices.index(targetUpperIndex)])))
                    extraContractionsMatrices.append(np.eye(resultShape[len(targetLowerIndices) + targetUpperIndices.index(targetUpperIndex)]))
                    traced = True
                else:
                    # found = False
                    for contraction in contractionsList:
                        if contraction[0] == tUI:
                            tUI = contraction[1]
                            break
                    # if not found:

    lowerIndexLetters = "".join(newLowerIndexLettersList)
    upperIndexLetters = "".join(newUpperIndexLettersList)
    resultLowerIndexLetters = "".join(targetLowerIndexLettersList)
    resultUpperIndexLetters = "".join(targetUpperIndexLettersList)

    einsumSubstrings = []
    vertexSlices = []
    start = 0
    for vertex in tensorProduct.vertexList:
        end = start + vertex.excitationRank
        einsumSubstring = lowerIndexLetters[start:end] + upperIndexLetters[start:end]
        einsumSubstrings.append(einsumSubstring)
        start = end
        vertexSlice = []
        for lI, lowerIndex in enumerate(vertex.lowerIndices):
            followedIndex = followLowerIndexThroughContractions(lowerIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        for uI, upperIndex in enumerate(vertex.upperIndices):
            followedIndex = followUpperIndexThroughContractions(upperIndex, contractionsList)
            if isinstance(followedIndex, index.SpecificOrbitalIndex):
                vertexSlice.append(slice(followedIndex.value, followedIndex.value + 1))
            else:
                vertexSlice.append(slice(None))
        vertexSlices.append(tuple(vertexSlice))
    einsumString = ",".join(einsumSubstrings + extraContractionsEinsumSubstrings)
    einsumString += "->"
    einsumString += resultLowerIndexLetters
    einsumString += resultUpperIndexLetters

    specificIndexSlices = []
    for fI, finalIndex in enumerate(targetLowerIndices):
        finalIndexSpecificValue = findLowerIndexSpecificValue(finalIndex, contractionsLowerIndices, contractionsUpperIndices)
        if isinstance(finalIndexSpecificValue, int):
            specificIndexSlices.append(slice(finalIndexSpecificValue, finalIndexSpecificValue+1))
        else:
            specificIndexSlices.append(slice(None))
    for fI, finalIndex in enumerate(targetUpperIndices):
        finalIndexSpecificValue = findUpperIndexSpecificValue(finalIndex, contractionsLowerIndices, contractionsUpperIndices)
        if isinstance(finalIndexSpecificValue, int):
            specificIndexSlices.append(slice(finalIndexSpecificValue, finalIndexSpecificValue+1))
        else:
            specificIndexSlices.append(slice(None))
    finalSlices = tuple(specificIndexSlices)
    return (einsumString, vertexSlices, finalSlices, extraContractionsMatrices, specificContractionFactor)

def getContractedArray(tensorProduct, contractionsList_=[], prefactor=1.0, targetLowerIndices=None, targetUpperIndices=None, resultShape=None):
    try:
        (einsumString, vertexSlices, finalSlices, extraContractionsMatrices, specificContractionFactor) = tensorProduct.einsumInformation
    except AttributeError:
        tensorProduct.einsumInformation = getEinsumInformation(tensorProduct, contractionsList_, prefactor, targetLowerIndices, targetUpperIndices, resultShape)
        (einsumString, vertexSlices, finalSlices, extraContractionsMatrices, specificContractionFactor) = tensorProduct.einsumInformation
    if not specificContractionFactor:
        return 0
    if len(tensorProduct.vertexList) == 0 and len(extraContractionsMatrices) == 0:
        return prefactor * tensorProduct.prefactor
    result = prefactor * tensorProduct.prefactor * np.einsum(einsumString, *[maskArrayBySlice(vertex.tensor.getArray(), vertexSlices[v]) for v, vertex in enumerate(tensorProduct.vertexList)], *extraContractionsMatrices, optimize='optimal') # optimised einsum ordering 16/01/2023
    return maskArrayBySlice(result, finalSlices)


def followUpperIndexThroughContractions(upperIndex, contractionsList):
    '''
    Take an upper index and follow it through a list of contractions
    '''
    currentIndex = upperIndex
    specificValues = []
    while True:
    # while not isinstance(currentIndex, index.SpecificOrbitalIndex):
#        if isinstance(currentIndex, index.SpecificOrbitalIndex):
#            specificValues.append(currentIndex.value)
        found = False
        for c, contraction in enumerate(contractionsList):
            if contraction[1] == currentIndex:
                found = True
                currentIndex = contraction[0]
                if isinstance(currentIndex, index.SpecificOrbitalIndex):
                    return currentIndex
                break
        if not found:
            break
    return currentIndex#, specificValues

def followLowerIndexThroughContractions(lowerIndex, contractionsList):
    '''
    Take a lower index and follow it through a list of contractions
    '''
    currentIndex = lowerIndex
    specificValues = []
    # while not isinstance(currentIndex, index.SpecificOrbitalIndex):
    while True:
#        if isinstance(currentIndex, index.SpecificOrbitalIndex):
#            specificValues.append(currentIndex.value)
        found = False
        for c, contraction in enumerate(contractionsList):
            if contraction[0] == currentIndex:
                found = True
                currentIndex = contraction[1]
                if isinstance(currentIndex, index.SpecificOrbitalIndex):
                    return currentIndex
                break
        if not found:
            break
    return currentIndex#, specificValues

def findLowerIndexSpecificValue(lowerIndex, lowerIndexList, upperIndexList):
    currentIndex = lowerIndex
    while True:
        if isinstance(currentIndex, index.SpecificOrbitalIndex):
            return currentIndex.value
        try:
            if upperIndexList[lowerIndexList.index(currentIndex)] is lowerIndex:
                return None
            currentIndex = upperIndexList[lowerIndexList.index(currentIndex)]
        except ValueError:
            return None

def findUpperIndexSpecificValue(upperIndex, lowerIndexList, upperIndexList):
    currentIndex = upperIndex
    while True:
        if isinstance(currentIndex, index.SpecificOrbitalIndex):
            return currentIndex.value
        try:
            if lowerIndexList[upperIndexList.index(currentIndex)] is upperIndex:
                return None
            currentIndex = lowerIndexList[upperIndexList.index(currentIndex)]
        except ValueError:
            return None

def maskArrayBySlice(array, slice):
    result = np.zeros_like(array)
    result[slice] = array[slice]
    return result