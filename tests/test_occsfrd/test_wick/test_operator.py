'''
Classes and functions for fermionic operator algebra to implement Wick contractions
'''
import numpy as np
from numbers import Number
from copy import copy

class BasicOperator:
    '''
    Class for a basic fermionic creation or annihilation operator, with arbitrary index
    
    Attributes:
    index                 (index.Index): the index for the orbitals on which this operator acts
    creation_annihilation (bool)       : True for creation, False for annihilation
    spin                  (bool)       : True for alpha, False for beta
    quasi_cre_ann         (bool)       : creation or annihilation with respect to Fermi vacuum
    '''
    def __init__(self, index_, creation_annihilation_, spin_):
        self.index = index_
        self.spin = spin_
        self.creation_annihilation = creation_annihilation_
        self.quasi_cre_ann = not (self.creation_annihilation == self.index.occupiedInVacuum)

    def conjugate(self):
        '''
        Returns:
        (BasicOperator) Hermitian conjugate of self
        '''
        return BasicOperator(self.index, not self.creation_annihilation, self.spin)

    def __copy__(self):
        return BasicOperator(self.index, self.creation_annihilation, self.spin)

    def __str__(self):
        string = "a"
        if self.creation_annihilation:
            string = string + "^"
        else:
            string = string + "_"
        if self.spin:
            string = string + "{" + self.index.__str__() + "\\alpha}"
        else:
            string = string + "{" + self.index.__str__() + "\\beta}"
        return string

    def __eq__(self, other):
        if isinstance(other, BasicOperator):
            return self.index == other.index and self.spin == other.spin and self.creation_annihilation == other.creation_annihilation
        return False

    def __mul__(self, other):
        if isinstance(other, Number):
            return OperatorProduct([self], other)
        elif isinstance(other, BasicOperator):
            return OperatorProduct([self, other])
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return OperatorProduct([self], other)
        else:
            return NotImplemented


class OperatorProduct:
    '''
    Product of creation/annihilation operators
    
    Attributes:
    operatorList             (list) : list of BasicOperator objects representing the creation/annihilation operators in the product
    prefactor                (float): prefactor
    contractionsList         (list) : list of (Index, Index) tuples representing contractions between indices appearing in the product
    normalOrderedStartPoints (list) : positions at which any normal ordered subsequences begin.
                                      Contractions within these subsequences do not need to be checked
    '''
    def __init__(self, operatorList_=[], prefactor_=1., contractionsList_=[], normalOrderedStartPoints_=[]):
        self.operatorList = operatorList_
        self.prefactor = prefactor_
        self.contractionsList = contractionsList_
        if normalOrderedStartPoints_ == []:
            self.normalOrderedStartPoints = list(range(len(self.operatorList)))
        else:
            self.normalOrderedStartPoints = normalOrderedStartPoints_


    def isProportional(self, other):
        '''
        Check if self is proportional to other, differing only in the prefactor
        
        Inputs:
        other (OperatorProduct): operatorProduct to which self is being compared

        Returns:
        (bool) True if self is proportional to other; False otherwise
        '''
        if isinstance(other, OperatorProduct):
            return self.operatorList == other.operatorList and set(self.contractionsList) == set(other.contractionsList) and self.normalOrderedStartPoints == other.normalOrderedStartPoints
        else:
            return NotImplemented
            
    def checkNilpotency(self):
        '''
        Check if any creation or annihilation operators are applied twice without being undone in between,
        which would evaluate to zero by nilpotency of the fermionic operators

        Returns: 0 if product is zero by nilpotency; 1 otherwise
        '''
        nonZero = True
        i = 0
        while i < len(self.operatorList):
            j = i + 1
            while j < len(self.operatorList):
                if self.operatorList[j] == self.operatorList[i]:
                    nonZero = False
                elif self.operatorList[j] == self.operatorList[i].conjugate():
                    break
                j = j + 1
            i = i + 1
        return nonZero

    def conjugate(self):
        '''
        Returns:
        Hermitian conjugate of self
        '''
        return OperatorProduct([o.conjugate() for o in reversed(self.operatorList)], np.conjugate(self.prefactor), self.contractionsList)

    def __copy__(self):
        return OperatorProduct(copy(self.operatorList), self.prefactor, copy(self.contractionsList), copy(self.normalOrderedStartPoints))

    def __str__(self):
        string = str(self.prefactor)
        if(len(self.operatorList) + len(self.contractionsList) > 0):
            string = string + " * "
        for o in self.operatorList:
            string = string + o.__str__()
        for contraction in self.contractionsList:
            string = string + "\delta^{" + contraction[0].__str__() + "}_{" + contraction[1].__str__() + "}"
        return string

    def __mul__(self, other):
        if isinstance(other, BasicOperator):
            return OperatorProduct(self.operatorList + [other], self.prefactor, self.contractionsList, self.normalOrderedStartPoints + [len(self.operatorList)])
        elif isinstance(other, OperatorProduct):
            return OperatorProduct(self.operatorList + other.operatorList, self.prefactor * other.prefactor, self.contractionsList + other.contractionsList, self.normalOrderedStartPoints + [p + len(self.operatorList) for p in other.normalOrderedStartPoints])
        elif isinstance(other, OperatorSum):
#            newSummandList = []
#            for s in other.summandList:
#                newSummandList.append(self * s)
#            return operatorSum(newSummandList)
            return OperatorSum([self * s for s in other.summandList])
        elif isinstance(other, Number):
            return OperatorProduct(self.operatorList, self.prefactor * other, self.contractionsList, self.normalOrderedStartPoints)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, BasicOperator):
            return OperatorProduct([other] + self.operatorList, self.prefactor, self.contractionsList, [0] + [p + 1 for p in self.normalOrderedStartPoints])
        elif isinstance(other, Number):
            return OperatorProduct(self.operatorList, other * self.prefactor, self.contractionsList, self.normalOrderedStartPoints)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, OperatorProduct):
            if self.isProportional(other):
                return OperatorProduct(self.operatorList, self.prefactor + other.prefactor, self.contractionsList, self.normalOrderedStartPoints)
            else:
                return OperatorSum([self, other])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, OperatorProduct):
            return self.operatorList == other.operatorList and self.prefactor == other.prefactor and self.contractionsList == other.contractionsList and self.normalOrderedStartPoints == other.normalOrderedStartPoints
        elif isinstance(other, Number):
            return self.operatorList == [] and self.prefactor == other
        else:
            return NotImplemented

class OperatorSum:
    """
    Class for a sum of operator products
    
    Attributes:
        summandList (list): list of OperatorProduct objects being summed
    """
    def __init__(self, summandList_=[]):
        self.summandList = self.collectSummandList(summandList_)

    def collectSummandList(self, summandList):
        """
        Consolidate terms in summandList that are proportional

        Args:
            summandList (list): list of operator products to be consolidated

        Returns:
            list: distinct terms with total prefactors
        """
        oldSummandList = copy(summandList)
        newSummandList = []
        while len(oldSummandList) > 0:
            newSummand = oldSummandList[0]
            i = 1
            while i < len(oldSummandList):
                if newSummand.isProportional(oldSummandList[i]):
                    newSummand += oldSummandList[i]
                    oldSummandList.pop(i)
                else:
                    i += 1
            newSummandList.append(newSummand)
            oldSummandList.pop(0)
        return newSummandList

    def conjugate(self):
        """
        Returns:
            wick.operator.OperatorSum: Hermitian conjugate of self
        """
        return OperatorSum([s.conjugate() for s in self.summandList])

    def __copy__(self):
        return OperatorSum([copy(summand) for summand in self.summandList])

    def __str__(self):
        if len(self.summandList) == 0:
            return str(0)
        string = self.summandList[0].__str__()
        s = 1
        while s < len(self.summandList):
            string = string + "\n + " + self.summandList[s].__str__()
            s = s + 1
        return string

    def __add__(self, other):
        if isinstance(other, OperatorSum):
            return OperatorSum(self.summandList + other.summandList)
        elif isinstance(other, OperatorProduct):
            if other.prefactor == 0:
                return self
            newSummandList = copy(self.summandList)
            alreadyInSum = False
            for summand in newSummandList:
                if summand.isProportional(other):
                    alreadyInSum = True
                    summand.prefactor += other.prefactor
            if not alreadyInSum:
                newSummandList.append(other)
            return OperatorSum(newSummandList)
        elif other == 0:
            return self
        elif isinstance(other, Number):
            return self + OperatorProduct([], other, [], [])
        else:
            return NotImplemented
            
    def __radd__(self, other):
        if isinstance(other, OperatorProduct):
            if other.prefactor == 0:
                return self
            newSummandList = copy(self.summandList)
            alreadyInSum = False
            for summand in newSummandList:
                if summand.isProportional(other):
                    alreadyInSum = True
                    summand.prefactor += other.prefactor
            if not alreadyInSum:
                newSummandList.append(other)
            return OperatorSum(newSummandList)
        elif other == 0:
            return self
        elif isinstance(other, Number):
            return self + OperatorProduct([], other, [], [])
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, BasicOperator) or isinstance(other, OperatorProduct) or isinstance(other, Number):
#            newSummandList = []
#            for s in self.summandList:
#                newSummandList.append(s * other)
#            return operatorSum(newSummandList)
            return OperatorSum([s * other for s in self.summandList])
        elif isinstance(other, OperatorSum):
            newSummandList = []
            for o in other.summandList:
                partialSum = self * o
                newSummandList = newSummandList + partialSum.summandList
            return OperatorSum(newSummandList)
#        elif isinstance(other, Number):
#            return OperatorSum([self.summandList[s] * other for s in range(len(self.summandList))])
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, BasicOperator) or isinstance(other, Number):
            return OperatorSum([other * s for s in self.summandList])
        elif isinstance(other, OperatorProduct):
            newSummandList = []
            for s in self.summandList:
                newSummandList.append(other * s)
            return OperatorSum(newSummandList)
#        elif isinstance(other, Number):
#            return OperatorSum([other * self.summandList[s] for s in range(len(self.summandList))])
        else:
            return NotImplemented

    def __eq__(self, other):
        if self.summandList == []:
            return other == 0
        elif len(self.summandList) == 1:
            return self.summandList[0] == other
        else:
            return NotImplemented
            
def normalOrder(operator):
    """
    Args:
        operator (wick.operator.OperatorProduct) or (wick.operator.OperatorSum) and a list corresponding to which orbitals are occupied in the Fermi vacuum
    
    Returns:
        wick.operator.OperatorProduct or wick.operator.OperatorSum: normal ordered form of input, with respect to vacuum
    """
    if isinstance(operator, OperatorSum):
        return OperatorSum([normalOrder(product) for product in operator.summandList])
    quasiCreationList, quasiAnnihilationList = [], []
    quasiCreationCount = 0
    sign = 1
    for o in range(len(operator.operatorList)):
        op = operator.operatorList[o]
        if bool(op.quasi_cre_ann):
            quasiCreationList.append(op)
            if (o - quasiCreationCount) % 2 == 1:
                sign = -sign
            quasiCreationCount += 1
        else:
            quasiAnnihilationList.append(op)
    return OperatorProduct(quasiCreationList + quasiAnnihilationList, sign * operator.prefactor, operator.contractionsList, [0])

def excitation(creationIndicesList_, annihilationIndicesList_, spinList_):
    """
    Excitation operator from orbitals in annihilationIndicesList_ to orbitals in creationIndicesList_, with no spin flips

    Args:
        creationIndicesList_ (list): list of index.Index objects for orbitals into which electrons are excited
        annihilationIndicesList_ (list): list of index.Index objects for orbitals from which electrons are excited
        spinList_ (list): list of bools for the spin (True / 0 for alpha, False / 0 for beta) of each electron

    Returns:
        OperatorProduct: the excitation operator
    """
    operatorList = []
    for i in range(len(creationIndicesList_)):
        operatorList.append(BasicOperator(creationIndicesList_[i], True, spinList_[i]))
    for i in range(len(annihilationIndicesList_)):
        operatorList.append(BasicOperator(annihilationIndicesList_[-1-i], False, spinList_[-1-i]))
    return OperatorProduct(operatorList)

def spinFreeExcitation(creationList_, annihilationList_):
    """
    Spin-free excitation formed by summing excitations with all possible spin combinations

    Args:
        creationIndicesList_ (list): list of index.Index objects for orbitals into which electrons are excited
        annihilationIndicesList_ (list): list of index.Index objects for orbitals from which electrons are excited

    Returns:
        OperatorSum: the spin-free excitation operator
    """
    summandList = []
    spinLists = np.reshape(np.zeros(len(creationList_)), (1, -1))
    for i in range(len(creationList_)):
        newspinLists = copy(spinLists)
        for s in range(len(spinLists)):
            newspinLists[s,i] = 1
        spinLists = np.concatenate((spinLists, newspinLists))
    for l in range(len(spinLists)):
        spinList = spinLists[l]
        summandList.append(excitation(creationList_, annihilationList_, spinList))
    return OperatorSum(summandList)