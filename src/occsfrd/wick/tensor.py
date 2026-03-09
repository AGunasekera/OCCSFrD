"""
Classes and functions for tensor algebra to implement Wick contractions
"""
import numpy as np
from numbers import Number
from copy import copy
import itertools
import networkx as nx

from occsfrd.wick import index, operator, contractions

class Tensor:
    """
    Class for amplitude tensors of spin-free excitation operators

    Attributes:
    name            (str)       : name given to tensor
    lowerIndexTypes (list)      : list of single characters describing types of orbitals in which electrons are created by the tensor operator
    upperIndexTypes (list)      : list of single characters describing types of orbitals from which electrons are annihilated by the tensor operator
    excitationRank  (int)       : number of electrons annihilated and created
    spinFree        (bool)      : whether or not the tensor represents a spin-free operator
    array           (np.ndarray): array of coefficients

    Orbital types:
    'g': general  (any orbital)
    'p': particle (empty in fermi vacuum)
    'h': hole     (filled in fermi vacuum)
    'c': core     (doubly occupied in reference)
    'a': active   (singly occupied in reference)
    'v': virtual  (vacant in reference)
    """
    def __init__(self, name, lowerIndexTypesList, upperIndexTypesList, spinFree=True, distinguishableParticles=True):
        self.name = name
        self.lowerIndexTypes = lowerIndexTypesList
        self.upperIndexTypes = upperIndexTypesList
        self.excitationRank = len(self.lowerIndexTypes)
        self.spinFree = spinFree
        self.array = None
        self.indexRangeStartPoints = None
        self.distinguishableParticles = distinguishableParticles

    def getShape(self, vacuum):
        """
        Given a fermi vacuum and the set of index types of for the tensor (general, particle, or hole),
        find the appropriate shape of the coefficient array, and set it to zeros.

        Args:
            vacuum (list): list of int (or bool) representing the orbital occupation numbers in the vacuum; 1 in position i means orbital i is occupied in vacuo, 0 means empty.

        Returns:
            Sets self.array to the zero np.ndarray of the correct shape
        """
        Norbs = len(vacuum)
        Nocc = sum(vacuum)
        shapeList = []
        self.indexRangeStartPoints = []
        for iType in self.lowerIndexTypes:
            if iType == 'g':
                shapeList.append(Norbs)
                self.indexRangeStartPoints.append(0)
            elif iType == 'p':
                shapeList.append(Norbs - Nocc)
                self.indexRangeStartPoints.append(Nocc)
            elif iType == 'h':
                shapeList.append(Nocc)
                self.indexRangeStartPoints.append(0)
            else:
                print('Orbital index type Error')
        for iType in self.upperIndexTypes:
            if iType == 'g':
                shapeList.append(Norbs)
                self.indexRangeStartPoints.append(0)
            elif iType == 'p':
                shapeList.append(Norbs - Nocc)
                self.indexRangeStartPoints.append(Nocc)
            elif iType == 'h':
                shapeList.append(Nocc)
                self.indexRangeStartPoints.append(0)
            else:
                print('Orbital index type Error')
        if self.spinFree:
            self.array = np.zeros(tuple(shapeList))
        else:
            self.array = np.zeros(tuple([2 * length for l, length in enumerate(shapeList)]))

    def getShapeActive(self, nelec, norbs):
        '''
        Given a fermi vacuum and the set of index types of for the tensor (general, particle, hole, core, active, or virtual),
        find the appropriate shape of the coefficient array, and set it to zeros.

        Inputs:
        nelec (tuple): (number of alpha electrons, number of beta electrons)
        norbs   (int): number of orbitals

        Results:
        Sets self.array to the zero np.ndarray of the correct shape
        '''
        nCore = nelec[1]
        nActive = nelec[0] - nelec[1]
        shapeList = []
        self.indexRangeStartPoints = []
        for iType in self.lowerIndexTypes:
            if iType == 'g':
                shapeList.append(norbs)
                self.indexRangeStartPoints.append(0)
            elif iType == 'p':
                shapeList.append(norbs - nCore)
                self.indexRangeStartPoints.append(nCore)
            elif iType == 'h':
                shapeList.append(nCore)
                self.indexRangeStartPoints.append(0)
            elif iType == 'c':
                shapeList.append(nCore)
                self.indexRangeStartPoints.append(0)
            elif iType == 'a':
                shapeList.append(nActive)
                self.indexRangeStartPoints.append(nCore)
            elif iType == 'v':
                shapeList.append(norbs - nActive - nCore)
                self.indexRangeStartPoints.append(nCore + nActive)
            else:
                print('Orbital index type Error')
        for iType in self.upperIndexTypes:
            if iType == 'g':
                shapeList.append(norbs)
                self.indexRangeStartPoints.append(0)
            elif iType == 'p':
                shapeList.append(norbs - nCore)
                self.indexRangeStartPoints.append(nCore)
            elif iType == 'h':
                shapeList.append(nCore)
                self.indexRangeStartPoints.append(0)
            elif iType == 'c':
                shapeList.append(nCore)
                self.indexRangeStartPoints.append(0)
            elif iType == 'a':
                shapeList.append(nActive)
                self.indexRangeStartPoints.append(nCore)
            elif iType == 'v':
                shapeList.append(norbs - nActive - nCore)
                self.indexRangeStartPoints.append(nCore + nActive)
            else:
                print('Orbital index type Error')
        if self.spinFree:
            self.array = np.zeros(tuple(shapeList))
        else:
            self.array = np.zeros(tuple([2 * length for l, length in enumerate(shapeList)]))

    def getArray(self):
        return self.array

    def setArray(self, array):
        '''
        Sets new coeffeicient array for tensor, checking for correct shape
        '''
        if array.shape == self.array.shape:
            self.array = array
        else:
            print("Array is of wrong shape:", array.shape, self.array.shape)

    def getOperator(self, normalOrdered=True):
        '''
        Gets operator corresponding to tensor
        '''
        return TensorProduct([self]).getOperator(normalOrderedParts=normalOrdered)

    def getDiagrams(self, vacuum):
        """
        Decomposes a tensor that has multiple orbital ranges combined into its constituent SubDiagram parts, with separate orbital ranges.

        Parameters
        ----------
        vacuum : (list)
            list of int (or bool) representing the orbital occupation numbers in the vacuum; 1 in position i means orbital i is occupied in vacuo, 0 means empty.
        """
        Nocc = sum(vacuum)
        Norbs = len(vacuum)
        diagrams = []
        lowerGeneralIndexCount = sum(i == 'g' for i in self.lowerIndexTypes)
        lowerSplits = list(itertools.combinations_with_replacement(['h', 'p'], lowerGeneralIndexCount))
        upperGeneralIndexCount = sum(i == 'g' for i in self.upperIndexTypes)
        upperSplits = list(itertools.combinations_with_replacement(['h', 'p'], upperGeneralIndexCount))
        for lowerSplit in lowerSplits:
            lowerSlices = [slice(None)] * self.excitationRank
            lowerSplitIndexTypes = list(lowerSplit)
            lGI = 0
            newLowerIndexTypes = copy(self.lowerIndexTypes)
            for lI in range(len(newLowerIndexTypes)):
                if newLowerIndexTypes[lI] == 'g':
                    newLI = lowerSplitIndexTypes[lGI]
                    if newLI == 'h':
                        lowerSlices[lI] = slice(None,Nocc)
                    elif newLI == 'p':
                        lowerSlices[lI] = slice(Nocc, None)
                    newLowerIndexTypes[lI] = newLI
                    lGI += 1
            for upperSplit in upperSplits:
                upperSlices = [slice(None)] * self.excitationRank
                upperSplitIndexTypes = list(upperSplit)
                uGI = 0
                newUpperIndexTypes = copy(self.upperIndexTypes)
                for uI in range(len(newUpperIndexTypes)):
                    if newUpperIndexTypes[uI] == 'g':
                        newUI = upperSplitIndexTypes[uGI]
                        if newUI == 'h':
                            upperSlices[uI] = slice(None,Nocc)
                        elif newUI == 'p':
                            upperSlices[uI] = slice(Nocc, None)
                        newUpperIndexTypes[uI] = newUI
                        uGI += 1
                slices = tuple(lowerSlices + upperSlices)
                print(lowerSplitIndexTypes)
                print(upperSplitIndexTypes)
                print(newLowerIndexTypes)
                print(newUpperIndexTypes)
                print(slices)
#                diagram = Tensor(self.name, newLowerIndexTypes, newUpperIndexTypes)
#                diagram.array = self.array[slices]
                diagram = SubDiagram(self, newLowerIndexTypes, newUpperIndexTypes, arraySlices=slices)
                diagrams.append(diagram)
        return diagrams

    def getAllDiagrams(self, vacuum):
        Nocc = sum(vacuum)
        Norbs = len(vacuum)
        diagrams = []
        lowerGeneralIndexCount = sum(i == 'g' for i in self.lowerIndexTypes)
        lowerSplits = list(itertools.product(['h', 'p'], repeat=lowerGeneralIndexCount))
        upperGeneralIndexCount = sum(i == 'g' for i in self.upperIndexTypes)
        upperSplits = list(itertools.product(['h', 'p'], repeat=upperGeneralIndexCount))
        for lowerSplit in lowerSplits:
            lowerSlices = [slice(None)] * self.excitationRank
            lowerSplitIndexTypes = list(lowerSplit)
            lGI = 0
            newLowerIndexTypes = copy(self.lowerIndexTypes)
            for lI in range(len(newLowerIndexTypes)):
                if newLowerIndexTypes[lI] == 'g':
                    newLI = lowerSplitIndexTypes[lGI]
                    if newLI == 'h':
                        lowerSlices[lI] = slice(None,Nocc)
                    elif newLI == 'p':
                        lowerSlices[lI] = slice(Nocc, None)
                    newLowerIndexTypes[lI] = newLI
                    lGI += 1
            for upperSplit in upperSplits:
#                print(lowerSplit, upperSplit)
                upperSlices = [slice(None)] * self.excitationRank
                upperSplitIndexTypes = list(upperSplit)
                uGI = 0
                newUpperIndexTypes = copy(self.upperIndexTypes)
                for uI in range(len(newUpperIndexTypes)):
                    if newUpperIndexTypes[uI] == 'g':
                        newUI = upperSplitIndexTypes[uGI]
                        if newUI == 'h':
                            upperSlices[uI] = slice(None,Nocc)
                        elif newUI == 'p':
                            upperSlices[uI] = slice(Nocc, None)
                        newUpperIndexTypes[uI] = newUI
                        uGI += 1
                slices = tuple(lowerSlices + upperSlices)
#                print(lowerSplitIndexTypes)
#                print(upperSplitIndexTypes)
#                print(newLowerIndexTypes)
#                print(newUpperIndexTypes)
#                print(slices)
#                diagram = Tensor(self.name, newLowerIndexTypes, newUpperIndexTypes)
#                diagram.array = self.array[slices]
                diagram = SubDiagram(self, newLowerIndexTypes, newUpperIndexTypes, arraySlices=slices)
                diagrams.append(diagram)
        return diagrams

    def getAllDiagramsGeneral(self):
        diagrams = []
        lowerGeneralIndexCount = sum(i == 'g' for i in self.lowerIndexTypes)
        lowerSplits = list(itertools.product(['h', 'p'], repeat=lowerGeneralIndexCount))
        upperGeneralIndexCount = sum(i == 'g' for i in self.upperIndexTypes)
        upperSplits = list(itertools.product(['h', 'p'], repeat=upperGeneralIndexCount))
        for lowerSplit in lowerSplits:
            lowerSplitIndexTypes = list(lowerSplit)
            lGI = 0
            newLowerIndexTypes = copy(self.lowerIndexTypes)
            for lI in range(len(newLowerIndexTypes)):
                if newLowerIndexTypes[lI] == 'g':
                    newLI = lowerSplitIndexTypes[lGI]
                    newLowerIndexTypes[lI] = newLI
                    lGI += 1
            for upperSplit in upperSplits:
#                print(lowerSplit, upperSplit)
                upperSplitIndexTypes = list(upperSplit)
                uGI = 0
                newUpperIndexTypes = copy(self.upperIndexTypes)
                for uI in range(len(newUpperIndexTypes)):
                    if newUpperIndexTypes[uI] == 'g':
                        newUI = upperSplitIndexTypes[uGI]
                        newUpperIndexTypes[uI] = newUI
                        uGI += 1
#                print(lowerSplitIndexTypes)
#                print(upperSplitIndexTypes)
#                print(newLowerIndexTypes)
#                print(newUpperIndexTypes)
#                diagram = Tensor(self.name, newLowerIndexTypes, newUpperIndexTypes)
                diagram = SubDiagram(self, newLowerIndexTypes, newUpperIndexTypes)
                diagrams.append(diagram)
        self.diagrams = diagrams

    def getAllDiagramsActive(self, active=True):
        diagrams = []
        if active:
            lowerGeneralIndexCount = sum(i == 'g' for i in self.lowerIndexTypes)
            lowerGSplits = list(itertools.product(['c', 'a', 'v'], repeat=lowerGeneralIndexCount))
            upperGeneralIndexCount = sum(i == 'g' for i in self.upperIndexTypes)
            upperGSplits = list(itertools.product(['c', 'a', 'v'], repeat=upperGeneralIndexCount))
            lowerParticleIndexCount = sum(i == 'p' for i in self.lowerIndexTypes)
            lowerPSplits = list(itertools.product(['a', 'v'], repeat=lowerParticleIndexCount))
            upperParticleIndexCount = sum(i == 'p' for i in self.upperIndexTypes)
            upperPSplits = list(itertools.product(['a', 'v'], repeat=upperParticleIndexCount))
        else:
            lowerGeneralIndexCount = sum(i == 'g' for i in self.lowerIndexTypes)
            lowerGSplits = list(itertools.product(['c', 'v'], repeat=lowerGeneralIndexCount))
            upperGeneralIndexCount = sum(i == 'g' for i in self.upperIndexTypes)
            upperGSplits = list(itertools.product(['c', 'v'], repeat=upperGeneralIndexCount))
            lowerParticleIndexCount = sum(i == 'p' for i in self.lowerIndexTypes)
            lowerPSplits = list(itertools.product(['v'], repeat=lowerParticleIndexCount))
            upperParticleIndexCount = sum(i == 'p' for i in self.upperIndexTypes)
            upperPSplits = list(itertools.product(['v'], repeat=upperParticleIndexCount))
        for lowerGSplit in lowerGSplits:
            lGI = 0
            lowerGSplitIndexTypes = list(lowerGSplit)
            for lowerPSplit in lowerPSplits:
                lPI = 0
                lowerPSplitIndexTypes = list(lowerPSplit)
                newLowerIndexTypes = copy(self.lowerIndexTypes)
                for lI in range(len(self.lowerIndexTypes)):
                    newLI = newLowerIndexTypes[lI]
                    if newLowerIndexTypes[lI] == 'g':
                        newLI = lowerGSplitIndexTypes[lGI]
                        lGI += 1
                    elif newLowerIndexTypes[lI] == 'p':
                        newLI = lowerPSplitIndexTypes[lPI]
                        lPI += 1
                    elif newLowerIndexTypes[lI] == 'h':
                        newLI = "c"
                    newLowerIndexTypes[lI] = newLI
                for upperGSplit in upperGSplits:
                    uGI = 0
                    upperGSplitIndexTypes = list(upperGSplit)
                    for upperPSplit in upperPSplits:
                        uPI = 0
                        upperPSplitIndexTypes = list(upperPSplit)
                        newUpperIndexTypes = copy(self.upperIndexTypes)
                        for uI in range(len(self.upperIndexTypes)):
                            newUI = newUpperIndexTypes[uI]
                            if newUpperIndexTypes[uI] == 'g':
                                newUI = upperGSplitIndexTypes[uGI]
                                uGI += 1
                            elif newUpperIndexTypes[uI] == 'p':
                                newUI = upperPSplitIndexTypes[uPI]
                                uPI += 1
                            elif newUpperIndexTypes[uI] == 'h':
                                newUI = 'c'
                            newUpperIndexTypes[uI] = newUI
#                        diagram = Tensor(self.name, newLowerIndexTypes, newUpperIndexTypes)
                        diagram = SubDiagram(self, newLowerIndexTypes, newUpperIndexTypes)
                        diagrams.append(diagram)
        self.diagrams = diagrams
    
    def assignDiagramArrays(self, vacuum):
        Nocc = sum(vacuum)
        for diagram in self.diagrams:
            lowerSlices = [slice(0, None)] * self.excitationRank
            upperSlices = [slice(0, None)] * self.excitationRank
            for lI, lowerIndex in enumerate(diagram.lowerIndexTypes):
                if lowerIndex == 'h':
                    lowerSlices[lI] = slice(0, Nocc)
                elif lowerIndex == 'p':
                    lowerSlices[lI] = slice(Nocc, None)
            for uI, upperIndex in enumerate(diagram.upperIndexTypes):
                if upperIndex == 'h':
                    upperSlices[uI] = slice(0, Nocc)
                elif upperIndex == 'p':
                    upperSlices[uI] = slice(Nocc, None)
            slices = tuple(lowerSlices + upperSlices)
    #        diagram.array = self.array[slices]
            # diagram.sliceArray(slices)
            diagram.setSlices(slices)
    
    def assignDiagramArraysActive(self, nCore, nActive, nVirtual):
        for diagram in self.diagrams:
            lowerSlices = [slice(None)] * self.excitationRank
            upperSlices = [slice(None)] * self.excitationRank
            for lI, lowerIndex in enumerate(diagram.lowerIndexTypes):
                if lowerIndex == 'c':
                    lowerSlices[lI] = slice(0, nCore)
                elif lowerIndex == 'a':
                    if self.lowerIndexTypes[lI] == 'p' or self.lowerIndexTypes[lI] == 'a':
                        lowerSlices[lI] = slice(0, nActive)
                    else:
                        lowerSlices[lI] = slice(nCore, nCore + nActive)
                elif lowerIndex == 'v':
                    if self.lowerIndexTypes[lI] == 'p':
                        lowerSlices[lI] = slice(nActive, None)
                    elif self.lowerIndexTypes[lI] == 'v':
                        lowerSlices[lI] = slice(0, None)
                    else:
                        lowerSlices[lI] = slice(nCore + nActive, None)
            for uI, upperIndex in enumerate(diagram.upperIndexTypes):
                if upperIndex == 'c':
                    upperSlices[uI] = slice(0, nCore)
                elif upperIndex == 'a':
                    if self.upperIndexTypes[uI] == 'p' or self.upperIndexTypes[uI] == 'a':
                        upperSlices[uI] = slice(0, nActive)
                    else:
                        upperSlices[uI] = slice(nCore, nCore + nActive)
                elif upperIndex == 'v':
                    if self.upperIndexTypes[uI] == 'p':
                        upperSlices[uI] = slice(nActive, None)
                    elif self.upperIndexTypes[uI] == 'v':
                        upperSlices[uI] = slice(0, None)
                    else:
                        upperSlices[uI] = slice(nCore + nActive, None)
            slices = tuple(lowerSlices + upperSlices)
#            diagram.array = self.array[slices]
            # diagram.sliceArray(slices)
            diagram.setSlices(slices)

    # def reassembleArray(self):
    #     for d, diagram in enumerate(self.diagrams):
    #         self.array[diagram.arraySlices] = diagram.array

    def conjugate(self):
        conjugate = Tensor(self.name, self.upperIndexTypes, self.lowerIndexTypes)
        try:
            conjugate.array = np.transpose(self.array, [*range(len(self.lowerIndexTypes), len(self.lowerIndexTypes) + len(self.upperIndexTypes))] + [*range(len(self.lowerIndexTypes))])
        except AttributeError:
            pass
        return conjugate

    def __add__(self, other):
        if isinstance(other, Tensor):
            return TensorSum([TensorProduct([self]), TensorProduct([other])])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return TensorProduct([self, other])
        elif isinstance(other, Number):
            return TensorProduct([self], other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return TensorProduct([self], other)
        else:
            return NotImplemented

    def __str__(self):
        string = self.name + "_{"
        for pType in self.lowerIndexTypes:
            string += pType
        string += "}^{"
        for qType in self.upperIndexTypes:
            string += qType
        string += "}"
        return string

class SubDiagram(Tensor):
    '''
    Sub-class for diagrams created when specifying orbital spaces of indices in a Tensor
    '''
    def __init__(self, parentTensor, lowerIndexTypesList, upperIndexTypesList):#, arraySlices=None):
        self.parentTensor = parentTensor
        self.name = self.parentTensor.name
        self.lowerIndexTypes = lowerIndexTypesList
        self.upperIndexTypes = upperIndexTypesList
        self.excitationRank = len(self.lowerIndexTypes)
        self.spinFree = self.parentTensor.spinFree
        self.distinguishableParticles = self.parentTensor.distinguishableParticles
        # if arraySlices is None:
        #     self.arraySlices = [slice(None)] * (len(self.lowerIndexTypes) + len(self.upperIndexTypes))
        # else:
        #     self.arraySlices = arraySlices
#        self.indexRangeStartPoints = [startPoint + self.arraySlices[sP].start for sP, startPoint in enumerate(self.parentTensor.indexRangeStartPoints)]
        # if arraySlices is not None:
        #     self.arraySlices = arraySlices
        #     # try:
        #     #     self.array = parentTensor.array[self.arraySlices]
        #     # except AttributeError:
        #     #     pass

    # def sliceArray(self, arraySlices):
    #     self.arraySlices = arraySlices
    #     # try:
    #     #     self.array = self.parentTensor.array[self.arraySlices]
    #     # except AttributeError:
    #     #     print(self.name, "no array found")
    #     #     pass

    def setSlices(self, slices):
        self.arraySlices = slices
        self.indexRangeStartPoints = [startPoint + self.arraySlices[sP].start for sP, startPoint in enumerate(self.parentTensor.indexRangeStartPoints)]

    def getArray(self):
        return self.parentTensor.array[self.arraySlices]

    def setArray(self, array):
        try:
            assert array.shape == self.parentTensor.array[self.arraySlices].shape
            self.parentTensor.array[self.arraySlices] = array
        except AssertionError:
            print("Sub-array is of wrong shape:", array.shape, self.parentTensor, self.parentTensor.array.shape, self.arraySlices, self.parentTensor.array[self.arraySlices].shape)
        

class normalOrderedProduct(Tensor):
    '''
    Sub-class for tensor formed as normal ordered product of other tensors
    '''
    def __init__(self, tensorProduct):
        #, prefactor=1.0, vertexList=None, normalOrderedSlices=[], contractionsList=[]):
        self.tensorProduct = tensorProduct
        assert all(tensor.spinFree for t, tensor in enumerate(self.tensorProduct.tensorList)) or all(not tensor.spinFree for t, tensor in enumerate(self.tensorProduct.tensorList))
        name = "{}" if len(self.tensorProduct.tensorList) == 0 else "{" + "".join(tensor.name for t, tensor in enumerate(self.tensorProduct.tensorList)) + "}"
        lowerIndexTypes = [] if len(self.tensorProduct.tensorList) == 0 else sum([tensor.lowerIndexTypes for t, tensor in enumerate(self.tensorProduct.tensorList)], [])
        upperIndexTypes = [] if len(self.tensorProduct.tensorList) == 0 else sum([tensor.upperIndexTypes for t, tensor in enumerate(self.tensorProduct.tensorList)], [])
        spinFree = True if len(self.tensorProduct.tensorList) == 0 else self.tensorProduct.tensorList[0].spinFree
        super().__init__(name, lowerIndexTypes, upperIndexTypes, spinFree)
        # self.array = None

    def calculateArray(self):
        if len(self.tensorProduct.tensorList) == 0:
            self.array = np.array(self.tensorProduct.prefactor)
        else:
            self.array = contractions.getContractedArray(self.tensorProduct, [])[0]

class Vertex:
    '''
    Class for amplitude tensors of spin-free excitation operators
    '''
    def __init__(self, tensor, lowerIndicesList, upperIndicesList):
        self.name = tensor.name
        self.tensor = tensor
        self.lowerIndices = lowerIndicesList
        self.upperIndices = upperIndicesList
        self.excitationRank = len(self.lowerIndices)

    # def applyContraction(self, contraction):
    #     contractionApplied = False
    #     if isinstance(contraction[0], index.SpecificOrbitalIndex):
    #         for uI, upperIndex in enumerate(self.upperIndices):
    #             if upperIndex == contraction[1]:
    #                 self.upperIndices[uI] = contraction[0].contractedCopy(contraction[1])
    #                 contractionApplied = True
    #                 # upperIndex = copy(contraction[0])
    #                 # upperIndex.contractedFrom = contraction[1]
    #                 break
    #     elif isinstance(contraction[1], index.SpecificOrbitalIndex):
    #         for lI, lowerIndex in enumerate(self.lowerIndices):
    #             if lowerIndex == contraction[0]:
    #                 self.lowerIndices[lI] = contraction[1].contractedCopy(contraction[0])
    #                 contractionApplied = True
    #                 # lowerIndex = contraction[1]
    #                 # lowerIndex.contractedFrom = contraction[0]
    #                 break
    #     else:
    #         for lI, lowerIndex in enumerate(self.lowerIndices):
    #             if lowerIndex == contraction[0]:
    #                 self.lowerIndices[lI] = contraction[1]
    #                 contractionApplied = True
    #                 break
    #         if not contractionApplied:
    #             for uI, upperIndex in enumerate(self.upperIndices):
    #                 if upperIndex == contraction[1]:
    #                     self.upperIndices[uI] = contraction[0]
    #                     contractionApplied = True
    #                     break
    #     return contractionApplied

    def getOperator(self, normalOrdered=True):
        if normalOrdered:
            if self.tensor.spinFree:
                return operator.normalOrder(operator.spinFreeExcitation(self.lowerIndices, self.upperIndices))
            else:
                return operator.normalOrder(operator.operatorSum([operator.excitation(self.lowerIndices, self.upperIndices, [True] * (2 * self.excitationRank))]))
        else:
            if self.tensor.spinFree:
                return operator.spinFreeExcitation(self.lowerIndices, self.upperIndices)
            else:
                return operator.operatorSum([operator.excitation(self.lowerIndices, self.upperIndices, [True] * (2 * self.excitationRank))])

    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.name == other.name and self.tensor == other.tensor and self.lowerIndices == other.lowerIndices and self.upperIndices == other.upperIndices
        else:
            return NotImplemented

    def __copy__(self):
        return Vertex(self.tensor, copy(self.lowerIndices), copy(self.upperIndices))

    def __str__(self):
        string = self.name + "_{"
        for p in self.lowerIndices:
            string += p.__str__()
        string += "}^{"
        for q in self.upperIndices:
            string += q.__str__()
        string += "}"
        return string

class node:
    def __init__(self, annihilationIndex, creationIndex):
        self.inIndex = annihilationIndex
        self.outIndex = creationIndex
        self.inContracted = False
        self.outContracted = False
        # self.whichInParticle = None
        # self.whichOutParticle = None

class TensorProduct:
    def __init__(self, tensorList, prefactor=1., vertexList=None, normalOrderedSlices=[], contractionsList=[], freeLowerIndices=None, freeUpperIndices=None):
        self.tensorList = tensorList
        self.lowerIndices = {'g':[], 'v':[], 'c':[], 'a':[], 'h':[], 'p':[]}
        self.upperIndices = {'g':[], 'v':[], 'c':[], 'a':[], 'h':[], 'p':[]}
        self.prefactor = prefactor
        self.vertexList = vertexList
        if self.vertexList is None:
            self.vertexList = self.getVertexList(tensorList)
        self.normalOrderedSlices = normalOrderedSlices
        self.contractionsList = contractionsList
        lowerIndices = [lI for vertex in self.vertexList for lI in vertex.lowerIndices] + [contraction[1] for c, contraction in enumerate(self.contractionsList)]
        upperIndices = [uI for vertex in self.vertexList for uI in vertex.upperIndices] + [contraction[0] for c, contraction in enumerate(self.contractionsList)]
        if freeLowerIndices is None:
            self.freeLowerIndices = [lowerIndex for lI, lowerIndex in enumerate(lowerIndices) if lowerIndex not in upperIndices]
        else:
            self.freeLowerIndices = freeLowerIndices
        if freeUpperIndices is None:
            self.freeUpperIndices = [upperIndex for uI, upperIndex in enumerate(upperIndices) if upperIndex not in lowerIndices]
        else:
            self.freeUpperIndices = freeUpperIndices
        if len(self.freeLowerIndices) == len(self.freeUpperIndices):
            self.freeIndexNodes = [node(lowerIndex, self.freeUpperIndices[lI]) for lI, lowerIndex in enumerate(self.freeLowerIndices)]
#        self.normalOrderedSliceStartPoints = [*range(len(self.tensorList))]
#        for s, slice in enumerate(normalOrderedSlices)
            

    def applyContraction(self, contraction):
        # contractionApplied = False
        # for vertex1 in self.vertexList:
        #     for vertex2 in self.vertexList:
        #         if vertex1.applyContraction(contraction, vertex2):
        #             contractionApplied = True
        #             break
        #     if contractionApplied:
        #         break
        # for vertex in reversed(self.vertexList):
        #     if vertex.applyContraction(contraction):
        #         break
        self.contractionsList.append(contraction)
        if isinstance(contraction[0], index.SpecificOrbitalIndex) or isinstance(contraction[1], index.SpecificOrbitalIndex):
            return
        if contraction[0] not in self.freeLowerIndices:
            print(self)
            print(self.freeLowerIndices)
            print(contraction[0])
        if contraction[1] not in self.freeUpperIndices:
            print(self)
            print(self.freeUpperIndices)
            print(contraction[1])
        lI = self.freeLowerIndices.index(contraction[0])
        uI = self.freeUpperIndices.index(contraction[1])
        correspondingLowerIndex = self.freeLowerIndices.pop(uI)
        correspondingUpperIndex = self.freeUpperIndices.pop(lI)
        removedfreeLowerIndexNode = self.freeIndexNodes[lI]
        if uI != lI:
            self.freeLowerIndices.remove(contraction[0])
            self.freeUpperIndices.remove(contraction[1])
            removedfreeUpperIndexNode = self.freeIndexNodes.pop(uI)
            self.freeLowerIndices.append(correspondingLowerIndex)
            self.freeUpperIndices.append(correspondingUpperIndex)
            self.freeIndexNodes.append(node(correspondingLowerIndex, correspondingUpperIndex))
        self.freeIndexNodes.remove(removedfreeLowerIndexNode)
        # newFreeLowerIndices = self.freeLowerIndices[:lI] + self.freeLowerIndices[lI+1:]
        # newFreeUpperIndices = self.freeUpperIndices[:lI] + self.freeUpperIndices[lI+1:]
#        for fLI, freeLowerIndex in enumerate(self.freeLowerIndices):

        # if contraction[0] in self.freeLowerIndices:
        #     self.freeLowerIndices.remove(contraction[0])
        # if contraction[1] in self.freeUpperIndices:
        #     self.freeUpperIndices.remove(contraction[1])
        for orbitalType in self.lowerIndices:
            if contraction[0] in self.lowerIndices[orbitalType]:
                self.lowerIndices[orbitalType].remove(contraction[0])
        for orbitalType in self.upperIndices:
            if contraction[1] in self.upperIndices[orbitalType]:
                self.upperIndices[orbitalType].remove(contraction[1])

    def addNewIndex(self, orbitalType, lowerBool):
        count = len(self.lowerIndices[orbitalType]) + len(self.upperIndices[orbitalType])
        newIndexName = orbitalType + "_{" + str(count) + "}"
    #    newIndex = Index(newIndexName, False, False)
        if orbitalType == "c" or orbitalType == "h":
            newIndex = index.Index(newIndexName, True, False)
    #        newIndex.occupiedInVacuum = True
        elif orbitalType == "a":
            newIndex = index.Index(newIndexName, False, True)
    #        newIndex.active = True
        elif orbitalType == "p" or orbitalType == "v":
            newIndex = index.Index(newIndexName, False, False)
            pass
        else:
            newIndex = index.Index(newIndexName, False, False)
            print("orbital type not recognised; assuming virtual by default")
        if lowerBool:
            self.lowerIndices[orbitalType].append(newIndex)
        else:
            self.upperIndices[orbitalType].append(newIndex)
        return newIndex

    def getVertexList(self, tensorList_):
        vertexList = []
        for t in tensorList_:
            lowerIndexList = []
            for i in t.lowerIndexTypes:
                lowerIndexList.append(self.addNewIndex(i, True))
            upperIndexList = []
            for i in t.upperIndexTypes:
                upperIndexList.append(self.addNewIndex(i, False))
            vertexList.append(Vertex(t, lowerIndexList, upperIndexList))
        return vertexList

    def getOperator(self, normalOrderedParts=True):
        op = operator.OperatorProduct([], self.prefactor)
#        for v, vertex in enumerate(self.vertexList):
#            inSlice = False
#            for s, slice in enumerate(self.normalOrderedSlices):
#                if v == slice.start:
#                if v in range(len(self.vertexList)[slice]):
#            if any([v in range(len(self.vertexList)[slice]) for slice in self.normalOrderedSlices])
        vertexOperators = [vertex.getOperator(normalOrderedParts) for v, vertex in enumerate(self.vertexList)]
        for v, vertex in enumerate(self.vertexList):
            inSlice = False
            for s, NOSlice in enumerate(self.normalOrderedSlices):
                if v == NOSlice.start:
                    sliceOperator = np.product(vertexOperators[NOSlice])
                    op = op * operator.normalOrder(sliceOperator)
                    inSlice = True
                elif v in range(*NOSlice.indices(len(self.vertexList))):
                    inSlice = True
            if not inSlice:
                op = op * vertexOperators[v]
#        for slice in self.normalOrderedSlices:
#            operator = operator * 
#        if self.normalOrdered:
#            return normalOrder(operator)
        op.contractionsList = self.contractionsList
        return op

    def getVacuumExpectationValue(self, normalOrderedParts=True):
        return contractions.vacuumExpectationValue(self.getOperator(normalOrderedParts))

    def getGraph(self, active=False):
        graph = nx.DiGraph()
        for vertex in self.vertexList:
            vertex.nodes = []
            for i in range(vertex.excitationRank):
                vertex.nodes.append(node(vertex.upperIndices[i], vertex.lowerIndices[i]))
        # graph.add_nodes_from(self.freeIndexNodes, vertex=0, freeOutType="", freeInType="", tensorName='\\Phi')
        # graph.add_edges_from(itertools.combinations(self.freeIndexNodes, 2), connection="interaction")
        # print(*self.vertexList)
        # print("free", *self.freeIndexNodes)
        graph.add_nodes_from([(node_, {"whichParticle": n}) for n, node_ in enumerate(self.freeIndexNodes)], vertex=len(self.vertexList), tensorName="free")
        for v1, vertex1 in enumerate(self.vertexList):
            # graph.add_edges_from(itertools.permutations(vertex1.nodes, 2), connection="interaction")
#            graph.add_nodes_from(vertex1.nodes, vertex=v1+1, freeOutType="", freeInType="", tensorName=vertex1.tensor.name, whichInParticle=None, whichOutParticle=None)
            # print(v1, vertex1.nodes)
            # print(vertex1.tensor.name)
            # print(graph.number_of_nodes())
            # if vertex1.tensor.name[:5] == '{\Phi':
            if vertex1.tensor.distinguishableParticles:
                graph.add_nodes_from([(node_, {"whichParticle": n}) for n, node_ in enumerate(vertex1.nodes)], vertex=v1, tensorName=vertex1.tensor.name)
                graph.add_edges_from(itertools.combinations(vertex1.nodes, 2), connection="interaction", specificActiveIndexValue=None)
                # print("distinguishable particles", vertex1)
            else:
                graph.add_nodes_from([(node_, {"whichParticle": None}) for n, node_ in enumerate(vertex1.nodes)], vertex=v1, tensorName=vertex1.tensor.name)
                graph.add_edges_from(itertools.permutations(vertex1.nodes, 2), connection="interaction", specificActiveIndexValue=None)
        for v1, vertex1 in enumerate(self.vertexList):
            for node1 in vertex1.nodes:
                curIndex = node1.outIndex
                specificActiveIndexValue = None
                if isinstance(curIndex, index.SpecificOrbitalIndex):
                    specificActiveIndexValue = curIndex.value
                keepTracing = True
                while keepTracing:
                    # print(curIndex)
                    keepTracing = False
                    for c, contraction in enumerate(self.contractionsList):
                        if curIndex == contraction[0]:
                            curIndex = contraction[1]
                            if isinstance(curIndex, index.SpecificOrbitalIndex):
                                specificActiveIndexValue = curIndex.value
                            keepTracing = True
                for v2, vertex2 in enumerate(self.vertexList):
                    for node2 in vertex2.nodes:
                        if node2.inIndex == curIndex:
                            node1.outContracted = True
                            node2.inContracted = True
                            if node1.outIndex.active or node2.inIndex.active:
                                graph.add_edge(node1, node2, connection="active propagation", specificActiveIndexValue=specificActiveIndexValue)
                                # if isinstance(node1.outIndex, index.SpecificOrbitalIndex):
                                #     graph.add_edge(node1, node2, connection="active projection", specificActiveIndexValue=node1.outIndex.value)
                                # elif isinstance(node2.inIndex, index.SpecificOrbitalIndex):
                                #     graph.add_edge(node1, node2, connection="active projection", specificActiveIndexValue=node2.inIndex.value)
                                # else:
                                #     graph.add_edge(node1, node2, connection="active projection", specificActiveIndexValue=None)
                            else:
                                graph.add_edge(node1, node2, connection="inactive propagation", specificActiveIndexValue=None)
                if not node1.outContracted:
                    for fIN, freeIndexNode in enumerate(self.freeIndexNodes):
                        # print(fIN, freeIndexNode, freeIndexNode.inIndex, curIndex)
                        if freeIndexNode.inIndex == curIndex:
                            # print("project")
                            node1.outContracted = True
                            freeIndexNode.inContracted = True
                            if node1.outIndex.active or freeIndexNode.inIndex.active:
                                graph.add_edge(node1, freeIndexNode, connection="active projection", specificActiveIndexValue=specificActiveIndexValue)
                                # if isinstance(node1.outIndex, index.SpecificOrbitalIndex):
                                #     graph.add_edge(node1, freeIndexNode, connection="active projection", specificActiveIndexValue=node1.outIndex.value)
                                # elif isinstance(freeIndexNode.inIndex, index.SpecificOrbitalIndex):
                                #     graph.add_edge(node1, freeIndexNode, connection="active projection", specificActiveIndexValue=freeIndexNode.inIndex.value)
                                # else:
                                #     graph.add_edge(node1, freeIndexNode, connection="active projection", specificActiveIndexValue=None)
                            else:
                                graph.add_edge(node1, freeIndexNode, connection="inactive projection", specificActiveIndexValue=None)
                            curIndex = freeIndexNode.outIndex
                            specificActiveIndexValue = None
                            keepTracing = True
                            while keepTracing:
                                keepTracing = False
                                for c, contraction in enumerate(self.contractionsList):
                                    if curIndex == contraction[0]:
                                        curIndex = contraction[1]
                                        if isinstance(curIndex, index.SpecificOrbitalIndex):
                                            specificActiveIndexValue = curIndex.value
                                        keepTracing = True
                            for v2, vertex2 in enumerate(self.vertexList):
                                for node2 in vertex2.nodes:
                                    if node2.inIndex == curIndex:
                                        # for fIN, freeIndexNode in enumerate(self.freeIndexNodes):
                                    # print("project reverse")
                                        freeIndexNode.outContracted = True
                                        node2.inContracted = True
                                        if freeIndexNode.outIndex.active or node2.inIndex.active:
                                            graph.add_edge(freeIndexNode, node2, connection="active projection", specificActiveIndexValue=specificActiveIndexValue)
                                        # if isinstance(freeIndexNode.outIndex, index.SpecificOrbitalIndex):
                                        #     graph.add_edge(freeIndexNode, node2, connection="active projection", specificActiveIndexValue=freeIndexNode.outIndex.value)
                                        # elif isinstance(node2.inIndex, index.SpecificOrbitalIndex):
                                        #     graph.add_edge(freeIndexNode, node2, connection="active projection", specificActiveIndexValue=node2.inIndex.value)
                                        # else:
                                        #     graph.add_edge(freeIndexNode, node2, connection="active projection", specificActiveIndexValue=None)
                                        else:
                                            graph.add_edge(freeIndexNode, node2, connection="inactive projection", specificActiveIndexValue=None)
#         for uI, upperIndex in enumerate(self.freeUpperIndices):
#             curIndex = upperIndex
#             keepTracing = True
#             while keepTracing:
#                 keepTracing = False
#                 for c, contraction in enumerate(self.contractionsList):
#                     if curIndex == contraction[0]:
#                         keepTracing = True
#                         curIndex = contraction[1]            
#                 for v, vertex in enumerate(self.vertexList):
#                     for n in vertex.nodes:
#                         if curIndex == n.inIndex:
#                             keepTracing = True
#                             graph.nodes[n]["whichInParticle"] = uI
#                             curIndex = n.outIndex
#         for lI, lowerIndex in enumerate(self.freeLowerIndices):
#             curIndex = lowerIndex
#             keepTracing = True
#             while keepTracing:
#                 keepTracing = False
#                 for c, contraction in enumerate(self.contractionsList):
#                     if curIndex == contraction[1]:
#                         keepTracing = True
#                         curIndex = contraction[0]            
#                 for v, vertex in enumerate(self.vertexList):
#                     for n in vertex.nodes:
#                         if curIndex == n.outIndex:
#                             keepTracing = True
#                             graph.nodes[n]["whichOutParticle"] = lI
#                             curIndex = n.inIndex
# #            print("pair", lowerIndex, curIndex)
#         for v1, vertex1 in enumerate(self.vertexList):
#             if active:
#                 for node1 in vertex1.nodes:
#                     if not node1.outContracted:
# #                        print("free lower indices", *self.freeLowerIndices, "check Index", node1.outIndex)
# #                        graph.nodes[node1]["whichOutParticle"] = self.freeLowerIndices.index(node1.outIndex)
#                         if node1.outIndex.occupiedInVacuum:
#                             graph.nodes[node1]["freeOutType"] = "c"
#                         elif node1.outIndex.active:
#                             graph.nodes[node1]["freeOutType"] = "a"
#                         else:
#                             graph.nodes[node1]["freeOutType"] = "v"
#                     else:
#                         graph.nodes[node1]["whichOutParticle"] = None
#                     if not node1.inContracted:
# #                        print("free upper indices", *self.freeUpperIndices, "check Index", node1.inIndex)
# #                        graph.nodes[node1]["whichInParticle"] = self.freeUpperIndices.index(node1.inIndex)
#                         if node1.inIndex.occupiedInVacuum:
#                             graph.nodes[node1]["freeInType"] = "c"
#                         elif node1.inIndex.active:
#                             graph.nodes[node1]["freeInType"] = "a"
#                         else:
#                             graph.nodes[node1]["freeInType"] = "v"
#                     else:
#                         graph.nodes[node1]["whichInParticle"] = None
#             else:
#                 for node1 in vertex1.nodes:
#                     if not node1.outContracted:
#                         if node1.outIndex.occupiedInVacuum:
#                             graph.nodes[node1]["freeOutType"] = "h"
#                         else:
#                             graph.nodes[node1]["freeOutType"] = "p"
#                     else:
#                         graph.nodes[node1]["whichOutParticle"] = None
#                     if not node1.inContracted:
#                         if node1.inIndex.occupiedInVacuum:
#                             graph.nodes[node1]["freeInType"] = "h"
#                         else:
#                             graph.nodes[node1]["freeInType"] = "p"
#                     else:
#                         graph.nodes[node1]["whichInParticle"] = None
        # print("nNodes", graph.number_of_nodes())
        return graph

    def getGraphOld(self, active=False):
        graph = nx.DiGraph()
        for vertex in self.vertexList:
            vertex.nodes = []
            for i in range(vertex.excitationRank):
                vertex.nodes.append(node(vertex.upperIndices[i], vertex.lowerIndices[i]))
        for v1, vertex1 in enumerate(self.vertexList):
            graph.add_nodes_from(vertex1.nodes, vertex=v1, freeOutType="", freeInType="", tensorName=vertex1.tensor.name, whichInParticle=None, whichOutParticle=None)
            if vertex1.tensor.name == '\\Phi':
                graph.add_edges_from(itertools.combinations(vertex1.nodes, 2), connection="interaction")
            else:
                graph.add_edges_from(itertools.permutations(vertex1.nodes, 2), connection="interaction")
        for v1, vertex1 in enumerate(self.vertexList):
            for node1 in vertex1.nodes:
                curIndex = node1.outIndex
                keepTracing = True
                while keepTracing:
                    keepTracing = False
                    for c, contraction in enumerate(self.contractionsList):
                        if curIndex == contraction[0]:
                            keepTracing = True
                            curIndex = contraction[1]
                # for c, contraction in enumerate(self.contractionsList):
                #     if node1.outIndex == contraction[0]:
                #         node1.outContracted = True
                #     if node1.inIndex == contraction[1]:
                #         node1.inContracted = True
                for v2, vertex2 in enumerate(self.vertexList):
                    for node2 in vertex2.nodes:
                        if node2.inIndex == curIndex:
                            node1.outContracted = True
                            node2.inContracted = True
                            if node1.outIndex.active or node2.inIndex.active:
                                graph.add_edge(node1, node2, connection="active propagation")
                            else:
                                graph.add_edge(node1, node2, connection="inactive propagation")
        for uI, upperIndex in enumerate(self.freeUpperIndices):
            curIndex = upperIndex
            keepTracing = True
            while keepTracing:
                keepTracing = False
                for c, contraction in enumerate(self.contractionsList):
                    if curIndex == contraction[0]:
                        keepTracing = True
                        curIndex = contraction[1]            
                for v, vertex in enumerate(self.vertexList):
                    for n in vertex.nodes:
                        if curIndex == n.inIndex:
                            keepTracing = True
                            graph.nodes[n]["whichInParticle"] = uI
                            curIndex = n.outIndex
        for lI, lowerIndex in enumerate(self.freeLowerIndices):
            curIndex = lowerIndex
            keepTracing = True
            while keepTracing:
                keepTracing = False
                for c, contraction in enumerate(self.contractionsList):
                    if curIndex == contraction[1]:
                        keepTracing = True
                        curIndex = contraction[0]            
                for v, vertex in enumerate(self.vertexList):
                    for n in vertex.nodes:
                        if curIndex == n.outIndex:
                            keepTracing = True
                            graph.nodes[n]["whichOutParticle"] = lI
                            curIndex = n.inIndex
#            print("pair", lowerIndex, curIndex)
        for v1, vertex1 in enumerate(self.vertexList):
            if active:
                for node1 in vertex1.nodes:
                    if not node1.outContracted:
#                        print("free lower indices", *self.freeLowerIndices, "check Index", node1.outIndex)
#                        graph.nodes[node1]["whichOutParticle"] = self.freeLowerIndices.index(node1.outIndex)
                        if node1.outIndex.occupiedInVacuum:
                            graph.nodes[node1]["freeOutType"] = "c"
                        elif node1.outIndex.active:
                            graph.nodes[node1]["freeOutType"] = "a"
                        else:
                            graph.nodes[node1]["freeOutType"] = "v"
                    else:
                        graph.nodes[node1]["whichOutParticle"] = None
                    if not node1.inContracted:
#                        print("free upper indices", *self.freeUpperIndices, "check Index", node1.inIndex)
#                        graph.nodes[node1]["whichInParticle"] = self.freeUpperIndices.index(node1.inIndex)
                        if node1.inIndex.occupiedInVacuum:
                            graph.nodes[node1]["freeInType"] = "c"
                        elif node1.inIndex.active:
                            graph.nodes[node1]["freeInType"] = "a"
                        else:
                            graph.nodes[node1]["freeInType"] = "v"
                    else:
                        graph.nodes[node1]["whichInParticle"] = None
            else:
                for node1 in vertex1.nodes:
                    if not node1.outContracted:
#                        print(self, "free lower indices", *self.freeLowerIndices, "check Index", node1.outIndex)
#                        graph.nodes[node1]["whichOutParticle"] = self.freeLowerIndices.index(node1.outIndex)
                        if node1.outIndex.occupiedInVacuum:
                            graph.nodes[node1]["freeOutType"] = "h"
                        else:
                            graph.nodes[node1]["freeOutType"] = "p"
                    else:
                        graph.nodes[node1]["whichOutParticle"] = None
                    if not node1.inContracted:
#                        print(self, "free upper indices", *self.freeUpperIndices, "check Index", node1.inIndex)
#                        graph.nodes[node1]["whichInParticle"] = self.freeUpperIndices.index(node1.inIndex)
                        if node1.inIndex.occupiedInVacuum:
                            graph.nodes[node1]["freeInType"] = "h"
                        else:
                            graph.nodes[node1]["freeInType"] = "p"
                    else:
                        graph.nodes[node1]["whichInParticle"] = None
            # for v2, vertex2 in enumerate(self.vertexList):
            #     for node1 in vertex1.nodes:
            #         for node2 in vertex2.nodes:
            #             if node1.outIndex == node2.inIndex:
            #                 node1.outContracted = True
            #                 node2.inContracted = True
            #                 if node1.outIndex.active:
            #                     graph.add_edge(node1, node2, connection="active propagation")
            #                 else:
            #                     graph.add_edge(node1, node2, connection="inactive propagation")
            # if active:
            #         if not node1.outContracted:
            #             if node1.outIndex.occupiedInVacuum:
            #                 graph.nodes[node1]["freeOutType"]="c"
            #             elif node1.outIndex.active:
            #                 graph.nodes[node1]["freeOutType"]="a"
            #             else:
            #                 graph.nodes[node1]["freeOutType"]="v"
            #         if not node1.inContracted:
            #             if node1.inIndex.occupiedInVacuum:
            #                 graph.nodes[node1]["freeInType"]="c"
            #             elif node1.inIndex.active:
            #                 graph.nodes[node1]["freeInType"]="a"
            #             else:
            #                 graph.nodes[node1]["freeInType"]="v"
            # else:
            #     for node1 in vertex1.nodes:
            #         if not node1.outContracted:
            #             if node1.outIndex.occupiedInVacuum:
            #                 graph.nodes[node1]["freeOutType"]="h"
            #             else:
            #                 graph.nodes[node1]["freeOutType"]="p"
            #         if not node1.inContracted:
            #             if node1.inIndex.occupiedInVacuum:
            #                 graph.nodes[node1]["freeInType"]="h"
            #             else:
            #                 graph.nodes[node1]["freeInType"]="p"
        return graph

    def drawGraph(self, active=False):
        graph = self.getGraph(active)
        nx.draw(graph, nx.multipartite_layout(graph, "vertex", "horizontal", scale=-1))

    def nodeMatch(self, node1, node2):
        # print(node1, node2)
        return node1["tensorName"] == node2["tensorName"] and node1["whichParticle"] == node2["whichParticle"] # and node1["freeInType"] == node2["freeInType"] and node1["freeOutType"] == node2["freeOutType"]# and ((node1["whichInParticle"] is None and node2["whichInParticle"] is None) or ((node1["whichInParticle"] is not None and node2["whichInParticle"] is not None) and (node1["whichInParticle"] == node2["whichInParticle"]))) and ((node1["whichOutParticle"] is None and node2["whichOutParticle"] is None) or ((node1["whichOutParticle"] is not None and node2["whichOutParticle"] is not None) and (node1["whichOutParticle"] == node2["whichOutParticle"])))

    def edgeMatch(self, edge1, edge2):
        return edge1["connection"] == edge2["connection"] and edge1["specificActiveIndexValue"] == edge2["specificActiveIndexValue"]

    def isProportional(self, other, active=False):
        selfGraph = self.getGraph(active)
        # print("nNodesInSelfGraph", selfGraph.number_of_nodes())
        otherGraph = other.getGraph(active)
        # print("nNodesInOtherGraph", otherGraph.number_of_nodes())
        # for node in selfGraph.nodes:
        #     print(node.inIndex, node.outIndex)
        # for node in otherGraph.nodes:
        #     print(node.inIndex, node.outIndex)
        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(selfGraph, otherGraph, self.nodeMatch, self.edgeMatch)
#        print(self.getFreeIndexPairs(selfGraph), self.getFreeIndexPairs(otherGraph))
#        return sorted([t.name for t in self.tensorList]) == sorted([t.name for t in other.tensorList]) and DiGM.is_isomorphic() and self.getFreeIndexPairs(selfGraph) == self.getFreeIndexPairs(otherGraph)
#        print(DiGM.is_isomorphic())
        indicesMatch = True
        for fLI, freeLowerIndex in enumerate(self.freeLowerIndices):
            if freeLowerIndex not in other.freeLowerIndices:
                indicesMatch = False
            # print(freeLowerIndex, self.freeUpperIndices[fLI], other.freeLowerIndices.index(freeLowerIndex), other.freeUpperIndices[other.freeLowerIndices.index(freeLowerIndex)])
            elif other.freeUpperIndices[other.freeLowerIndices.index(freeLowerIndex)] != self.freeUpperIndices[fLI]:
                indicesMatch = False
        return DiGM.is_isomorphic() and indicesMatch

    def followPropagation(self, graph, node):
        currentNode = node
        while currentNode.outContracted:
            for nbr, datadict in graph.adj[node].items():
                if datadict["connection"] == "active propagation" or datadict["connection"] == "inactive propagation":
                    currentNode = nbr
        return currentNode

    def getFreeIndexPairs(self, graph):
        freeIndexPairsDict = {}
        for startNode in graph.nodes:
            if not startNode.inContracted:
                inIndex = startNode.inIndex
                endNode = self.followPropagation(graph, startNode)
                outIndex = endNode.outIndex
#                print(inIndex, outIndex)
                freeIndexPairsDict[startNode.inIndex] = endNode.outIndex
        return freeIndexPairsDict

    def isProportional1(self, other):
        selfGraph = self.getGraph()
        otherGraph = other.getGraph()
        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(selfGraph, otherGraph)
        selfFreeIndexPairs = self.getFreeIndexPairs(selfGraph)
        otherFreeIndexPairs = other.getFreeIndexPairs(otherGraph)
        return (self.tensorList == other.tensorList) and DiGM.is_isomorphic() and all([DiGM.semantic_feasibility(DiGM.mapping[n], n) for n in DiGM.mapping.keys()]) and all([DiGM.mapping[selfFreeIndexPairs[startNode]] == otherFreeIndexPairs[DiGM.mapping[startNode]] for startNode in selfFreeIndexPairs.keys()]) and len(selfFreeIndexPairs) == len(otherFreeIndexPairs)

    def isConnected(self, active=False):
        selfGraph = self.getGraph(active)
        return nx.is_weakly_connected(selfGraph)

    def __eq__(self, other):
        if isinstance(other, TensorProduct):
            return self.isProportional(other) and self.prefactor == other.prefactor
        else:
            return NotImplemented

    def __copy__(self):
        return TensorProduct(copy(self.tensorList), copy(self.prefactor), [copy(vertex) for vertex in self.vertexList], copy(self.normalOrderedSlices), copy(self.contractionsList), copy(self.freeLowerIndices), copy(self.freeUpperIndices))

    def __add__(self, other):
        if isinstance(other, TensorProduct):
            return TensorSum([self, other])
        elif isinstance(other, Tensor):
            return TensorSum([self, TensorProduct([other])])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Tensor):
            return TensorSum([TensorProduct([other]), self])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return TensorProduct(self.tensorList + [other], self.prefactor, normalOrderedSlices=self.normalOrderedSlices, contractionsList=self.contractionsList)
        elif isinstance(other, TensorProduct):
            return TensorProduct(self.tensorList + other.tensorList, self.prefactor * other.prefactor, normalOrderedSlices=self.normalOrderedSlices+[slice(NOSlice.start+len(self.tensorList), NOSlice.stop+len(self.tensorList)) for s, NOSlice in enumerate(other.normalOrderedSlices)], contractionsList=self.contractionsList+other.contractionsList)
        elif isinstance(other, Number):
            return TensorProduct(self.tensorList, self.prefactor * other, self.vertexList, normalOrderedSlices=self.normalOrderedSlices, contractionsList=self.contractionsList, freeLowerIndices=self.freeLowerIndices, freeUpperIndices=self.freeUpperIndices)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Tensor):
            if self.contractionsList != []:
                print("New contracted indices not assigned")
                return NotImplemented
            # newVertexList = self.getVertexList([other] + self.tensorList)
            # newVertexList = newVertexList[1:] + [newVertexList[0]]
            # return TensorProduct([other] + self.tensorList, self.prefactor, vertexList=newVertexList, normalOrderedSlices=[slice(NOSlice.start+1, NOSlice.stop+1) for s, NOSlice in enumerate(self.normalOrderedSlices)], contractionsList=self.contractionsList)
            return TensorProduct([other] + self.tensorList, self.prefactor, normalOrderedSlices=[slice(NOSlice.start+1, NOSlice.stop+1) for s, NOSlice in enumerate(self.normalOrderedSlices)], contractionsList=self.contractionsList)
        elif isinstance(other, Number):
            return TensorProduct(self.tensorList, other * self.prefactor, self.vertexList, normalOrderedSlices=self.normalOrderedSlices, contractionsList=self.contractionsList, freeLowerIndices=self.freeLowerIndices, freeUpperIndices=self.freeUpperIndices)
        else:
            return NotImplemented

    def __str__(self):
        string = str(self.prefactor)
        if(len(self.vertexList) > 0):
            string = string + " * "
        for v in self.vertexList:
            string += v.__str__()
        for c in self.contractionsList:
            string += "\delta^{" + c[0].__str__() + "}_{" + c[1].__str__() + "}"
        return string

class TensorSum:
    def __init__(self, summandList):
        self.summandList = summandList

    def getOperator(self, normalOrderedParts=True):
        operator = 0
    #    for summand in self.summandList:
    #        operator = operator + summand.getOperator(spinFree, normalOrderedParts)
    #    return operator
        return sum([summand.getOperator(normalOrderedParts) for summand in self.summandList])

    def collectIsomorphicTerms(self, active=False):
        collected = TensorSum([])
        for summand in self.summandList:
            included = False
            for uniqueSummand in collected.summandList:
                if summand.isProportional(uniqueSummand, active):
                    included = True
#                    print("Summand", summand, "is proportional to", uniqueSummand)
                    uniqueSummand.prefactor += summand.prefactor
            if not included:
#                print("New summand", summand)
                collected.summandList.append(copy(summand))
        return collected

    def getConnectedTerms(self, active=False):
        linked = TensorSum([copy(summand) for summand in self.summandList if summand.isConnected()])
        return linked
    
    def collectConnectedIsomorphicTerms(self, active=False):
        collected = TensorSum([])
        for summand in self.summandList:
            included = False
            for uniqueSummand in collected.summandList:
                if summand.isProportional(uniqueSummand, active):
                    included = True
                    uniqueSummand.prefactor += summand.prefactor
            if summand.isConnected() and not included:
                collected.summandList.append(copy(summand))
        return collected

    def __copy__(self):
        return TensorSum([copy(summand) for summand in self.summandList])

    def __add__(self, other):
        if isinstance(other, TensorSum):
            return TensorSum(self.summandList + other.summandList)
        elif isinstance(other, TensorProduct):
            return TensorSum(self.summandList + [other])
        elif isinstance(other, Tensor):
            return TensorSum(self.summandList + [TensorProduct([other])])
        elif isinstance(other, Number):
            if other == 0:
                return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, TensorProduct):
            return TensorSum([other] + self.summandList)
        elif isinstance(other, Tensor):
            return TensorSum(self.summandList + [TensorProduct([other])])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor) or isinstance(other, TensorProduct) or isinstance(other, TensorSum) or isinstance(other, Number):
            return sum([summand * other for summand in self.summandList])

    def __rmul__(self, other):
        if isinstance(other, Tensor) or isinstance(other, TensorProduct) or isinstance(other, TensorSum) or isinstance(other, Number):
            return sum([other * summand for summand in self.summandList])

    def __str__(self):
        if len(self.summandList) == 0:
            return ""
        string = self.summandList[0].__str__()
        for summand in self.summandList[1:]:
            string += "\\\\ \n + "
            string += summand.__str__()
        return string