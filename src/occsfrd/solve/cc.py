import itertools
from time import time
import numpy as np
import networkx as nx
from pyscf import ao2mo
from copy import copy

from occsfrd.solve import diis
from occsfrd.wick import tensor, contractions
from occsfrd.ansatz import utils
from occsfrd.interface import storeequations

def amplitudeUpdates(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=0):
    if amplitudeDiagram.excitationRank == 1:
        return singlesAmplitudeUpdates(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=levelShift)
    elif amplitudeDiagram.excitationRank == 2:
        return doublesAmplitudeUpdates(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=levelShift)

def doublesAmplitudeUpdates(doublesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
    amplitudeUpdates = np.zeros_like(doublesTensor.getArray())
    residualArray = residual.getArray()
    for i in range(amplitudeUpdates.shape[0]):
        for j in range(amplitudeUpdates.shape[1]):
            for k in range(amplitudeUpdates.shape[2]):
                for l in range(amplitudeUpdates.shape[3]):
                    denominator = (fockMatrix[i + doublesTensor.indexRangeStartPoints[0], i + doublesTensor.indexRangeStartPoints[0]] + fockMatrix[j + doublesTensor.indexRangeStartPoints[1], j + doublesTensor.indexRangeStartPoints[1]] - fockMatrix[k + doublesTensor.indexRangeStartPoints[2], k + doublesTensor.indexRangeStartPoints[2]] - fockMatrix[l + doublesTensor.indexRangeStartPoints[3], l + doublesTensor.indexRangeStartPoints[3]]) + levelShift
                    amplitudeUpdates[i,j,k,l] = -residualArray[i,j,k,l] / denominator
                    if denominator < 0:
                        print("negative denominator", denominator, residual, i, j, k, l)
    return amplitudeUpdates

def singlesAmplitudeUpdates(singlesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
    amplitudeUpdates = np.zeros_like(singlesTensor.getArray())
    residualArray = residual.getArray()
    for i in range(amplitudeUpdates.shape[0]):
        for j in range(amplitudeUpdates.shape[1]):
            denominator = fockMatrix[i + singlesTensor.indexRangeStartPoints[0], i + singlesTensor.indexRangeStartPoints[0]] - fockMatrix[j + singlesTensor.indexRangeStartPoints[1], j + singlesTensor.indexRangeStartPoints[1]] + 0.5 * levelShift
            amplitudeUpdates[i,j] = -residualArray[i,j] / denominator
    return amplitudeUpdates

def iterateAmplitudes(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=0):
    if amplitudeDiagram.excitationRank == 1:
        return iterateSinglesAmplitudes(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=levelShift)
    elif amplitudeDiagram.excitationRank == 2:
        return iterateDoublesAmplitudes(amplitudeDiagram, residualDiagram, fockArray, spinFree, levelShift=levelShift)

def iterateDoublesAmplitudes(doublesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
# def iterateDoublesAmplitudes(doublesTensor, residual, fockMatrix, spinFree=True, levelShift=False):
    amplitudes = doublesTensor.getArray()
    residualArray = residual.getArray()
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
            for k in range(amplitudes.shape[2]):
                for l in range(amplitudes.shape[3]):
#                    amplitudes[i,j,k,l] -= residual.getArray()[i,j,k,l] / (fockMatrix[i + amplitudes.shape[2], i + amplitudes.shape[2]] + fockMatrix[j + amplitudes.shape[3], j + amplitudes.shape[3]] - fockMatrix[k, k] - fockMatrix[l, l])
                    denominator = (fockMatrix[i + doublesTensor.indexRangeStartPoints[0], i + doublesTensor.indexRangeStartPoints[0]] + fockMatrix[j + doublesTensor.indexRangeStartPoints[1], j + doublesTensor.indexRangeStartPoints[1]] - fockMatrix[k + doublesTensor.indexRangeStartPoints[2], k + doublesTensor.indexRangeStartPoints[2]] - fockMatrix[l + doublesTensor.indexRangeStartPoints[3], l + doublesTensor.indexRangeStartPoints[3]]) + levelShift
                    amplitudes[i,j,k,l] -= residualArray[i,j,k,l] / denominator
                    # denominator = (fockMatrix[i + doublesTensor.indexRangeStartPoints[0], i + doublesTensor.indexRangeStartPoints[0]] + fockMatrix[j + doublesTensor.indexRangeStartPoints[1], j + doublesTensor.indexRangeStartPoints[1]] - fockMatrix[k + doublesTensor.indexRangeStartPoints[2], k + doublesTensor.indexRangeStartPoints[2]] - fockMatrix[l + doublesTensor.indexRangeStartPoints[3], l + doublesTensor.indexRangeStartPoints[3]])
                    # amplitudes[i,j,k,l] -= residualArray[i,j,k,l] / abs(denominator)
                    if denominator < 0:
                        print("negative denominator", denominator, residual, i, j, k, l)
#                     if levelShift:
#                         denominator += 0.5
# #                    if not (((i + doublesTensor.indexRangeStartPoints[0] == k + doublesTensor.indexRangeStartPoints[2]) and (j + doublesTensor.indexRangeStartPoints[1] == l + doublesTensor.indexRangeStartPoints[3])) or ((i + doublesTensor.indexRangeStartPoints[0] == l + doublesTensor.indexRangeStartPoints[3]) and (j + doublesTensor.indexRangeStartPoints[1] == k + doublesTensor.indexRangeStartPoints[2]))):
# #                    if not (np.isclose(denominator, 0, rtol=0, atol=1e-4)):
#                     if denominator > 0:
#                         amplitudes[i,j,k,l] -= residualArray[i,j,k,l] / denominator
#                     else:
#                         # print("zero denominator")
#                         if levelShift:
#                             amplitudes[i,j,k,l] -= residualArray[i,j,k,l] / denominator
#    if spinFree:
#        amplitudes = (1./3.) * amplitudes + (1./6.) * amplitudes.swapaxes(0,1)
    return amplitudes

def iterateSinglesAmplitudes(singlesTensor, residual, fockMatrix, spinFree=True, levelShift=0.):
    amplitudes = singlesTensor.getArray()
    residualArray = residual.getArray()
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
#            amplitudes[i,j] -= (2 - (i==j)) * residual.array[i,j] / (fockMatrix[i + amplitudes.shape[1], i + amplitudes.shape[1]] - fockMatrix[j, j])
#            amplitudes[i,j] -= residual.array[i,j] / (fockMatrix[i + amplitudes.shape[1], i + amplitudes.shape[1]] - fockMatrix[j, j])
            denominator = fockMatrix[i + singlesTensor.indexRangeStartPoints[0], i + singlesTensor.indexRangeStartPoints[0]] - fockMatrix[j + singlesTensor.indexRangeStartPoints[1], j + singlesTensor.indexRangeStartPoints[1]] + 0.5 * levelShift
            amplitudes[i,j] -= residualArray[i,j] / denominator
    return amplitudes

def iterateTriplesAmplitudes(triplesTensor, residual, fockMatrix):
    amplitudes = triplesTensor.array
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
            for k in range(amplitudes.shape[2]):
                for l in range(amplitudes.shape[3]):
                    for m in range(amplitudes.shape[4]):
                        for n in range(amplitudes.shape[5]):
                            amplitudes[i,j,k,l,m,n] -= residual.array[i,j,k,l,m,n] / (fockMatrix[i + triplesTensor.indexRangeStartPoints[0], i + triplesTensor.indexRangeStartPoints[0]] + fockMatrix[j + triplesTensor.indexRangeStartPoints[1], j + triplesTensor.indexRangeStartPoints[1]] + fockMatrix[k + triplesTensor.indexRangeStartPoints[2], k + triplesTensor.indexRangeStartPoints[2]] - fockMatrix[l + triplesTensor.indexRangeStartPoints[3], l + triplesTensor.indexRangeStartPoints[3]] - fockMatrix[m + triplesTensor.indexRangeStartPoints[4], m + triplesTensor.indexRangeStartPoints[4]] - fockMatrix[n + triplesTensor.indexRangeStartPoints[5], n + triplesTensor.indexRangeStartPoints[5]])
    return amplitudes

def convergeDoublesAmplitudes(doublesTensor, CCDEnergyEquation, CCDAmplitudeEquation, fockTensor, tol=10, spinFree=True, biorthogonal=False, verbosity=0):
    residualTensor = tensor.Tensor("R", ['p', 'p'], ['h', 'h'])
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    Energy = contractions.contractTensorSum(CCDEnergyEquation)
    residualTensor.array = contractions.contractTensorSum(CCDAmplitudeEquation)
    if not biorthogonal:
        residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
    if verbosity > 1:
        print(residualTensor.array)
    thresh = pow(10, -tol)
    while True:
        print(Energy)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, residualTensor, fockTensor.array, spinFree)
        residualTensor.array = contractions.contractTensorSum(CCDAmplitudeEquation)
        if not biorthogonal:
            residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
        oldEnergy = Energy
        Energy = contractions.contractTensorSum(CCDEnergyEquation)
        if np.all(abs(residualTensor.array) < thresh) and (Energy - oldEnergy) < thresh:
            break
    print(Energy)
    if verbosity > 1:
        print(doublesTensor.array)

def convergeCollectedDoublesAmplitudes(doublesTensor, CCDEnergyEquation, collectedCCDAmplitudeEquation, fockTensor, tol=10, verbosity=0):
    residualTensor = tensor.Tensor("R", ['p', 'p'], ['h', 'h'])
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    Energy = contractions.contractTensorSum(CCDEnergyEquation)
    residualTensor.array = contractions.contractTensorSum(collectedCCDAmplitudeEquation)
#    residualTensor.array = residualTensor.array - (1./2.) * residualTensor.array.swapaxes(0,1)
    if verbosity > 1:
        print(residualTensor.array)
    thresh = pow(10, -tol)
    while True:
        print(Energy)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, residualTensor, fockTensor.array)
        residualTensor.array = contractions.contractTensorSum(collectedCCDAmplitudeEquation)
#        residualTensor.array = residualTensor.array - (1./2.) * residualTensor.array.swapaxes(0,1)
        oldEnergy = Energy
        Energy = contractions.contractTensorSum(CCDEnergyEquation)
        if np.all(abs(residualTensor.array) < thresh) and (Energy - oldEnergy) < thresh:
            break
    print(Energy)
    if verbosity > 1:
        print(doublesTensor.array)

def convergeCCSDAmplitudes(singlesTensor, doublesTensor, CCSDEnergyEquation, singlesCCSDAmplitudeEquation, doublesCCSDAmplitudeEquation, fockTensor, tol=10, biorthogonal=False, verbosity=0):
    singlesResidualTensor = tensor.Tensor("R", ['p'], ['h'])
    doublesResidualTensor = tensor.Tensor("R", ['p', 'p'], ['h', 'h'])
    singlesTensor.array = np.zeros_like(singlesTensor.array)
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    Energy = contractions.contractTensorSum(CCSDEnergyEquation)
    singlesResidualTensor.array = contractions.contractTensorSum(singlesCCSDAmplitudeEquation)
    doublesResidualTensor.array = contractions.contractTensorSum(doublesCCSDAmplitudeEquation)
    if not biorthogonal:
        doublesResidualTensor.array = (1./3.) * doublesResidualTensor.array + (1./6.) * doublesResidualTensor.array.swapaxes(0,1)
    thresh = pow(10, -tol)
    while True:
        print(Energy)
        singlesTensor.array = iterateSinglesAmplitudes(singlesTensor, singlesResidualTensor, fockTensor.array)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, doublesResidualTensor, fockTensor.array)
        singlesResidualTensor.array = contractions.contractTensorSum(singlesCCSDAmplitudeEquation)
        doublesResidualTensor.array = contractions.contractTensorSum(doublesCCSDAmplitudeEquation)
        if not biorthogonal:
            doublesResidualTensor.array = (1./3.) * doublesResidualTensor.array + (1./6.) * doublesResidualTensor.array.swapaxes(0,1)
        oldEnergy = Energy
        Energy = contractions.contractTensorSum(CCSDEnergyEquation)
        if np.all(abs(singlesResidualTensor.array) < thresh) and np.all(abs(doublesResidualTensor.array) < thresh) and (Energy - oldEnergy) < thresh:
            break
    print(Energy)
    if verbosity > 1:
        print(singlesTensor.array)
        print(doublesTensor.array)

def convergeCCSDTAmplitudes(singlesTensor, doublesTensor, triplesTensor, CCSDTEnergyEquation, singlesCCSDTAmplitudeEquation, doublesCCSDTAmplitudeEquation, triplesCCSDTAmplitudeEquation,  fockTensor, tol=10, verbosity=0):
    singlesResidualTensor = tensor.Tensor("R", ['p'], ['h'])
    doublesResidualTensor = tensor.Tensor("R", ['p', 'p'], ['h', 'h'])
    triplesResidualTensor = tensor.Tensor("R", ['p', 'p', 'p'], ['h', 'h', 'h'])
    singlesTensor.array = np.zeros_like(singlesTensor.array)
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    triplesTensor.array = np.zeros_like(triplesTensor.array)
    Energy = contractions.contractTensorSum(CCSDTEnergyEquation)
    singlesResidualTensor.array = contractions.contractTensorSum(singlesCCSDTAmplitudeEquation)
    doublesResidualTensor.array = contractions.contractTensorSum(doublesCCSDTAmplitudeEquation)
    triplesResidualTensor.array = contractions.contractTensorSum(triplesCCSDTAmplitudeEquation)
    thresh = pow(10, -tol)
    while True:
        print(Energy)
        singlesTensor.array = iterateSinglesAmplitudes(singlesTensor, singlesResidualTensor, fockTensor.array)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, doublesResidualTensor, fockTensor.array)
        triplesTensor.array = iterateTriplesAmplitudes(triplesTensor, triplesResidualTensor, fockTensor.array)
        singlesResidualTensor.array = contractions.contractTensorSum(singlesCCSDTAmplitudeEquation)
        doublesResidualTensor.array = contractions.contractTensorSum(doublesCCSDTAmplitudeEquation)
        triplesResidualTensor.array = contractions.contractTensorSum(triplesCCSDTAmplitudeEquation)
        oldEnergy = Energy
        Energy = contractions.contractTensorSum(CCSDTEnergyEquation)
        if np.all(abs(singlesResidualTensor.array) < thresh) and np.all(abs(doublesResidualTensor.array) < thresh) and np.all(abs(triplesResidualTensor.array) < thresh) and (Energy - oldEnergy) < thresh:
            break
    print(Energy)
    if verbosity > 1:
        print(singlesTensor.array)
        print(doublesTensor.array)
        print(triplesTensor.array)

def convergeUnlinkedDoublesAmplitudes(doublesTensor, unlinkedCCDEnergyEquation, unlinkedCCDAmplitudeEquation, unlinkedCCDAmplitudeEquationCorrectionOverE, fockTensor, tol=10, spinFree=True, biorthogonal=False, verbosity=0):
    residualTensor = tensor.Tensor("R", ['p', 'p'], ['h', 'h'])
    residualVertex = tensor.TensorProduct([residualTensor]).vertexList[0]
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    Energy = contractions.contractTensorSum(unlinkedCCDEnergyEquation)
    residualTensor.array = contractions.contractTensorSum(unlinkedCCDAmplitudeEquation + (-1) * Energy * unlinkedCCDAmplitudeEquationCorrectionOverE, residualVertex.lowerIndices, residualVertex.upperIndices)
#    testResidualArray = contractTensorSum(newLinkedDoublesAmplitudeEquation)
#    print((abs(residualTensor.array-testResidualArray) < pow(10, -12)).all())
    if not biorthogonal:
        residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
#    print(residualTensor.array)
    thresh = pow(10, -tol)
    while True:
        print(Energy)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, residualTensor, fockTensor.array, spinFree)
        oldEnergy = Energy
        Energy = contractions.contractTensorSum(unlinkedCCDEnergyEquation)
        residualTensor.array = contractions.contractTensorSum(unlinkedCCDAmplitudeEquation + (-1) * Energy * unlinkedCCDAmplitudeEquationCorrectionOverE, residualVertex.lowerIndices, residualVertex.upperIndices)
#        testResidualArray = contractTensorSum(newLinkedDoublesAmplitudeEquation)
#        print((abs(residualTensor.array-testResidualArray) < pow(10, -16)).all())
        if not biorthogonal:
            residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
        if np.all(abs(residualTensor.array) < thresh) and (Energy - oldEnergy) < thresh:
            break
    print(Energy)
    if verbosity > 1:
        print(doublesTensor.array)

def convergeSeparateUnlinkedDoublesAmplitudesOnlyLinked(amplitudeTensors, unlinkedEnergyEquation, unlinkedAmplitudeEquations, unlinkedAmplitudeEquationCorrectionsOverE, fockTensor, nCore, nActive, nVirtual, tol=10, spinFree=True, biorthogonal=False, maxIterations=10000, verbosity=0):
#    print(nCore, nActive, nVirtual)
    residualTensors = []
    for tensor in amplitudeTensors:
#        print(tensor.array.shape)
        tensor.array = np.zeros_like(tensor.array)
        tensor.assignDiagramArraysActive(nCore, nActive, nVirtual)
        residualTensor = tensor.Tensor("R", tensor.lowerIndexTypes, tensor.upperIndexTypes)
        residualTensor.getShapeActive((nCore + nActive, nCore), nCore + nActive + nVirtual)
#        print(residualTensor.array.shape)
#        residualTensor.array = np.zeros_like(tensor.array)
        residualTensor.getAllDiagramsActive()
        residualTensor.assignDiagramArraysActive(nCore, nActive, nVirtual)
        residualTensors.append(residualTensor)
#    print(*residualTensors)
#    print(unlinkedEnergyEquation)
    Energy = contractions.contractTensorSum(unlinkedEnergyEquation)
    newEquations = []
    for r, residualTensor in enumerate(residualTensors):
        newEquations.append([])
        for equation in unlinkedAmplitudeEquations[r]:
            newEquations[r].append(tensor.TensorSum([summand for s, summand in enumerate(equation.summandList) if nx.is_weakly_connected(summand.getGraph())]))
        for d, diag in enumerate(residualTensor.diagrams):
            residualVertex = tensor.TensorProduct([diag]).vertexList[0]
            diag.array = contractions.contractTensorSum(newEquations[r][d], residualVertex.lowerIndices, residualVertex.upperIndices)
#        diag.array = CC.contractTensorSum(unlinkedCCDAmplitudeEquations[d] + (-1) * Energy * unlinkedCCDAmplitudeEquationCorrectionsOverE[d], residualVertex.lowerIndices, residualVertex.upperIndices)
#        print(diag.arraySlices, diag.array.shape, diag.array)
#    print(residualTensor.array.shape, residualTensor.array)
#    reassembleArray(residualTensor)
        residualTensor.reassembleArray()
#    testResidualArray = contractTensorSum(newLinkedDoublesAmplitudeEquation)
#    print((abs(residualTensor.array-testResidualArray) < pow(10, -12)).all())
        if not biorthogonal and residualTensor.excitationRank == 2:
            residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
#    print(residualTensor.array)
    thresh = pow(10, -tol)
#    print(residualTensor.array.shape)
#    print(doublesTensor.array.shape)
    i = 1
    while True:
        print(Energy)
        for t, tensor in enumerate(amplitudeTensors):
            if tensor.excitationRank == 1:
                tensor.array = iterateSinglesAmplitudes(tensor, residualTensors[t], fockTensor.array, spinFree)
            elif tensor.excitationRank == 2:
                tensor.array = iterateDoublesAmplitudes(tensor, residualTensors[t], fockTensor.array, spinFree)
        oldEnergy = Energy
        Energy = contractions.contractTensorSum(unlinkedEnergyEquation)
        for r, residualTensor in enumerate(residualTensors):
            for d, diag in enumerate(residualTensor.diagrams):
                residualVertex = tensor.TensorProduct([diag]).vertexList[0]
                diag.array = contractions.contractTensorSum(newEquations[r][d], residualVertex.lowerIndices, residualVertex.upperIndices)
#            diag.array = CC.contractTensorSum(unlinkedCCDAmplitudeEquations[d] + (-1) * Energy * unlinkedCCDAmplitudeEquationCorrectionsOverE[d], residualVertex.lowerIndices, residualVertex.upperIndices)
            residualTensor.reassembleArray()
#        testResidualArray = contractTensorSum(newLinkedDoublesAmplitudeEquation)
#        print((abs(residualTensor.array-testResidualArray) < pow(10, -16)).all())
            if not biorthogonal and residualTensor.excitationRank == 2:
                residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
        i += 1
        if (np.all([np.all(abs(residualTensor.array) < thresh) for r, residualTensor in enumerate(residualTensors)]) and (Energy - oldEnergy) < thresh) or i > maxIterations:
            break
    print(Energy)
    if verbosity > 1:
        print([tensor.array for tensor in amplitudeTensors])

def convergeUnlinkedAmplitudes(Norbs, Nelec, Nactive, amplitudeTensors, unlinkedEnergyEquationAndNorm, unlinkedAmplitudeEquationsAndCorrectionsOverE, fockTensor, Rtol=10, Etol=8, spinFree=True, biorthogonal=False, verbosity=0, levelShift=0., maxIter=100, nDIIS=0, maxOrder=2, onlyConnect=False):
# def convergeUnlinkedAmplitudes(Norbs, Nelec, Nactive, amplitudeTensors, unlinkedEnergyEquation, unlinkedAmplitudeEquationsAndCorrectionsOverE, fockTensor, tol=8, spinFree=True, biorthogonal=False, verbose=False, levelShift=False):
    t0 = time()
    residualTensors = [tensor.Tensor("R", amplitudeTensor.lowerIndexTypes, amplitudeTensor.upperIndexTypes) for amplitudeTensor in amplitudeTensors]
    residualVertices = []
    for rT, residualTensor in enumerate(residualTensors):
        residualTensor.getAllDiagramsActive()
        residualTensor.getShapeActive(Nelec, Norbs)
        residualTensor.assignDiagramArraysActive(Nelec[1], Nactive, Norbs-Nactive-Nelec[1])
        # print(residualTensor, *residualTensor.diagrams)
    for rD, residualDiagram in enumerate(itertools.chain.from_iterable([residualTensor.diagrams for residualTensor in residualTensors])):
        equation = unlinkedAmplitudeEquationsAndCorrectionsOverE[rD]
        if len(equation[0].summandList) == 0:
            residualVertices.append(tensor.TensorProduct([residualDiagram]).vertexList[0])
        else:
            lowerIndexTypes = [lowerIndex.name[0] for lowerIndex in equation[0].summandList[0].freeLowerIndices]
            upperIndexTypes = [upperIndex.name[0] for upperIndex in equation[0].summandList[0].freeUpperIndices]
            assert residualDiagram.lowerIndexTypes == lowerIndexTypes and residualDiagram.upperIndexTypes == upperIndexTypes
#        residualVertices += [tensor.TensorProduct([residualDiagram]).vertexList[0] for residualDiagram in residualTensor.diagrams]
            residualVertices.append(tensor.Vertex(residualDiagram, equation[0].summandList[0].freeLowerIndices, equation[0].summandList[0].freeUpperIndices))
        print(residualDiagram)
    for amplitudeTensor in amplitudeTensors:
        amplitudeTensor.array = np.zeros_like(amplitudeTensor.array)
    # Energy = contractions.contractTensorSum(unlinkedEnergyEquation, lowerIndexList=[], upperIndexList=[])
    # Energy = contractions.testOldContractTensorSum(unlinkedEnergyEquation, lowerIndexList=[], upperIndexList=[])
#    contractions.testEqualTermsInTensorSum(unlinkedEnergyEquation)
    IntermediateNorm = unlinkedEnergyEquationAndNorm[1]
    EnergyEquationsByOrder = tuple(tensor.TensorSum([summand for summand in unlinkedEnergyEquationAndNorm[0].summandList if len(summand.tensorList) == order+1]) for order in range(maxOrder+1))
    EnergyContributionsByOrder = [contractions.contractTensorSum(equation, lowerIndexList=[], upperIndexList=[]) for equation in EnergyEquationsByOrder]
    Energy = sum(EnergyContributionsByOrder)
    Energy0 = Energy
    print("Reference energy relative to vacuum:", Energy)
    print("Intermediate normalization check:", contractions.contractTensorSum(IntermediateNorm, lowerIndexList=[], upperIndexList=[]))
    Ecorr = Energy - Energy0
    # print(Energy, Ecorr)
    i = 0
#    print(*residualVertices)
    UnlinkedPartsByOrder = []
    onlyConnectedPartsByOrder = []
    for eq, equation in enumerate(unlinkedAmplitudeEquationsAndCorrectionsOverE):
        if len(equation[0].summandList) > 0:
            lowerIndexTypes = [lowerIndex.name[0] for lowerIndex in equation[0].summandList[0].freeLowerIndices]
            upperIndexTypes = [upperIndex.name[0] for upperIndex in equation[0].summandList[0].freeUpperIndices]
        unlinkedPartSplitByOrder = tuple([tensor.TensorSum([summand for summand in equation[1].summandList if len(summand.tensorList) == order]) for order in range(maxOrder+1)])
        UnlinkedPartsByOrder.append(unlinkedPartSplitByOrder)
        if onlyConnect:
            onlyConnectedPartsSplitByOrder = tuple([tensor.TensorSum([summand for summand in equation[0].summandList if len(summand.tensorList) == order+1 if summand.isConnected()]) for order in range(maxOrder+1)])
            onlyConnectedPartsByOrder.append(onlyConnectedPartsSplitByOrder)
        # print(eq, lowerIndexTypes, upperIndexTypes, residualVertices[eq].tensor.lowerIndexTypes, residualVertices[eq].tensor.upperIndexTypes)
#        residualVertices[eq].upperIndices = equation[0].summandList[0].freeUpperIndices
#        residualVertices[eq].lowerIndices = equation[0].summandList[0].freeLowerIndices
#        residualVertices[eq].tensor.setArray(contractions.contractTensorSum(equation[0] + (-1) * Energy * equation[1]))
        for rV, residualVertex in enumerate(residualVertices):
            if equation[0].summandList == []:
                pass
            elif equation[0].summandList[0].freeLowerIndices == residualVertex.lowerIndices and equation[0].summandList[0].freeUpperIndices == residualVertex.upperIndices:
#            elif upperIndexTypes == residualVertex.tensor.upperIndexTypes and lowerIndexTypes == residualVertex.tensor.lowerIndexTypes:
                # print(eq, rV, residualVertex, lowerIndexTypes, upperIndexTypes)
                residualShape = residualVertex.tensor.getArray().shape
                # print(eq, rV)
                # if eq == 4 and rV == 4:
                #     print("test")
                # residualVertex.tensor.setArray(contractions.contractTensorSum(equation[0] + (-1) * Energy * equation[1], resultShape=residualShape))
                # print(maxOrder, [(i, j) for i in range(maxOrder+1) for j in range(maxOrder+1-i)])
                # print(len(EnergyContributionsByOrder), len(UnlinkedPartsByOrder[eq]))
                if onlyConnect:
                    residualVertex.tensor.setArray(contractions.contractTensorSum(sum([onlyConnectedPartsByOrder[eq][i] for i in range(maxOrder+1)]), resultShape=residualShape))
                else:
                    residualVertex.tensor.setArray(contractions.contractTensorSum(equation[0] + (-1) * sum([EnergyContributionsByOrder[i] * UnlinkedPartsByOrder[eq][j] for i in range(maxOrder+1) for j in range(maxOrder+1-i)]), resultShape=residualShape))
                # residualVertex.tensor.setArray(contractions.testOldContractTensorSum(equation[0] + (-1) * Energy * equation[1]))
                # print("test", residualVertex)
                # contractions.testEqualTermsInTensorSum(equation[0] + (-1) * Energy * equation[1], resultShape=residualShape)
                # print(contractions.testEqualTensorSum(equation[0] + (-1) * Energy * equation[1], resultShape=residualShape))
#    print(*residualVertices)
    if not biorthogonal:
        for residualTensor in residualTensors:
            if residualTensor.excitationRank == 2 and residualTensor.lowerIndexTypes[0] == residualTensor.lowerIndexTypes[1] and residualTensor.upperIndexTypes[0] == residualTensor.upperIndexTypes[1]:
                residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
    Rthresh = pow(10, -Rtol)
    Ethresh = pow(10, -Etol)
    totalNormSquared = sum([utils.normSquared(residualTensor) for residualTensor in residualTensors])

    print("Iteration:", i, "Correlation energy:", Ecorr, "Residual norm squared:", totalNormSquared, "Intermediate normalization check:", contractions.contractTensorSum(IntermediateNorm, lowerIndexList=[], upperIndexList=[]))
    diisWeights = np.ndarray((0,), dtype=object)
    diisAmpUpdates = np.ndarray((0,), dtype=object)
    diisAmps = np.ndarray((0,), dtype=object)
    if nDIIS>0:
        diisAmpUpdates = np.reshape(np.array([amplitudeUpdates(amplitudeDiagram, residualTensors[aT].diagrams[aD], fockTensor.array, spinFree, levelShift=levelShift) for aT, amplitudeTensor in enumerate(amplitudeTensors) for aD, amplitudeDiagram in enumerate(amplitudeTensor.diagrams)], dtype=object), (1, -1))
        diisWeights = np.array([1])
        # diisAmps = copy(diisAmpUpdates)
        diisAmps = np.reshape(np.array([copy(amplitudeDiagram.getArray()) for aT, amplitudeTensor in enumerate(amplitudeTensors) for aD, amplitudeDiagram in enumerate(amplitudeTensor.diagrams)], dtype=object), (1, -1))
    while i < maxIter:
        if nDIIS > 0:
            diisAmps = np.concatenate((diisAmps, np.reshape(diis.updateAmpsDIIS(diisWeights, diisAmps, diisAmpUpdates), (1, -1))), 0)
        k = 0
        for a, amplitudeTensor in enumerate(amplitudeTensors):
            # if verbose:
            #     print(amplitudeTensor, "Amplitude norm squared:", utils.normSquared(amplitudeTensor))
            for aD, amplitudeDiagram in enumerate(amplitudeTensor.diagrams):
                if nDIIS > 0:
                    amplitudeDiagram.setArray(diisAmps[-1][k])
                    k += 1
                else:
                    amplitudeDiagram.setArray(iterateAmplitudes(amplitudeDiagram, residualTensors[a].diagrams[aD], fockTensor.array, spinFree, levelShift=levelShift))
                if verbosity > 1:
                    if amplitudeDiagram.excitationRank == 1 or (amplitudeDiagram.excitationRank == 2 and "a" in amplitudeDiagram.upperIndexTypes and "a" in amplitudeDiagram.lowerIndexTypes):
                        print("Singles and spectator amplitudes:", amplitudeDiagram)
                        print(amplitudeDiagram.getArray())
                # if amplitudeDiagram.upperIndexTypes == ["a", "a"] and (amplitudeDiagram.lowerIndexTypes == ["a", "a"] or amplitudeDiagram.lowerIndexTypes == ["a", "v"] or amplitudeDiagram.lowerIndexTypes == ["v", "a"]):
                #     print("Exclude amplitude from interacting space", amplitudeDiagram)
                # else:
                #     amplitudeDiagram.setArray(iterateDoublesAmplitudes(amplitudeDiagram, residualTensors[a].diagrams[aD], fockTensor.array, spinFree, levelShift=levelShift))
#            amplitudeTensor.array = iterateDoublesAmplitudes(amplitudeTensor, residualTensors[a], fockTensor.array, spinFree)
        i += 1
        oldEnergy = Energy
        # Energy = contractions.contractTensorSum(unlinkedEnergyEquation, lowerIndexList=[], upperIndexList=[])
        EnergyContributionsByOrder = [contractions.contractTensorSum(equation, lowerIndexList=[], upperIndexList=[]) for equation in EnergyEquationsByOrder]
        Energy = sum(EnergyContributionsByOrder)
        Ecorr = Energy - Energy0
        # print(Ecorr)
        for eq, equation in enumerate(unlinkedAmplitudeEquationsAndCorrectionsOverE):
            for rV, residualVertex in enumerate(residualVertices):
                if equation[0].summandList == []:
                    pass
                elif equation[0].summandList[0].freeLowerIndices == residualVertex.lowerIndices and equation[0].summandList[0].freeUpperIndices == residualVertex.upperIndices:
            #    elif upperIndexTypes == residualVertex.tensor.upperIndexTypes and lowerIndexTypes == residualVertex.tensor.lowerIndexTypes:
                    residualShape = residualVertex.tensor.getArray().shape
                    # residualVertex.tensor.setArray(contractions.contractTensorSum(equation[0] + (-1) * Energy * equation[1], resultShape=residualShape))
                    if onlyConnect:
                        residualVertex.tensor.setArray(contractions.contractTensorSum(sum([onlyConnectedPartsByOrder[eq][i] for i in range(maxOrder+1)]), resultShape=residualShape))
                    else:
                        residualVertex.tensor.setArray(contractions.contractTensorSum(equation[0] + (-1) * sum([EnergyContributionsByOrder[i] * UnlinkedPartsByOrder[eq][j] for i in range(maxOrder+1) for j in range(maxOrder+1-i)]), resultShape=residualShape))# + (-1) * EnergyContributionsByOrder[1] * UnlinkedPartsByOrder[eq][1] + (-1) * EnergyContributionsByOrder[2] * UnlinkedPartsByOrder[eq][0]
                    # residualVertex.tensor.setArray(contractions.testOldContractTensorSum(equation[0] + (-1) * Energy * equation[1]))
        if not biorthogonal:
            for residualTensor in residualTensors:
                if residualTensor.excitationRank == 2 and residualTensor.lowerIndexTypes[0] == residualTensor.lowerIndexTypes[1] and residualTensor.upperIndexTypes[0] == residualTensor.upperIndexTypes[1]:
                    residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
        totalNormSquared = sum([utils.normSquared(residualTensor) for residualTensor in residualTensors])
        print("Iteration:", i, "Correlation energy:", Ecorr, "Residual norm squared:", totalNormSquared, "Intermediate normalization check:", contractions.contractTensorSum(IntermediateNorm, lowerIndexList=[], upperIndexList=[]))
        if nDIIS > 0:
            newAmpUpdates = np.array([amplitudeUpdates(amplitudeDiagram, residualTensors[aT].diagrams[aD], fockTensor.array, spinFree, levelShift=levelShift) for aT, amplitudeTensor in enumerate(amplitudeTensors) for aD, amplitudeDiagram in enumerate(amplitudeTensor.diagrams)], dtype=object)
            diisAmpUpdates = np.concatenate((diisAmpUpdates, np.reshape(newAmpUpdates, (1, -1))), 0)
            if diisAmpUpdates.shape[0] > nDIIS:
                diisAmpUpdates = diisAmpUpdates[1:]
                diisAmps = diisAmps[1:]
            diisWeights = diis.getDIISWeights(diisAmpUpdates)
        if verbosity > 0:
            # for residualTensor in residualTensors:
                # print(residualTensor, "Residual norm squared:", utils.normSquared(residualTensor))
            for rD, residualDiagram in enumerate(itertools.chain.from_iterable([residualTensor.diagrams for residualTensor in residualTensors])):
                print(residualDiagram, "Residual norm squared:", utils.normSquared(residualDiagram))
            # print("Overall residual norm squared:", totalNormSquared)
#        if all([np.all(abs(residualTensor.array) < thresh) for residualTensor in residualTensors]) and (Energy - oldEnergy) < thresh:
        if totalNormSquared < Rthresh and abs(Energy - oldEnergy) < Ethresh:
            break
    # Energy = contractions.contractTensorSum(unlinkedEnergyEquation, lowerIndexList=[], upperIndexList=[])
    EnergyContributionsByOrder = [contractions.contractTensorSum(equation, lowerIndexList=[], upperIndexList=[]) for equation in EnergyEquationsByOrder]
    Energy = sum(EnergyContributionsByOrder)
    # i += 1
    Ecorr = Energy - Energy0
    print("Final iteration:", i, "Correlation energy:", Ecorr)
    # print(Ecorr)
    if verbosity > 1:
        print(*[amplitudeTensor.array for amplitudeTensor in amplitudeTensors])
    t1 = time()
    print("Time to converge:", t1-t0)
    return {"correlation energy": Ecorr, "amplitudes": amplitudeTensors, "total electronic energy": Energy}

def runUnlinkedCC(mf, equationsDict, levelShift=0., verbosity=0, biorthogonal=False, Rtol=10, Etol=8, maxIter=100, nDIIS=0, maxOrder=2, onlyConnect=False):
# def runUnlinkedCC(mf, equationsDict, levelShift=False):
    mol = mf.mol
#    equationsDict = storeequations.load(equationFileName)

    hTensor = equationsDict["tensors"][0]
    gTensor = equationsDict["tensors"][1]
    amplitudeTensors = equationsDict["tensors"][2:]
    specificIndices = equationsDict["specificIndices"]
    energyEquationAndNorm = equationsDict["equations"][0]
    amplitudeEquations = equationsDict["equations"][1:]
    
    Norbs = mol.nao
    Nocc = mf.nelec[1]
    Nactive = len(specificIndices)
    Nvirtual = Norbs - Nactive - Nocc
    vacuum = [1] * Nocc + [0] * (Norbs - Nocc)

    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False)

    hTensor.getShapeActive(mf.nelec, Norbs)
    gTensor.getShapeActive(mf.nelec, Norbs)
    gTensor.array = eri.reshape((Norbs, Norbs, Norbs, Norbs)).swapaxes(2,3).swapaxes(1,2)
    fock = h1
    for p in range(Norbs):
        for q in range(Norbs):
            fock[p,q] += sum([2 * gTensor.array[p,i,q,i] - gTensor.array[p,i,i,q] for i in range(Nocc)])
    #        fock[p,q] += sum([2 * gTensor.array[p,i,q,i] - gTensor.array[p,i,i,q] for i in range(Nocc,Nocc+int(Nactive/2))])
    hTensor.array = fock

    # print("core-core", fock[:Nocc,:Nocc])
    # print("active-active", fock[Nocc:Nocc+Nactive, Nocc:Nocc+Nactive])
    # print("virtual-virtual", fock[Nocc+Nactive:, Nocc+Nactive:])

    if Nactive != 0:
        dm = mf.make_rdm1()[0,mf.nelec[1]:mf.nelec[0], mf.nelec[1]:mf.nelec[0]] + mf.make_rdm1()[1,mf.nelec[1]:mf.nelec[0], mf.nelec[1]:mf.nelec[0]]
    #    print(dm)
        for p in range(Norbs):
            for q in range(Norbs):
                hTensor.array[p,q] += sum([dm[u,v] * gTensor.array[p,u,q,v] - 0.5 * dm[u,v] * gTensor.array[p,u,v,q] for v in range(mf.nelec[1]-mf.nelec[0]) for u in range(mf.nelec[1]-mf.nelec[0])])

    gTensor.assignDiagramArraysActive(Nocc, Nactive, Nvirtual)
    hTensor.assignDiagramArraysActive(Nocc, Nactive, Nvirtual)

    for tTensor in amplitudeTensors:
        tTensor.getShapeActive(mf.nelec, Norbs)
        tTensor.assignDiagramArraysActive(Nocc, Nactive, Nvirtual)
    
    if nDIIS > 0:
        print("Convergence aided with DIIS, subspace size:", nDIIS)

    if levelShift != 0.:
        print("Level shift applied:", levelShift)
    # if levelShift:
    #     print("level shift applied")

    return convergeUnlinkedAmplitudes(Norbs, mf.nelec, Nactive, amplitudeTensors, energyEquationAndNorm, amplitudeEquations, hTensor, levelShift=levelShift, verbosity=verbosity, biorthogonal=biorthogonal, Rtol=Rtol, Etol=Etol, maxIter=maxIter, nDIIS=nDIIS, maxOrder=maxOrder, onlyConnect=onlyConnect)
    # return convergeUnlinkedAmplitudes(Norbs, mf.nelec, Nactive, amplitudeTensors, energyEquation, amplitudeEquations, hTensor, levelShift=levelShift)

def getReferenceEnergy(mf, equationsDict, levelShift=0., verbosity=0, biorthogonal=False, Rtol=10, Etol=8, maxIter=100, nDIIS=0, maxOrder=2, onlyConnect=False):

# def runUnlinkedCC(mf, equationsDict, levelShift=False):
    mol = mf.mol
#    equationsDict = storeequations.load(equationFileName)

    hTensor = equationsDict["tensors"][0]
    gTensor = equationsDict["tensors"][1]
    amplitudeTensors = equationsDict["tensors"][2:]
    specificIndices = equationsDict["specificIndices"]
    energyEquationAndNorm = equationsDict["equations"][0]
    amplitudeEquations = equationsDict["equations"][1:]
    
    Norbs = mol.nao
    Nocc = mf.nelec[1]
    Nactive = len(specificIndices)
    Nvirtual = Norbs - Nactive - Nocc
    vacuum = [1] * Nocc + [0] * (Norbs - Nocc)

    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False)

    hTensor.getShapeActive(mf.nelec, Norbs)
    gTensor.getShapeActive(mf.nelec, Norbs)
    gTensor.array = eri.reshape((Norbs, Norbs, Norbs, Norbs)).swapaxes(2,3).swapaxes(1,2)
    fock = h1
    for p in range(Norbs):
        for q in range(Norbs):
            fock[p,q] += sum([2 * gTensor.array[p,i,q,i] - gTensor.array[p,i,i,q] for i in range(Nocc)])
    #        fock[p,q] += sum([2 * gTensor.array[p,i,q,i] - gTensor.array[p,i,i,q] for i in range(Nocc,Nocc+int(Nactive/2))])
    hTensor.array = fock

    # print("core-core", fock[:Nocc,:Nocc])
    # print("active-active", fock[Nocc:Nocc+Nactive, Nocc:Nocc+Nactive])
    # print("virtual-virtual", fock[Nocc+Nactive:, Nocc+Nactive:])

    if Nactive != 0:
        dm = mf.make_rdm1()[0,mf.nelec[1]:mf.nelec[0], mf.nelec[1]:mf.nelec[0]] + mf.make_rdm1()[1,mf.nelec[1]:mf.nelec[0], mf.nelec[1]:mf.nelec[0]]
    #    print(dm)
        for p in range(Norbs):
            for q in range(Norbs):
                hTensor.array[p,q] += sum([dm[u,v] * gTensor.array[p,u,q,v] - 0.5 * dm[u,v] * gTensor.array[p,u,v,q] for v in range(mf.nelec[1]-mf.nelec[0]) for u in range(mf.nelec[1]-mf.nelec[0])])

    gTensor.assignDiagramArraysActive(Nocc, Nactive, Nvirtual)
    hTensor.assignDiagramArraysActive(Nocc, Nactive, Nvirtual)

    for tTensor in amplitudeTensors:
        tTensor.getShapeActive(mf.nelec, Norbs)
        tTensor.assignDiagramArraysActive(Nocc, Nactive, Nvirtual)

    for amplitudeTensor in amplitudeTensors:
        amplitudeTensor.array = np.zeros_like(amplitudeTensor.array)

    IntermediateNorm = energyEquationAndNorm[1]
    EnergyEquationsByOrder = tuple(tensor.TensorSum([summand for summand in energyEquationAndNorm[0].summandList if len(summand.tensorList) == order+1]) for order in range(maxOrder+1))
    EnergyContributionsByOrder = [contractions.contractTensorSum(equation, lowerIndexList=[], upperIndexList=[]) for equation in EnergyEquationsByOrder]
    Energy = sum(EnergyContributionsByOrder)
    Energy0 = Energy
    print("Reference energy relative to vacuum:", Energy)
    print("Intermediate normalization check:", contractions.contractTensorSum(IntermediateNorm, lowerIndexList=[], upperIndexList=[]))

    return Energy, IntermediateNorm
