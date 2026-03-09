from copy import copy
from time import time
from occsfrd import wick
from . import utils

def projectionManifold(amplitudeTensor):
    """
    Take an amplitude (excitation) tensor and return the corresponding projection (de-excitation) tensor
    
    Args:
        amplitudeTensor (wick.tensor.Tensor): the amplitude tensor whose conjugate is being projected onto
    
    Returns:
        wick.tensor.Tensor: projection tensor
    """
    # assert amplitudeTensor.name[:2] == '{t'
    if amplitudeTensor.name[:2] == '{t':
        newName = '{\Phi' + amplitudeTensor.name[2:]
    else:
        newName = '{\Phi}'

    return wick.tensor.Tensor(newName, amplitudeTensor.upperIndexTypes, amplitudeTensor.lowerIndexTypes, amplitudeTensor.spinFree, distinguishableParticles=True)

def getEnergyEquation(transformedHamiltonian, referenceOperator=None, spinFree=True, verbose=True):
    """
    Get energy equation in an open shell coupled cluster ansatz

    Args:
        transformedHamiltonian (wick.tensor.TensorSum): The tensor expression for the similarity-transformed Hamiltonian, or more generally the Hamiltonian multiplied by wave operator
        referenceOperator (wick.operator.OperatorSum, optional): The operator creating the reference CSF wavefunction out of the Fermi vacuum. Defaults to None.
        spinFree (bool, optional): Whether or not the tensors represent operators that are spin-free. Defaults to True.
        verbose (bool, optional): Whether or not to print time taken and number of terms in equation. Defaults to True.

    Returns:
        wick.operator.OperatorSum: The energy equation
    """
#    return wick.contractions.evaluateWick(transformedHamiltonian, referenceOperator=referenceOperator).collectIsomorphicTerms()
    t0 = time()
    equation = wick.contractions.evaluateWick(transformedHamiltonian, referenceOperator=referenceOperator)
    t1 = time()
    if verbose:
        print("Time to find energy equation:", t1 - t0)
        print(equation)
        print("number of terms:", len(equation.summandList))
    return equation

def getEnergyEquationUnlinked(HamiltonianAndWaveOperator, referenceOperator=None, spinFree=True, verbose=True):
    t0 = time()
    equation = wick.contractions.evaluateWick(HamiltonianAndWaveOperator, referenceOperator=referenceOperator).collectConnectedIsomorphicTerms()
    t1 = time()
    if verbose:
        print("Time to find energy equation:", t1 - t0)
        print(equation)
        print("number of terms:", len(equation.summandList))
    return equation

def getAmplitudeEquation(transformedHamiltonian, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    t0 = time()
    projection = projectionManifold(amplitudeTensor)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * transformedHamiltonian, referenceOperator=referenceOperator)
#    amplitudeEquation = projectedAmplitudeEquation.collectIsomorphicTerms()
    amplitudeEquation = projectedAmplitudeEquation
    # for summand in amplitudeEquation.summandList:
    #     summand.tensorList.pop(0)
    #     projectionVertex = summand.vertexList.pop(0)
    #     summand.freeLowerIndices = projectionVertex.upperIndices
    #     summand.freeUpperIndices = projectionVertex.lowerIndices
    utils.deProjectEquation(amplitudeEquation)
    t1 = time()
    if verbose:
        print("Amplitude:", amplitudeTensor)
        print("Time to find amplitude equation:", t1 - t0)
        print(amplitudeEquation)
        print("number of terms:", len(amplitudeEquation.summandList))
    return amplitudeEquation

def getAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    """
    Generate amplitude equation in unlinked formalism and separate linked component of renormalisation terms.

    Generate the linked and unlinked parts of the coupled cluster amplitude equations, as defined by the given Hamiltonian, wave operator, and amplitudes.
        
    Args:
        Hamiltonian           (wick.tensor.TensorSum): the sum of tensors corresponding to the operators that constitute the Hamiltonian
        waveOperator          (wick.tensor.TensorSum): the wave operator for the chosen ansatz, expressed as the sum of products of the corresponding coefficient tensors
        amplitudeTensor          (wick.tensor.Tensor): the amplitude tensor for the part of the cluster operator in this ansatz
        referenceOperator (wick.operator.OperatorSum): the second-quantised operator generating the open-shell reference out of the Fermi vacuum
        spinFree                               (bool): whether or not the CC equations are being prepared in orbtal basis, with spin summation
    
    Returns:
        wick.tensor.TensorSum: the part of the CC equations that includes the Hamiltonian
        wick.tensor.TensorSum: the unlinked part of the CC equations that will be multiplied by the energy in the unlinked CC equations.
    """
    t0 = time()
    projection = projectionManifold(amplitudeTensor)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * Hamiltonian * waveOperator, referenceOperator=referenceOperator)
    projectedUnlinkedPart = wick.contractions.evaluateWick(projection * waveOperator, referenceOperator=referenceOperator)
#    amplitudeEquation = projectedAmplitudeEquation.collectIsomorphicTerms()
    amplitudeEquation = projectedAmplitudeEquation
    # for summand in amplitudeEquation.summandList:
    #     summand.tensorList.pop(0)
    #     projectionVertex = summand.vertexList.pop(0)
    #     summand.freeLowerIndices = projectionVertex.upperIndices
    #     summand.freeUpperIndices = projectionVertex.lowerIndices
    utils.deProjectEquation(amplitudeEquation)
    unlinkedPart = projectedUnlinkedPart
    # for summand in unlinkedPart.summandList:
    #     summand.tensorList.pop(0)
    #     projectionVertex = summand.vertexList.pop(0)
    #     summand.freeLowerIndices = projectionVertex.upperIndices
    #     summand.freeUpperIndices = projectionVertex.lowerIndices
    utils.deProjectEquation(unlinkedPart)
    t1 = time()
    if verbose:
        print("Amplitude:", amplitudeTensor)
        print("Time to find amplitude equation:", t1 - t0)
        print(amplitudeEquation)
        print("number of terms:", len(amplitudeEquation.summandList))
        print("Unlinked part")
        print(unlinkedPart)
        print("number of terms:", len(unlinkedPart.summandList))
    return utils.cleanEquation(amplitudeEquation, 10), utils.cleanEquation(unlinkedPart, 10)

def getBiorthAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    t0 = time()
    projection = projectionManifold(amplitudeTensor)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * Hamiltonian * waveOperator, referenceOperator=referenceOperator)
    projectedUnlinkedPart = wick.contractions.evaluateWick(projection * waveOperator, referenceOperator=referenceOperator)
#    amplitudeEquation = projectedAmplitudeEquation.collectIsomorphicTerms()
    amplitudeEquation = projectedAmplitudeEquation
    # for summand in amplitudeEquation.summandList:
    #     summand.tensorList.pop(0)
    #     projectionVertex = summand.vertexList.pop(0)
    #     summand.freeLowerIndices = projectionVertex.upperIndices
    #     summand.freeUpperIndices = projectionVertex.lowerIndices
    unlinkedPart = projectedUnlinkedPart
    # for summand in unlinkedPart.summandList:
    #     summand.tensorList.pop(0)
    #     projectionVertex = summand.vertexList.pop(0)
    #     summand.freeLowerIndices = projectionVertex.upperIndices
    #     summand.freeUpperIndices = projectionVertex.lowerIndices
    biorthProjectedAmplitudeEquation = 1./6. * copy(projectedAmplitudeEquation)
    for summand in biorthProjectedAmplitudeEquation.summandList:
        summand.vertexList[0].lowerIndices[0], summand.vertexList[0].lowerIndices[1] = summand.vertexList[0].lowerIndices[1], summand.vertexList[0].lowerIndices[0]
    biorthProjectedAmplitudeEquation = biorthProjectedAmplitudeEquation + 1./3. * projectedAmplitudeEquation
    amplitudeEquation = biorthProjectedAmplitudeEquation
    biorthProjectedUnlinkedPart = 1./6. * copy(projectedUnlinkedPart)
    for summand in biorthProjectedUnlinkedPart.summandList:
        summand.vertexList[0].lowerIndices[0], summand.vertexList[0].lowerIndices[1] = summand.vertexList[0].lowerIndices[1], summand.vertexList[0].lowerIndices[0]
    biorthProjectedUnlinkedPart = biorthProjectedUnlinkedPart + 1./3. * projectedUnlinkedPart
    unlinkedPart = biorthProjectedUnlinkedPart
    utils.deProjectEquation(amplitudeEquation)
    utils.deProjectEquation(unlinkedPart)
    t1 = time()
    if verbose:
        print("Amplitude:", amplitudeTensor)
        print("Time to find amplitude equation:", t1 - t0)
        print(amplitudeEquation)
        print("number of terms:", len(amplitudeEquation.summandList))
        print("Unlinked part")
        print(unlinkedPart)
        print("number of terms:", len(unlinkedPart.summandList))
    return utils.cleanEquation(amplitudeEquation, 10), utils.cleanEquation(unlinkedPart, 10)

def getCollectedBiorthogonalAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    t0 = time()
    projection = projectionManifold(amplitudeTensor)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * Hamiltonian * waveOperator, referenceOperator=referenceOperator)
    biorthProjectedAmplitudeEquation = 1./6. * copy(projectedAmplitudeEquation)
    for summand in biorthProjectedAmplitudeEquation.summandList:
        summand.vertexList[0].lowerIndices[0], summand.vertexList[0].lowerIndices[1] = summand.vertexList[0].lowerIndices[1], summand.vertexList[0].lowerIndices[0]
    biorthProjectedAmplitudeEquation = biorthProjectedAmplitudeEquation + 1./3. * projectedAmplitudeEquation
    amplitudeEquation = biorthProjectedAmplitudeEquation
    collectedAmplitudeEquation = utils.cleanEquation(amplitudeEquation.collectIsomorphicTerms(), 10)
    utils.deProjectEquation(amplitudeEquation)
    utils.deProjectEquation(collectedAmplitudeEquation)

    projectedUnlinkedPart = wick.contractions.evaluateWick(projection * waveOperator, referenceOperator=referenceOperator)
    biorthProjectedUnlinkedPart = 1./6. * copy(projectedUnlinkedPart)
    for summand in biorthProjectedUnlinkedPart.summandList:
        summand.vertexList[0].lowerIndices[0], summand.vertexList[0].lowerIndices[1] = summand.vertexList[0].lowerIndices[1], summand.vertexList[0].lowerIndices[0]
    biorthProjectedUnlinkedPart = biorthProjectedUnlinkedPart + 1./3. * projectedUnlinkedPart
    unlinkedPart = biorthProjectedUnlinkedPart
    collectedUnlinkedPart = utils.cleanEquation(unlinkedPart.collectIsomorphicTerms(), 10)
    utils.deProjectEquation(unlinkedPart)
    utils.deProjectEquation(collectedUnlinkedPart)
    t1 = time()
    if verbose:
        print("Amplitude:", amplitudeTensor)
        print("Time to find amplitude equation:", t1 - t0)
        print(collectedAmplitudeEquation)
        print("number of terms:", len(collectedAmplitudeEquation.summandList))
        print("Unlinked part")
        print(collectedUnlinkedPart)
        print("number of terms:", len(collectedUnlinkedPart.summandList))
    return {"Uncollected": (utils.cleanEquation(amplitudeEquation, 10), utils.cleanEquation(unlinkedPart, 10)), "Collected": (collectedAmplitudeEquation, collectedUnlinkedPart)}

def getCollectedAmplitudeEquation_UnlinkedFormalism(Hamiltonian, waveOperator, amplitudeTensor, referenceOperator=None, spinFree=True, verbose=True):
    t0 = time()
    projection = projectionManifold(amplitudeTensor)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * Hamiltonian * waveOperator, referenceOperator=referenceOperator)
    amplitudeEquation = projectedAmplitudeEquation
    collectedAmplitudeEquation = utils.cleanEquation(amplitudeEquation.collectIsomorphicTerms(), 10)
    utils.deProjectEquation(amplitudeEquation)
    utils.deProjectEquation(collectedAmplitudeEquation)

    projectedUnlinkedPart = wick.contractions.evaluateWick(projection * waveOperator, referenceOperator=referenceOperator)
    unlinkedPart = projectedUnlinkedPart
    collectedUnlinkedPart = utils.cleanEquation(unlinkedPart.collectIsomorphicTerms(), 10)
    utils.deProjectEquation(unlinkedPart)
    utils.deProjectEquation(collectedUnlinkedPart)
    t1 = time()
    if verbose:
        print("Amplitude:", amplitudeTensor)
        print("Time to find amplitude equation:", t1 - t0)
        print(collectedAmplitudeEquation)
        print("number of terms:", len(collectedAmplitudeEquation.summandList))
        print("Unlinked part")
        print(collectedUnlinkedPart)
        print("number of terms:", len(collectedUnlinkedPart.summandList))
    return {"Uncollected": (utils.cleanEquation(amplitudeEquation, 10), utils.cleanEquation(unlinkedPart, 10)), "Collected": (utils.cleanEquation(collectedAmplitudeEquation, 10), utils.cleanEquation(collectedUnlinkedPart, 10))}

def getAmplitudeEquationOnlyLinked(HamiltonianAndWaveOperator, amplitudeTensor, referenceOperator=None, spinFree=True):
    projection = projectionManifold(amplitudeTensor)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * HamiltonianAndWaveOperator, referenceOperator=referenceOperator)
    amplitudeEquation = projectedAmplitudeEquation.collectConnectedIsomorphicTerms()
    # for summand in amplitudeEquation.summandList:
    #     summand.tensorList.pop(0)
    #     projectionVertex = summand.vertexList.pop(0)
    #     summand.freeLowerIndices = projectionVertex.upperIndices
    #     summand.freeUpperIndices = projectionVertex.lowerIndices
    utils.deProjectEquation(amplitudeEquation)
    return amplitudeEquation

def getBiorthDoublesAmplitudeEquationOnlyLinked(HamiltonianAndWaveOperator, doublesAmplitudeTensor, referenceOperator=None, spinFree=True):
    projection = projectionManifold(doublesAmplitudeTensor)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * HamiltonianAndWaveOperator, referenceOperator=referenceOperator)
    biorthProjectedAmplitudeEquation = 1./6. * copy(projectedAmplitudeEquation)
    for summand in biorthProjectedAmplitudeEquation.summandList:
        summand.vertexList[0].lowerIndices[0], summand.vertexList[0].lowerIndices[1] = summand.vertexList[0].lowerIndices[1], summand.vertexList[0].lowerIndices[0]
    biorthProjectedAmplitudeEquation = biorthProjectedAmplitudeEquation + 1./3. * projectedAmplitudeEquation
    amplitudeEquation = biorthProjectedAmplitudeEquation.collectIsomorphicTerms()
    # for summand in amplitudeEquation.summandList:
    #     summand.tensorList.pop(0)
    #     projectionVertex = summand.vertexList.pop(0)
    #     summand.freeLowerIndices = projectionVertex.upperIndices
    #     summand.freeUpperIndices = projectionVertex.lowerIndices
    utils.deProjectEquation(amplitudeEquation)
    return utils.cleanEquation(amplitudeEquation, 10)

def genNormalOrderedCCAnsatz(amplitudeTensorsList, order=2):
    """
    Generate the normal-ordered coupled cluster ansatz for an open shell system as tensor products.

    Taking the expectation value of this and setting to 0 yields a set of coupled cluster equations.
    
    Args:
        amplitudeTensorsList (list): list of amplitude tensors included in exponential ansatz

    Returns:
        wick.tensor.TensorSum: The sum of tensor products corresponding to the ansatz to be projected onto each excitation manifold
        wick.tensor.Tensor: Fock matrix as tensor object
        wick.tensor.Tensor: 2-particle interaction tensor
    """
    fockTensor = wick.tensor.Tensor("f", ['g'], ['g'])
    h2Tensor = wick.tensor.Tensor("v", ['g', 'g'], ['g', 'g'])

    fockTensor.getAllDiagramsActive()
    h2Tensor.getAllDiagramsActive()
    for amplitudeTensor in amplitudeTensorsList:
        amplitudeTensor.getAllDiagramsActive()

    normalOrderedBOHamiltonian = sum(fockTensor.diagrams) + (1. / 2.) * sum(h2Tensor.diagrams)

    T0 = sum([(1. / amplitudeTensor.excitationRank) * sum(amplitudeTensor.diagrams) for amplitudeTensor in amplitudeTensorsList if utils.isParticleHoleExcitation(amplitudeTensor)])
    T1 = sum([sum(amplitudeTensor.diagrams) for amplitudeTensor in amplitudeTensorsList if not utils.isParticleHoleExcitation(amplitudeTensor)])

    transformedHamiltonian = normalOrderedBOHamiltonian * utils.operatorExponential(T0 + T1, order)
    return transformedHamiltonian, fockTensor, h2Tensor