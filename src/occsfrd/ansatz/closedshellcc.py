from copy import copy
from occsfrd import wick
from occsfrd.ansatz import utils

def amplitude(excitationLevel):
    """
    Closed-shell amplitude tensor of a given (particle-hole) excitation level
    
    Args:
        excitationLevel (int): Excitation level
    
    Returns:
        wick.tensor.Tensor: Amplitude tensor of the given excitation level
    """
    return wick.tensor.Tensor('t', ['p'] * excitationLevel, ['h'] * excitationLevel)

def projectionManifold(excitationLevel):
    """
    Projection manifold of a given excitation level for closed shell CC

    Args:
        excitationLevel (int): Excitation level
    
    Returns:
        wick.tensor.Tensor: Projection (de-excitation) tensor
    """
    return wick.tensor.Tensor('\Phi', ['h'] * excitationLevel, ['p'] * excitationLevel)

def getEnergyEquation(similarityTransformedHamiltonian, spinFree=True):
    """
    Get the energy equation for a linked coupled cluster ansatz
    as given by the expectation value of the similarity transformed Hamiltonian.

    Args:
        similarityTransformedHamiltonian (wick.Tensor.TensorSum): The similarity transformed Hamiltonian
    
    Returns:
        wick.operator.TensorSum: Expectation value of the similarity transformed Hamiltonian. Each term is a scalar with complete contractions.
    """
#    return evaluateWick(similarityTransformedHamiltonian, spinFree).collectIsomorphicTerms()
    return wick.contractions.evaluateWick(similarityTransformedHamiltonian).collectIsomorphicTerms()

def getAmplitudeEquation(similarityTransformedHamiltonian, excitationLevel, spinFree=True):
    """
    Get the projected amplitude equation for a given excitation level in a linked coupled cluster ansatz
    as given by the expectation value of the similarity transformed Hamiltonian.

    Args:
        similarityTransformedHamiltonian (wick.Tensor.TensorSum): The similarity transformed Hamiltonian
        excitationLevel (int): Excitation level
    
    Returns:
        wick.operator.TensorSum: The terms in the amplitude equation of the given excitation level.
    """
    projection = projectionManifold(excitationLevel)
#    projectedAmplitudeEquation = evaluateWick(projection * similarityTransformedHamiltonian, spinFree)
    projectedAmplitudeEquation = wick.contractions.evaluateWick(projection * similarityTransformedHamiltonian)
    amplitudeEquation = projectedAmplitudeEquation.collectIsomorphicTerms()
    for summand in amplitudeEquation.summandList:
        summand.tensorList.pop(0)
        summand.vertexList.pop(0)
    return amplitudeEquation

def getEnergyEquationNew(similarityTransformedHamiltonian, spinFree=True):
    """
    Generate energy equation (new)

    A new routine (using the updated wick contraction evaluation routine) to generate energy equations from a provided similarity-transformed Hamiltonian.

    Args:
        similarityTransformedHamiltonian (wick.tensor.TensorSum): The similarity-transformed Hamiltonian
        spinFree (bool, optional): Whether or not the tensors are to be interpreted as representing spin-free operators. Defaults to True.

    Returns:
        wick.operator.OperatorSum: The non-zero contributions to the energy, which are the surviving completely contracted terms
    """
#    return evaluateWick(similarityTransformedHamiltonian, spinFree).collectIsomorphicTerms()
    return wick.contractions.evaluateWickNew(similarityTransformedHamiltonian)#.collectIsomorphicTerms()

def getAmplitudeEquationNew(similarityTransformedHamiltonian, excitationLevel, spinFree=True):
    """
    Generate energy equation (new)

    A new routine (using the updated wick contraction evaluation routine) to generate the projected amplitude equation for a given excitation level in a linked coupled cluster ansatz as given by the expectation value of the similarity transformed Hamiltonian.
    Does not attempt to identify and collate isomorphic terms
    
    Args:
        similarityTransformedHamiltonian (wick.Tensor.TensorSum): The similarity transformed Hamiltonian
        excitationLevel (int): Excitation level of the amplitude equation
        spinFree (bool, optional): Whether or not the tensors are to be interpreted as representing spin-free operators. Defaults to True.
    
    Returns:
        wick.operator.TensorSum: The terms in the amplitude equation of the given excitation level.
    """
    projection = projectionManifold(excitationLevel)
#    projectedAmplitudeEquation = evaluateWick(projection * similarityTransformedHamiltonian, spinFree)
    projectedAmplitudeEquation = wick.contractions.evaluateWickNew(projection * similarityTransformedHamiltonian)
    # amplitudeEquation = projectedAmplitudeEquation#.collectIsomorphicTerms()
    # for summand in amplitudeEquation.summandList:
    #     summand.tensorList.pop(0)
    #     summand.vertexList.pop(0)
    return projectedAmplitudeEquation

def getBiorthogonalSpinFreeDoublesEquation(similarityTransformedHamiltonian):
    """
    Get the biorthogonalised projected amplitude equation for double excitations.

    Args:
        similarityTransformedHamiltonian (wick.Tensor.TensorSum): The similarity transformed Hamiltonian
    
    Returns:
        wick.operator.TensorSum: Biorthogonalised amplitude equation.
    """
#    projectedDoublesAmplitudeEquation = evaluateWick(projectionManifold(2) * similarityTransformedHamiltonian, spinFree=True)
    projectedDoublesAmplitudeEquation = wick.contractions.evaluateWick(projectionManifold(2) * similarityTransformedHamiltonian)
    exchangeProjectedDoublesAmplitudeEquation = copy(projectedDoublesAmplitudeEquation)
    for term in exchangeProjectedDoublesAmplitudeEquation.summandList:
        p0 = term.vertexList[0].upperIndices[0]
        p1 = term.vertexList[0].upperIndices[1]
        for vertex in term.vertexList[1:]:
            for lI, lowerIndex in enumerate(vertex.lowerIndices):
                if lowerIndex == p0:
                    vertex.lowerIndices[lI] = p1
                elif lowerIndex == p1:
                    vertex.lowerIndices[lI] = p0
    thirdProjectedDoublesAmplitudeEquation = copy(projectedDoublesAmplitudeEquation)
    sixthExchangeProjectedDoublesAmplitudeEquation = copy(exchangeProjectedDoublesAmplitudeEquation)
    for term in thirdProjectedDoublesAmplitudeEquation.summandList:
        term.prefactor /= 3.
    for term in sixthExchangeProjectedDoublesAmplitudeEquation.summandList:
        term.prefactor /= 6.
    return utils.deProjectEquation((thirdProjectedDoublesAmplitudeEquation + sixthExchangeProjectedDoublesAmplitudeEquation).collectIsomorphicTerms())

def genClosedShellCCAnsatz(excitationLevels, trunc=4):
    """
    Generate the coupled cluster ansatz for a closed shell system as tensor products.
    Taking the expectation value of this and setting to 0 yields the coupled cluster equations.
    
    Args:
        excitationLevels (list) or (int): List of excitation levels included in the CC amplitudes. If int provided, use all levels up to and including excitationLevels

    Returns:
        wick.tensor.TensorSum: The BCH expansion for the similarity transformed Hamiltonian as a sum of tensor products corresponding to the ansatz to be projected onto each excitation manifold
    """
    if isinstance(excitationLevels, int):
        excitationLevels = list(range(1, excitationLevels+1))

    fockTensor = wick.tensor.Tensor("f", ['g'], ['g'])
    h2Tensor = wick.tensor.Tensor("v", ['g', 'g'], ['g', 'g'])

    fockTensor.getAllDiagramsGeneral()
    h2Tensor.getAllDiagramsGeneral()

    normalOrderedBOHamiltonian = sum(fockTensor.diagrams) + (1. / 2.) * sum(h2Tensor.diagrams)

    amplitudes = [amplitude(excitationLevel) for excitationLevel in excitationLevels]

    T = sum((1. / amp.excitationRank) * amp for amp in amplitudes)
    BCH = utils.BCHSimilarityTransform(normalOrderedBOHamiltonian, T, trunc)
    return BCH, fockTensor, h2Tensor, amplitudes
#
#    products = [BCH] + [(projectionManifold(excitationLevel) * BCH) for excitationLevel in excitationLevels]
#    return tuple(products)