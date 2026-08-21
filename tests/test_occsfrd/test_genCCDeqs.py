from occsfrd import ansatz, wick, interface

hTensor = wick.tensor.Tensor("h", ['g'], ['g'])
hTensor.getAllDiagramsActive()
gTensor = wick.tensor.Tensor("g", ['g', 'g'], ['g', 'g'])
gTensor.getAllDiagramsActive()
Hamiltonian = sum(hTensor.diagrams) + 0.5 * sum(gTensor.diagrams)

t2Tensor = wick.tensor.Tensor("{t_{2}}", ['p', 'p'], ['h', 'h'])

t2Tensor.getAllDiagramsActive()
amplitudeTensors = [t2Tensor]
amplitudeDiagrams = t2Tensor.diagrams
T = 0.5 * sum(t2Tensor.diagrams)

expT = ansatz.utils.operatorExponential(T, trunc=1)
normalisationCheck = ansatz.normalorderedcc.getEnergyEquation(expT)
energyEquation = ansatz.normalorderedcc.getEnergyEquation(Hamiltonian * expT)
amplitudeEquations = [ansatz.normalorderedcc.getAmplitudeEquation_UnlinkedFormalism(Hamiltonian, expT, tDiagram) for tDiagram in amplitudeDiagrams]

equations = [(energyEquation, normalisationCheck)] + amplitudeEquations