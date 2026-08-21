from pyscf import gto, ao2mo, scf, fci, mp, cc
from occsfrd import ansatz, wick, interface, solve

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
interface.storeequations.save("test_N0S0equations", equations, [hTensor, gTensor] + amplitudeTensors)

bohr = 0.529177249

H2sep = 1.605 * bohr

mol = gto.Mole()
mol.verbose = 1
mol.atom = 'Ne 0 0 0'
# mol.atom = 'N 0 0 0; N 0 0 1.1'
mol.spin = 0
# mol.atom = 'Li 0 0 0'
# mol.atom = 'C 0 0 0; H 1.09 0 0; H -0.545 0.944 0; H -0.545 -0.944 0'
# mol.spin = 1
# mol.atom = 'Be 0 0 0'
# mol.atom = 'O 0 0 0; O 0 0 1.1'
# mol.spin = 2
# mol.atom = 'B 0 0 0'
# mol.spin = 3
# mol.basis = 'sto-3g'
mol.basis = '6-31g'
# mol.basis = 'cc-pcvtz'
mol.build()

Enuc = mol.energy_nuc()

mf = scf.ROHF(mol)
mf.kernel()

printFCIComparison = True
if printFCIComparison:
    cisolver = fci.FCI(mol, mf.mo_coeff)
    cisolver.kernel()
    print("FCI energy", cisolver.e_tot)
    print("ROHF Energy", mf.e_tot)
    print("Correlation Energy", cisolver.e_tot - mf.e_tot)

# equationsDict = interface.storeequations.load("/home/dpt02/dpt/iclb0552/code/OpenShellCC/equations/UnlinkedNormalOrdered/CCD/linear/collectingTerms/N0S0equationsCollected")
# equationsDict = interface.storeequations.load("/home/dpt02/dpt/iclb0552/code/OpenShellCC/equations/UnlinkedNormalOrdered/CCD/linear/collectingTerms082023/N1S0.5equationsCollected")
# equationsDict = interface.storeequations.load("/home/dpt02/dpt/iclb0552/code/OpenShellCC/equations/UnlinkedNormalOrdered/CCD/linear/N3S1.5equations")
equationsDict = interface.storeequations.load("test_N0S0equations")

linearCCD = solve.cc.runUnlinkedCC(mf, equationsDict, levelShift=0, verbosity=1, Etol=10, Rtol=6, maxIter=1000)