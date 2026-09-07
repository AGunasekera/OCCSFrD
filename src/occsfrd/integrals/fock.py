from pyscf import ao2mo
from occsfrd.wick import tensor, contractions

def buildFock(mf, equationsDict, levelShift=0., verbosity=0, biorthogonal=False, Rtol=10, Etol=8, maxIter=100, nDIIS=0, maxOrder=2, onlyConnect=False):

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

    return hTensor, gTensor