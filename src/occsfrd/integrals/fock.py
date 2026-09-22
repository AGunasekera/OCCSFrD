import numpy as np
from pyscf import ao2mo
from occsfrd.wick import tensor, contractions

def putIntegralsIntoEquation(hArray, gArray, nElec, nOrbs, equationsDict):
    '''
    Take the arrays of one-electron and two-electron integrals and put them into the tensor objects in equationsDict
    '''
    hTensor = equationsDict["tensors"][0]
    gTensor = equationsDict["tensors"][1]

    hTensor.getShapeActive(nElec, nOrbs)
    gTensor.getShapeActive(nElec, nOrbs)

    hTensor.setArray(hArray)
    gTensor.setArray(gArray)

def initialiseAmplitudeTensors(nElec, nOrbs, nOcc, nActive, nVirtual, equationsDict):
    '''
    Take the arrays of one-electron and two-electron integrals and put them into the tensor objects in equationsDict
    '''
    amplitudeTensors = equationsDict["tensors"][2:]

    for tTensor in amplitudeTensors:
        tTensor.getShapeActive(nElec, nOrbs)
        tTensor.assignDiagramArraysActive(nOcc, nActive, nVirtual)

    for amplitudeTensor in amplitudeTensors:
        amplitudeTensor.array = np.zeros_like(amplitudeTensor.array)

def buildFock(mf, equationsDict, frozenCore=None):
    '''
    Build the one-electron and two-electron integrals from a pyscf mean-field object.
    Implement a frozen core approximation if specified
    '''
# def runUnlinkedCC(mf, equationsDict, levelShift=False):
    mol = mf.mol
#    equationsDict = storeequations.load(equationFileName)

    specificIndices = equationsDict["specificIndices"]
    
    Norbs = mol.nao
    nElec = mf.nelec
    Nocc = nElec[1]
    Nactive = len(specificIndices)
    Nvirtual = Norbs - Nactive - Nocc
    vacuum = [1] * Nocc + [0] * (Norbs - Nocc)

    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False)

    gArray = eri.reshape((Norbs, Norbs, Norbs, Norbs)).swapaxes(2,3).swapaxes(1,2)
    fock = h1
    for p in range(Norbs):
        for q in range(Norbs):
            fock[p,q] += sum([2 * gArray[p,i,q,i] - gArray[p,i,i,q] for i in range(Nocc)])

    if Nactive != 0:
        dm = mf.make_rdm1()[0,mf.nelec[1]:mf.nelec[0], mf.nelec[1]:mf.nelec[0]] + mf.make_rdm1()[1,mf.nelec[1]:mf.nelec[0], mf.nelec[1]:mf.nelec[0]]
    #    print(dm)
        for p in range(Norbs):
            for q in range(Norbs):
                fock[p,q] += sum([dm[u,v] * gArray[p,u,q,v] - 0.5 * dm[u,v] * gArray[p,u,v,q] for v in range(mf.nelec[1]-mf.nelec[0]) for u in range(mf.nelec[1]-mf.nelec[0])])

    if frozenCore is not None:
        Norbs -= frozenCore
        Nocc -=frozenCore

        fock = fock[frozenCore:,frozenCore:]
        gArray = gArray[frozenCore:,frozenCore:,frozenCore:,frozenCore:]

        nElec = (nElec[0] - frozenCore, nElec[1] - frozenCore)

    putIntegralsIntoEquation(fock, gArray, nElec, Norbs, equationsDict)
    initialiseAmplitudeTensors(nElec, Norbs, Nocc, Nactive, Nvirtual, equationsDict)