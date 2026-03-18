import sys
import requests
import pickle
from pyscf import gto, ao2mo, scf, fci, mp, cc
from openshellcc import wick, ansatz, solve, interface

assert (len(sys.argv) == 2) and isinstance(sys.argv[1], str)

ACCESS_TOKEN = sys.argv[1]
headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
deposition_url = 'https://sandbox.zenodo.org/api/deposit/depositions'
which_deposition = 0
r = requests.get(deposition_url, headers=headers)
r = requests.get("%s/%s" % (deposition_url, r.json()[which_deposition]['id']), headers=headers)
fNamesAndDownloadLinks = [(file['filename'], file['links']['download']) for file in r.json()['files']]

eqnFileName = "N0S0_qNOECCSD.eqn"

for name, link in fNamesAndDownloadLinks:
    if name == eqnFileName:
        with open(name[:-3]+"pkl", 'wb') as f:
            r = requests.get(link, headers=headers)
            f.write(r.content)
        
equationsDict = interface.storeequations.load(eqnFileName[:-4])
print(equationsDict)
# equationsDict = interface.storeequations.load("/home/dpt02/dpt/iclb0552/code/OpenShellCC/equations/UnlinkedNormalOrdered/CCSD/linear/N0S0equations")

print("CLOSED SHELL\n")

bohr = 0.529177249

H2sep = 1.605 * bohr

mol = gto.Mole()
mol.verbose = 1
mol.atom = '''
Ne 	0.0000 	0.0000 	0.0000
'''
mol.spin = 0
mol.basis = 'ccpvdz'
mol.build()

Enuc = mol.energy_nuc()

mf = scf.ROHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

print("Nuclear repulsion energy:", Enuc, "ROHF electronic energy:", mf.energy_elec(), "Total ROHF energy:", mf.e_tot)

printTrueCCSD = True
if printTrueCCSD:
    trueCCSD = cc.CCSD(mf)
    trueCCSD.diis=False
    trueCCSD.conv_tol = 1e-16
    trueCCSD.kernel()
    print("Spin-orbital CCSD correlation energy (PySCF):", trueCCSD.e_corr)

printFCIComparison = False
if printFCIComparison:
    cisolver = fci.FCI(mol, mf.mo_coeff)
    cisolver.kernel()
    print("FCI energy", cisolver.e_tot)
    print("ROHF Energy", mf.e_tot)
    print("Correlation Energy", cisolver.e_tot - mf.e_tot)

qCCSD = solve.cc.runUnlinkedCC(mf, equationsDict, levelShift=0, verbosity=0, Rtol=16, Etol=12, maxIter=1000, nDIIS=12, maxOrder=2)
print("Total CCSD energy:", Enuc + qCCSD["total electronic energy"])