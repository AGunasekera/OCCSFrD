import pytest
import numpy as np
from pyscf import gto, scf, cc
import sys, requests
import occsfrd

@pytest.fixture
def h2o_sto3g_pyscf_mol():
    mol = gto.Mole()
    mol.verbose = 1
    mol.atom = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
    mol.basis = 'sto3g'
    mol.spin = 0
    mol.build()

    return mol

@pytest.fixture
def h2o_631g_pyscf_mol():
    mol = gto.Mole()
    mol.verbose = 1
    mol.atom = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()

    return mol

@pytest.fixture
def h2o_sto3g_pyscf_rhf(h2o_sto3g_pyscf_mol):
    mf = scf.ROHF(h2o_sto3g_pyscf_mol)
    mf.kernel()

    return mf

@pytest.fixture
def h2o_631g_pyscf_rhf(h2o_631g_pyscf_mol):
    mf = scf.ROHF(h2o_631g_pyscf_mol)
    mf.kernel()

    return mf

@pytest.fixture
def h2o_sto3g_pyscf_ccsd(h2o_sto3g_pyscf_rhf):
    ccsd = cc.CCSD(h2o_sto3g_pyscf_rhf)
    ccsd.diis = False
    ccsd.conv_tol = 1e-10
    ccsd.kernel()

    return ccsd

@pytest.fixture
def h2o_631g_pyscf_ccsd(h2o_631g_pyscf_rhf):
    ccsd = cc.CCSD(h2o_631g_pyscf_rhf)
    ccsd.diis = False
    ccsd.conv_tol = 1e-10
    ccsd.kernel()

    return ccsd

@pytest.fixture
def h2o_sto3g_pyscf_ccd(h2o_sto3g_pyscf_rhf):
    ccd = cc.CCSD(h2o_sto3g_pyscf_rhf)
    old_update_amps = ccd.update_amps
    def update_amps(t1, t2, eris):
        t1, t2 = old_update_amps(t1, t2, eris)
        return (np.zeros_like(t1[0]), np.zeros_like(t1[1])), t2
    ccd.update_amps = update_amps
    ccd.conv_tol = 1e-10
    ccd.diis = False
    ccd.kernel()

    return ccd

@pytest.fixture
def h2o_631g_pyscf_ccd(h2o_631g_pyscf_rhf):
    ccd = cc.CCSD(h2o_631g_pyscf_rhf)
    old_update_amps = ccd.update_amps
    def update_amps(t1, t2, eris):
        t1, t2 = old_update_amps(t1, t2, eris)
        return (np.zeros_like(t1[0]), np.zeros_like(t1[1])), t2
    ccd.update_amps = update_amps
    ccd.conv_tol = 1e-10
    ccd.diis = False
    ccd.kernel()

    return ccd

# import sys
# import requests
# import pickle
# from pyscf import gto, ao2mo, scf, fci, mp, cc
# from occsfrd import wick, ansatz, solve, interface

@pytest.fixture
def closed_shell_CCDequations():
    # assert (len(sys.argv) == 2) and isinstance(sys.argv[1], str)

    # ACCESS_TOKEN = sys.argv[1]
    ACCESS_TOKEN = "tN3zyXtzUd3Si4iPCOQDWSpQqkHHbsORdeqrK5sDxZ0q7xAbeIurIDm4UwnY"
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
    deposition_url = 'https://sandbox.zenodo.org/api/deposit/depositions'
    which_deposition = 0
    r = requests.get(deposition_url, headers=headers)
    r = requests.get("%s/%s" % (deposition_url, r.json()[which_deposition]['id']), headers=headers)
    fNamesAndDownloadLinks = [(file['filename'], file['links']['download']) for file in r.json()['files']]

    eqnFileName = "N0S0_qNOECCD.eqn"

    for name, link in fNamesAndDownloadLinks:
        if name == eqnFileName:
            with open(name[:-3]+"pkl", 'wb') as f:
                r = requests.get(link, headers=headers)
                f.write(r.content)
        
    equationsDict = occsfrd.interface.storeequations.load(eqnFileName[:-4])
    return equationsDict

@pytest.fixture
def closed_shell_qNOECCSDequations():
    eqnFileName = "N0S0_qCCSD.eqn"

    equationsDict = occsfrd.interface.storeequations.load(eqnFileName[:-4])
    return equationsDict