import pytest
from pyscf import gto, scf, cc
import occsfrd

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
def h2o_631g_pyscf_rhf(h2o_631g_pyscf_mol):
    mf = scf.RHF(h2o_631g_pyscf_mol)
    mf.kernel()

    return mf

@pytest.fixture
def h2o_631g_pyscf_ccsd(h2o_631g_pyscf_mf):
    ccsd = cc.ccsd(h2o_631g_pyscf_mf)
    ccsd.diis = False
    ccsd.kernel()

    return ccsd