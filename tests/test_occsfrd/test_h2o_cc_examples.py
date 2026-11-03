import pytest
import numpy as np
import occsfrd

def test_ccsd_energy_match(h2o_sto3g_pyscf_rhf, h2o_sto3g_pyscf_ccsd, closed_shell_qNOECCSDequations):
    pyscf_ccsd_energy = h2o_sto3g_pyscf_ccsd.e_tot

    occsfrd_ccsd = occsfrd.solve.cc.runUnlinkedCC(h2o_sto3g_pyscf_rhf, closed_shell_qNOECCSDequations, levelShift=0, verbosity=1, Etol=10, Rtol=6, maxIter=1000, nDIIS=12, maxOrder=2)
    occsfrd_qnoeccsdenergy = h2o_sto3g_pyscf_rhf.e_tot + occsfrd_ccsd["total electronic energy"]
    expected_qnoeccsdenergy = -75.01246107595368
    # print(occsfrd_qnoeccsdenergy, pyscf_ccsd_energy)
    # print("diff:", occsfrd_qnoeccsdenergy - pyscf_ccsd_energy, np.isclose(occsfrd_qnoeccsdenergy, pyscf_ccsd_energy, rtol=0, atol=1e-10))
    assert not np.isclose(occsfrd_qnoeccsdenergy, pyscf_ccsd_energy, rtol=0, atol=1e-10)
    assert np.isclose(occsfrd_qnoeccsdenergy, expected_qnoeccsdenergy, rtol=0, atol=1e-10)