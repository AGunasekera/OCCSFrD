import pytest
import numpy as np
import occsfrd

def test_ccsd_energy_match(h2o_sto3g_pyscf_rhf, h2o_sto3g_pyscf_ccsd, closed_shell_qNOECCSDequations):
    pyscf_ccsd_energy = h2o_sto3g_pyscf_ccsd.e_tot

    occsfrd_ccsd = occsfrd.solve.cc.runUnlinkedCC(h2o_sto3g_pyscf_rhf, closed_shell_qNOECCSDequations, levelShift=0, verbosity=1, Etol=10, Rtol=6, maxIter=1000, nDIIS=12, maxOrder=2)
    occsfrd_qnoeccsdenergy = h2o_sto3g_pyscf_rhf.e_tot + occsfrd_ccsd["total electronic energy"]
    
    print(occsfrd_qnoeccsdenergy, pyscf_ccsd_energy)
    assert np.isclose(occsfrd_qnoeccsdenergy, pyscf_ccsd_energy, atol=1e-10)