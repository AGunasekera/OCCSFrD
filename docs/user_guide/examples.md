# Examples

## H2O closed-shell CCSD (from the test suite)

```python
import numpy as np
import occsfrd
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
mol.basis = "sto-3g"
mol.spin = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()

eqn = occsfrd.interface.storeequations.load("N0S0_qCCSD")

result = occsfrd.solve.cc.runUnlinkedCC(
    mf, eqn, levelShift=0, verbosity=1,
    Etol=10, Rtol=6, maxIter=1000, nDIIS=12, maxOrder=2,
)

print("CC energy:", mf.e_tot + result["total electronic energy"])
```

## Deriving closed-shell CCD equations

See `tests/test_occsfrd/test_equations.ipynb` for a worked example of deriving
CCD energy and amplitude equations from a BCH ansatz.
