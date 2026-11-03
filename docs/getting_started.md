# Getting started

This page walks through the basic workflow of OCCSFrD: from a PySCF mean-field
calculation to a converged coupled-cluster energy.

## 1. Build a molecule and mean field

```python
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
mol.basis = "sto-3g"
mol.spin = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()
```

## 2. Load a pre-derived equation set

OCCSFrD separates *equation derivation* from *equation solving*. Derived equations
are stored as pickled dictionaries (`.pkl`) and loaded with
`occsfrd.interface.storeequations.load`:

```python
import occsfrd

eqn = occsfrd.interface.storeequations.load("N0S0_qCCSD")  # loads N0S0_qCCSD.pkl
```

Equation sets for common methods (e.g. closed-shell CCSD, CCD) are distributed
via Zenodo and downloaded with
`occsfrd.interface.storeequations.downloadFromZenodo`.

## 3. Run the coupled-cluster calculation

```python
result = occsfrd.solve.cc.runUnlinkedCC(
    mf,
    eqn,
    levelShift=0,
    verbosity=1,
    Etol=10,
    Rtol=6,
    maxIter=1000,
    nDIIS=12,
    maxOrder=2,
)

cc_energy = mf.e_tot + result["total electronic energy"]
print("CC energy:", cc_energy)
```

## 4. (Optional) Derive your own equations

For research on new ansaetze, OCCSFrD can derive the Wick-contracted equations
directly. See the user guide sections on
[closed-shell CC](user_guide/closed_shell.md) and
[open-shell CC](user_guide/open_shell.md).
