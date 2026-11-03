# Overview

OCCSFrD is built around three ideas:

1. **Wick contraction engine** (`occsfrd.wick`): a general engine for evaluating
   vacuum expectation values of products of second-quantised operators. It handles
   spin-free and spin-adapted operators, normal ordering, and contraction of
   free indices.

2. **Ansatz generators** (`occsfrd.ansatz`): routines that build coupled-cluster
   ansaetze (closed-shell and open-shell / normal-ordered) as tensor products
   and derive the corresponding energy and amplitude equations via Wick
   contractions.

3. **Equation solver** (`occsfrd.solve`): takes a derived equation set (energy +
   amplitude equations) and a PySCF mean field, builds integrals, and iterates the
   amplitudes to convergence using DIIS.

A typical run has two phases:

- **Derivation** (offline, possibly once per ansatz): generate equations and
  store them with `occsfrd.interface.storeequations.save`.
- **Solving** (per molecule): load equations with `storeequations.load` and call
  `solve.cc.runUnlinkedCC`.

This separation keeps the expensive symbolic derivation out of the production
calculation loop.
