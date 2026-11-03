# Core concepts

## Indices

`occsfrd.wick.index.Index` represents an orbital index. It carries:

- `name`: a label.
- `occupiedInVacuum`: whether the index refers to orbitals occupied in the Fermi
  vacuum (hole-type) or empty (particle-type).
- `active`: whether the index refers to singly-occupied (active) orbitals.

`SpecificOrbitalIndex` is a subclass for a particular orbital value (used for
open-shell active orbitals).

## Basic operators

`occsfrd.wick.operator.BasicOperator` is a single fermionic creation or
annihilation operator with an index and spin. `OperatorProduct` and
`OperatorSum` represent products and sums of these. Operators can be normal
ordered with `operator.normalOrder`.

Spin-free excitations are built with `operator.spinFreeExcitation`, which sums
over all spin combinations.

## Tensors

`occsfrd.wick.tensor.Tensor` represents an amplitude tensor of a spin-free
excitation operator. Key attributes:

- `name`: label (e.g. `t`, `f`, `v`).
- `lowerIndexTypes` / `upperIndexTypes`: lists of orbital-type characters
  (`g` general, `p` particle, `h` hole, `c` core, `a` active, `v` virtual).
- `spinFree`: whether the tensor is spin-free.
- `array`: NumPy array of coefficients.
- `excitationRank`: number of created/annihilated electrons.

Tensors support `*` (product), `+`, scalar multiplication, and `.conjugate()`.

## Diagrams and contractions

`Tensor.getAllDiagramsGeneral()` / `getAllDiagramsActive()` enumerate the
possible index-type assignments (diagrams) for a tensor. Each diagram becomes a
term in `Tensor.diagrams`.

`contractions.evaluateWick(term)` performs the full Wick contraction of a
tensor (or operator) expression, returning a `TensorSum`/`OperatorSum` of
fully contracted terms.

## Ansatz

An ansatz is a similarity-transformed Hamiltonian `e^{-T} H e^{T}` expanded
(Baker-Campbell-Hausdorff) and expressed as a tensor product. Projecting onto
excitation manifolds yields the amplitude equations.

## Reference (open shell)

For open-shell systems the reference is a configuration state function (CSF)
built from active orbitals. `occsfrd.reference.UGA` provides
`singleCSF_ref` and `multipleCSF_ref` to generate the reference operator.
