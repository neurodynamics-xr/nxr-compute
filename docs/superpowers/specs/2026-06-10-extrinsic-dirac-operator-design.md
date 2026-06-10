# Extrinsic (Relative) Dirac Operator — Design

**Date:** 2026-06-10
**Status:** Approved (design); pending spec review → implementation plan
**Reference:** Liu, Jacobson, Crane — *A Dirac Operator for Extrinsic Shape
Analysis*, SGP 2017 (Computer Graphics Forum 36(5)). Discretization follows
Crane–Pinkall–Schröder, *Spin Transformations of Discrete Surfaces*
(SIGGRAPH 2011, "CPS11").

---

## Goal

Add a **purely geometric** curvature-aware differential operator to
nxr-compute: the Liu et al. one-parameter family

```
L(τ) = (1 − τ)·Δ  +  τ·D_N ,      τ ∈ [0, 1]
```

interpolating between the intrinsic Laplace–Beltrami operator (`τ = 0`) and
the **relative Dirac operator** `D_N` (`τ = 1`), which depends only on the
Gauss map (the surface's extrinsic bending). The operator and its eigenbasis
are computed from cortex geometry alone. **Any downstream use** — e.g.
expanding a leadfield onto this basis — is explicitly out of scope; this is a
geometry-only operator that produces a curvature-aware eigenbasis, exactly
analogous to how the cotan-Laplacian eigenbasis is produced today.

### Why this operator

The Laplace–Beltrami eigenbasis is purely intrinsic — it is blind to how the
cortex bends in space (a flat sheet and a rolled sheet are isospectral). The
relative Dirac family adds extrinsic sensitivity in a *principled* way: the
extrinsic block is the energy of the shape operator `dN` (the second
fundamental form), so the spectrum responds to normal curvature. This is the
canonical, geometry-fixed resolution of the "what is the coupling parameter"
question left open by the cortical covariant differential operators work
(`2026-06-10-cortical-covariant-differential-operators-design.md`): there, the
flat full-frame connection gave a spectrally *degenerate* Ambient Laplacian
because the frame connection has zero curvature; here, differentiating the
**Gauss map** instead of the **frame** injects genuine curvature content, and
`τ` is the principled interpolation Liu et al. define — not a free knob.

---

## Mathematical specification

### The operator (continuous)

Following the paper (§4):

- Extrinsic Dirac operator `Dψ = −(df ∧ dψ)/|df|²`, acting on quaternion-valued
  functions `ψ : M → ℍ`.
- Its square splits into intrinsic + extrinsic:
  `D²ψ = Δψ + (dN ∧ dψ)/|df|²`, where `Δ` is the (real) Laplace–Beltrami
  operator acting **componentwise** on the four quaternion components, and
  `dN` is the derivative of the Gauss map (the shape operator).
- The **relative Dirac operator** is the extrinsic block on its own:
  `D_N ψ = −(dN ∧ dψ)/|df|²`.
- The family is the convex combination `L(τ) = (1−τ)Δ + τ D_N`.

`L(τ)` is self-adjoint and elliptic for `τ < 1`. At `τ = 1`, `D_N` is only
positive-*semi*definite (it has a large kernel on flat regions, where the
Gauss map fails to be an immersion); any `τ < 1` regularizes it.

### Discretization (real 4×4-block representation)

We represent each quaternion `q = a + b·i + c·j + d·k` as a vector in ℝ⁴ with
component order **`[w, x, y, z] = [a, b, c, d]`**. Left-multiplication by `q`
is the 4×4 real matrix

```
L_q = [ a  −b  −c  −d ]
      [ b   a  −d   c ]
      [ c   d   a  −b ]
      [ d  −c   b   a ]
```

For a purely imaginary `v = (x, y, z)` (i.e. `a = 0`), `L_v` is antisymmetric.

**Intrinsic block** `Δ₄ = cotanL ⊗ I₄`. The existing geometry-central cotan
Laplacian, block-replicated across the four quaternion components (each scalar
entry `cotanL(u,v)` becomes `cotanL(u,v)·I₄`). This is the operator's value at
`τ = 0` and is the correctness anchor (see Tests).

**Extrinsic block** `D_N`. Per oriented face `f = ijk` with area `A_f` and
vertex normals `N_i, N_j, N_k`, the relative Dirac operator (Liu §4.2) is

```
(D_N ψ)_f = −(1 / 2A_f) · Σ_{(p,q,r) ∈ C(ijk)} (N_r − N_q) · ψ_p
```

where `C(ijk) = {(i,j,k), (j,k,i), (k,i,j)}` are the cyclic shifts. This
assembles a **rectangular** real matrix `D ∈ ℝ^{4F × 4V}`: block-row `f`,
block-columns `i,j,k` hold the 4×4 antisymmetric blocks
`−L_{(N_r − N_q)} / (2A_f)` (each `N_r − N_q` is a vector → imaginary
quaternion → `L_v`).

**Self-adjoint extrinsic operator (Galerkin form).** The relative-Dirac energy
`⟨⟨D_N ψ, ψ⟩⟩ = Σ_f A_f |(D_N ψ)_f|²` gives the symmetric PSD vertex operator

```
E = Dᵀ · ⋆_F · D ,      ⋆_F = diag(A_f) ⊗ I₄    (face-area 2-form mass)
```

`E ∈ ℝ^{4V × 4V}`, symmetric, positive-semidefinite. This Galerkin/energy
route (rather than CPS11 §5.5's rectangular-`D` eigen-procedure) is chosen
because it yields a square self-adjoint matrix that plugs into the existing
`solveEigenmodes` with zero new solver machinery; it realizes the same
quadratic form.

**The family (assembled).**

```
L(τ) = (1 − τ)·(cotanL ⊗ I₄)  +  τ·E          (4V × 4V, real symmetric)
```

a literal convex blend of two assembled matrices.

### Eigenproblem

The curvature-aware eigenbasis is the solution of the generalized symmetric
eigenproblem

```
L(τ) φ = λ B φ ,      B = M_Galerkin ⊗ I₄
```

`B` uses geometry-central's **Galerkin** vertex mass (`vertexGalerkinMassMatrix`,
already exposed as `operators().mass().galerkin()`), block-replicated ×4 — the
consistent (FEM) mass, no new assembly. Each quaternionic eigenfunction appears
as a **4-fold degenerate** real eigenpair (the four components differ by a unit
quaternion); this clustered degeneracy is exactly what the existing
`normalizeEigenmodes` M-orthonormalization handles.

---

## Architecture & surfaces

### C++ — operators facet

A new `dirac` sub-family on `OperatorsFacet`, sibling to
`laplacian()`/`mass()`/`hodge()`/`dec()`/`gradient3D()`:

```cpp
// Returns L(τ) = (1−τ)(cotanL ⊗ I₄) + τ·E, a [4V × 4V] real symmetric sparse
// matrix. τ ∈ [0,1] (τ=0 ⇒ block cotan-Laplacian; τ=1 ⇒ pure relative Dirac).
// Returned BY VALUE (the blend is τ-dependent); the expensive extrinsic block
// E is cached, the intrinsic block is the existing cached cotan ⊗ I₄.
Eigen::SparseMatrix<double> dirac(double tau) const;
```

- We expose the second-order family `L(τ)`, **not** the bare first-order Dirac
  `D` (rectangular, not directly eigendecomposable). The endpoints give the
  building blocks for free: `dirac(0.0)` is `cotanL ⊗ I₄`, `dirac(1.0)` is `E`.
- `τ` validated to `[0,1]`; out of range → `Error(InvalidInput)`.

### Caching

`OperatorId::Dirac` caches **only the extrinsic Galerkin block `E`** (the
expensive part — the per-face quaternionic assembly). The intrinsic block is
derived on demand from the already-cached `LaplacianCotan` via the cheap `⊗ I₄`
kron, and the `(1−τ)/τ` blend is a sparse axpy per call. This keeps the
one-matrix-per-`OperatorId` cache model intact and avoids per-`τ` cache bloat
over a continuous knob. `isOperatorCached(Dirac)` / `releaseOperator(Dirac)`
wired as for the other operators. `E` depends only on geometry (vertex normals
+ face areas), so it is **not** gauge-dependent and is not invalidated by
`setGauge`.

### Library backing

A new free function in a `nxr::manifold` namespace (e.g.
`differential::relativeDiracExtrinsicBlock(Manifold&) → SparseMatrix<double>`
returning `E`, or a small `dirac.cpp`/`dirac.h` pair), assembled from:
`geometry::vertexNormals` (the Gauss map `N`) and face areas. No
geometry-central operator exists for this — it is assembled directly from the
4×4-block formulas above (this is the one operator we build ourselves rather
than wrapping GC; Implementation Rule 1 wraps GC *where GC provides it*).

### MEX

`nxr_compute('operators', h, 'dirac', tau)` returns the native real sparse
`L(τ)` (`[4V × 4V]`). The one marshalling nuance: for the `dirac` family the
third argument is a numeric `τ` (double scalar) where other families take a
string subtype — `cmdOperators` dispatches on the family string and reads
`prhs[3]` as a double for `dirac`. The user can then `eigs(L, B, k,
'smallestabs')` directly (with `B = kron(speye(4), mass_galerkin)`), or route
through the existing eigen command.

### Storage convention

The `4V` dimension is **vertex-interleaved**: vertex `v`, quaternion component
`c ∈ {0,1,2,3}` maps to index `4v + c`, component order `[w, x, y, z]`. The
`cotanL ⊗ I₄` block at scalar entry `(u,v)` occupies rows `4u..4u+3`, cols
`4v..4v+3`. Documented alongside the §11 storage convention in CLAUDE.md. (No
JS binding flatten rule change — this is native sparse only, like the other
`operators` outputs.)

---

## Tests

Native test `test/test_dirac_operator.cpp` (icosphere fixture, mirroring the
existing operator tests):

1. **Intrinsic anchor (headline).** `dirac(0.0)` equals `cotanL ⊗ I₄`
   byte-for-byte (the paper's guarantee that the family reduces to the cotan
   Laplacian at the intrinsic extreme). Build `cotanL ⊗ I₄` independently and
   compare `.norm()` of the difference `< 1e-12`.
2. **Shape & symmetry.** `dirac(τ)` is `[4V × 4V]`; `L(τ) == L(τ)ᵀ` for several
   `τ ∈ {0, 0.25, 0.5, 0.75, 1}`.
3. **PSD behavior.** `L(τ)` is PSD for `τ < 1` (smallest generalized eigenvalue
   `≥ −1e-9` against `B`); `dirac(1.0) == E` is PSD with a nontrivial kernel on
   a flat fixture (e.g. a planar patch — `D_N` kernel dimension `> 0`).
4. **Sphere degeneracy sanity.** On an icosphere, the extrinsic end behaves as
   the paper describes (the Gauss map is conformal to the surface; the spectrum
   is highly structured) — assert the eigenvalues come in the expected 4-fold
   quaternionic multiplets after `normalizeEigenmodes`.
5. **Eigenbasis B-orthonormality.** After `solveEigenmodes(L(0.5), B, k)` +
   `normalizeEigenmodes`, `ΦᵀBΦ ≈ I` (`< 1e-9`).
6. **Cache lifecycle.** `dirac(τ)` populates `OperatorId::Dirac`;
   `isOperatorCached(Dirac)` true; `releaseOperator(Dirac)` clears it; a
   second `dirac(τ')` with different `τ` reuses the cached `E` (no reassembly)
   and returns the correctly re-blended matrix.

MEX test `bindings/mex/test/test_dirac_operator.m`: `operators(h,'dirac',0)`
matches `kron(speye(4), cotanL)`; `eigs(L, B, k, 'smallestabs')` returns real
eigenvalues; degeneracy multiplets present.

---

## Scope boundaries (v1)

**In:** the `operators().dirac(τ)` family (real `4V×4V` self-adjoint matrix),
its caching, the MEX `operators … dirac τ` surface, and the eigenbasis via the
existing `solveEigenmodes`/`normalizeEigenmodes` path. Closed cortex meshes.

**Deferred / out:**
- **Boundary conditions / infinite potential well** (Liu §5). Cortical
  hemispheres are closed (topological spheres); revisit only if patch analysis
  is needed.
- **Cross-surface canonical representative** (unit-quaternion gauge, Liu §4.3).
  Only matters when comparing eigenbases *across* surfaces; on a single fixed
  cortex the quaternion gauge is the analogue of the Laplacian's ± sign and
  does not affect projection energy.
- **The bare first-order Dirac `D`** (rectangular). Not eigendecomposable
  directly; YAGNI (`τ = 1` gives pure `D_N`).
- **Any leadfield / data-expansion concern.** Downstream of this operator;
  explicitly not part of this work.
- **MATLAB `+bct` reference.** Not authored for this operator; the `τ = 0`
  intrinsic anchor and the paper's discretization are the validation oracles.

---

## Open questions

None outstanding. The two design forks (Galerkin square form over the
rectangular-`D` procedure; `τ` as a continuous runtime knob) are resolved in
favor of Galerkin assembly and a free `τ`. Galerkin vertex mass for `B` is
sourced from the existing GC-backed `mass().galerkin()`.
