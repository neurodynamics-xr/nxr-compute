# Operator eigensolve — `eigenProblemFor` + `eigenOperator` + WASM `eigs()`

**Date:** 2026-06-15
**Status:** design → implementing
**Motivation:** WASM (and every binding) can eigensolve the Dirac families today
only by hand-assembling the generalized mass `B = M ⊗ I₄`, marshaling the operator
out as COO and the triplets back in, and living with Spectra's multiplicity
miscount on the 4-fold quaternionic clusters. The operator↔mass pairing and the
quaternionic structure are library knowledge that leaks into every consumer.

## Decisions (locked)

1. **Shared C++ helper, all bindings.** The "operator → generalized eigenproblem
   `(K, M, σ)`" mapping is the single source of truth in the core library
   (`solve::eigenProblemFor`). MEX / WASM / native all consume it. No per-binding
   mass duplication.
2. **ℍ-multiplet reconstruction + dense verify flag.** For the Dirac (blockSize 4)
   an opt-in pass reconstructs exact 4-fold multiplets from the right-quaternion
   symmetry; an opt-in dense path (`GeneralizedSelfAdjointEigenSolver`) gives an
   exact cross-check on small meshes.

## Core API (`nxr::manifold::solve`)

```cpp
enum class EigenOperator {
    LaplacianCotan, LaplacianGraph, Dirac, DiracFace
    // Covariant (3N) deferred — component-major (mass = kron(I₃, M), unlike the
    //   Dirac's interleaved kron(M, I₄)); ambient spectrum is the triply-
    //   degenerate Laplace–Beltrami spectrum anyway.
    // connection (complex Hermitian) deferred — needs the real2N embedding.
};

struct EigenOperatorSpec {
    EigenOperator op;
    double tau = 0.0;                                   // Dirac / DiracFace blend
    ops::MassMatrixVariant mass = ops::MassMatrixVariant::Galerkin;  // vertex domain
};

struct EigenProblem {
    Eigen::SparseMatrix<double> K;   // operator
    Eigen::SparseMatrix<double> M;   // natural generalized mass (block-expanded)
    double sigma;                    // shift for shift-invert (default -1e-8)
    int blockSize;                   // 1 scalar / 3 covariant / 4 Dirac
};

// SINGLE SOURCE OF TRUTH for operator → (K, M, σ, blockSize).
EigenProblem eigenProblemFor(Manifold&, const EigenOperatorSpec&);

// Assemble + eigensolve + (optional) normalize + (optional) ℍ-reconstruct.
EigenResult eigenOperator(Manifold&, const EigenOperatorSpec&, int k,
                          double sigma = -1e-8,
                          bool normalize = true,
                          bool reconstructMultiplets = false,
                          bool dense = false,
                          const CancellationToken& = {},
                          const ProgressObserver&  = {});
```

### Operator → mass table (the centralized knowledge)

| operator        | K                       | M                          | block |
|-----------------|-------------------------|----------------------------|-------|
| LaplacianCotan  | `cotanL`                | `mass` (lumped/galerkin)   | 1 |
| LaplacianGraph  | `graphL`                | `I` (standard eigenproblem)| 1 |
| Covariant       | `covariant` (3N, ambient)| `mass ⊗ I₃`               | 3 |
| Dirac           | `dirac(τ)` (4V)         | `M_galerkin ⊗ I₄`          | 4 |
| DiracFace       | `diracFace(τ)` (4F)     | `diag(faceArea) ⊗ I₄`      | 4 |

`blockKron(M, b)` expands a `[n×n]` sparse into `[bn×bn]` (each `(i,j,v)` →
`(b·i+c, b·j+c, v)`, c=0..b−1) — the same `kron(M, I_b)` the tests build by hand.

## ℍ-multiplet reconstruction (blockSize == 4)

`dirac(τ)` / `diracFace(τ)` commute with right-quaternion multiplication, so the
fixed real per-vertex 4×4 maps (component order `[w,x,y,z]`)

```
R_i: [w,x,y,z] → [-x, w, z, -y]
R_j: [w,x,y,z] → [-y,-z, w,  x]
R_k: [w,x,y,z] → [-z, y,-x,  w]
```

send any eigenvector to three more with the *same* eigenvalue. Reconstruction:

1. Eigensolve `k` (rounded up to a multiple of 4) modes.
2. Cluster eigenvalues by relative gap.
3. Per cluster, build the orbit `O = [S | R_iS | R_jS | R_kS]` of the cluster's
   computed vectors `S`.
4. Rank-reveal an M-orthonormal basis of `span(O)` via the dense M-Gram
   `G = OᵀMO` (small): keep eigenpairs with `λ > tol·λmax`, whiten. Rank `r` is
   the true multiplicity (provably divisible by 4).

This recovers *complete* multiplets even when Spectra under-converged a cluster
(found 5 of an 8-fold → closing under ℍ recovers all 8), provided ≥1 representative
per distinct eigenvalue converged.

## Dense verify flag

`dense=true` → Eigen `GeneralizedSelfAdjointEigenSolver(K, M)`, take the lowest
`k`. Exact eigenvalues + multiplicities. O(n³)/O(n²) — viable only for small
meshes / fixtures (a 10k-vertex cortex Dirac is 40k² dense ≈ 13 GB). Guarded by a
size cap; documented as verification-only, not the cortical path.

## WASM surface

```js
manifold.eigs({ operator: 'dirac', tau: 0.5, k: 16, multiplets: true })
manifold.eigs({ operator: 'diracFace', tau: 1.0, k: 16 })
manifold.eigs({ operator: 'laplacian', subtype: 'cotan', k: 64, mass: 'galerkin' })
manifold.eigs({ operator: 'covariant', k: 32 })
// returns { eigenvalues, eigenvectors (vMajor, n·k), k, nConverged, blockSize,
//           multiplicities? }
```

No JS-side `⊗I₄`, one boundary crossing (results only). `solveEigenmodesFromTriplets`
stays as the bring-your-own-matrix escape hatch. The N-API addon is untouched
(out of scope). Multiplicity labels are exact only with `multiplets:true` or
`dense:true`; otherwise the basis is correct but Spectra cluster counts are not
authoritative (documented).

## Scope / deferrals

- Connection Laplacian (complex Hermitian) eigensolve via real2N — deferred.
- First-order rectangular `D`/`D̃`/`D_int` have no eigendecomposition; their
  singular spectrum is `√(eig of the squared family)` — use `eigs('dirac', τ=1)`.
- In-heap eigenvector handles (avoid the result copy) — future optimization.
