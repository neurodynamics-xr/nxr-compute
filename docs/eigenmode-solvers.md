# Eigenmode solvers: Hodge–Dirac, spin Diracs, graph Laplacian (2026-10-23)

Request: Diellor 2026-10-23 — eigenmode solvers for the Hodge–Dirac, the extrinsic spin-transform Dirac, and the topological graph Laplacian.
**cortical-flow is the reference where implementations differ** (`hodgeDirac.ts`, `spinDirac.ts`, `solve-lb-hodge.py`); parity on `bunny-closed` and the glued sub-0002 cortex is `test/test_cf_parity.cpp`.

## What existed (the map)

| operator | matrix | registered for eigensolve (`EigenOperator`) | pencil |
|---|---|---|---|
| cotan Laplacian | yes | `LaplacianCotan` | (L, Galerkin or lumped) |
| graph Laplacian `d0ᵀd0` = D − A | yes (`graphLaplacian`) | `LaplacianGraph` | (L, I) — **degree mass missing** |
| Crane Dirac `diracD` (Gauss map), relative Dirac L(τ) | yes (`dirac::matrix`, `relativeDirac`) | `Dirac`, `DiracFace` | (L(τ), M⊗I₄) 4V |
| `diracIntrinsicD` (edge vectors) | yes (`matrixIntrinsic`) | no (rectangular 4F×4V; its square is in `Dirac` τ=0) | — |
| Hodge–Dirac d+δ | **none** | none | — |
| Hoffmann–Ye spin Diracs (the spin transformations) | **none** | none | — |

`diracD` **is** Crane's extrinsic Dirac and matches cortical-flow's export to 7e-16 (bunny); its square `E` (= `Dirac` τ=1, lumped mass) reproduces `diracD_mu.npy` to 1e-11. The *spin-transformation* Dirac of Hoffmann & Ye is a different operator (face spinors, hyperedges) and was added.

## Added

1. **`ops::hodgedirac::assemble(m, star1Floor=1e-6, mollify=1e-6)`** — d+δ on 0/1/2-forms of the intrinsic Delaunay mesh (mollify → `flipToDelaunay`), pencil (K, M = diag(⋆0,⋆1,⋆2)), ⋆0 the mixed-Voronoi mass of the *input* surface (before the flips, as cortical-flow), ⋆1 = ½(cot α + cot β) floored (Rheome PR #94), ⋆2 = 1/area. Vertex indices are the input's; edges/faces the intrinsic mesh's.
   `solve::hodgeDiracModes(h, count, bothSigns)` — the `count` lowest +μ by the **exact/coexact split** (Δ₀ and Δ₂ are SPD: no indefinite solve; μ = √λ; exact (ψ, dψ/μ, 0)/√2, coexact (0, δχ/μ, χ)/√2, −μ mirror = 1-form block negated), each mode tagged `kind` (0 exact · 2 coexact). `solve::classifyHodgeModes` tags pencil eigenvectors (1 = harmonic). Zero modes/harmonic 1-forms are not produced by the split; ask the pencil directly: `eigenOperator(HodgeDirac, k, sigma≠NaN)` is sparse-LU shift-invert on the indefinite pencil.
2. **`solve::eigenSliced(A, B, count, slice)`** — spectrum slicing: windows of `slice` shift-invert solves; each window keeps only its converged prefix (residual < 1e-8·max|λ|) and the next one starts at 0.98 of the covered radius. Thousands of modes need a `slice`-column Krylov basis only.
3. **`ops::spin::assemble(m, Kind)`** (`Extrinsic` D_f, `IntrinsicHY`, `IntrinsicYe`; 4F×4F, closed meshes), `ops::spin::vertexToFace(m, covariant)`, and **`solve::eigenSpin`** — Ye et al.'s quotient on vertex spinors (AᵀDM⁻¹DAψ = μ AᵀDAψ, k largest |1/μ|), with the ℍ-multiplets closed (Lanczos under-counts 4-fold clusters; `reconstructDiracMultiplets` is now public).
4. **Graph Laplacian with degree mass** — `EigenOperatorSpec::graphDegreeMass` (L φ = λ D φ). Identity mass was already the default.
5. Registry: `hodgeDirac`, `spinExtrinsicDirac`, `spinIntrinsicHYDirac`, `spinIntrinsicYeDirac`, `graphLaplacianDegreeMass`. Bindings (Node addon, WASM) `eigs({ operator: 'hodgeDirac' | 'spinExtrinsic' | 'spinIntrinsicHY' | 'spinIntrinsicYe' | 'graph', graphMass: 'degree', star1Floor, mollify, bothSigns, k, sigma })`; Hodge results carry `kind`.

## Parity (cortical-flow exports; `test_cf_parity`, `NXR_CF_CORTEX=1` adds the cortex)

| | bunny-closed (14 290 V) | glued sub-0002 cortex (19 692 V, 10 678 flips) |
|---|---|---|
| diracD / diracIntrinsicD matrix | 7e-16 / pass | 6e-17 (intrinsic) |
| diracD spectrum (lowest 8 nonzero μ) | 1e-11 | — |
| spin extrinsic / HY / Ye matrix, A cov/plain, H | 3e-17 / 2e-17 / 2e-17 / pass / 4e-19 | HY 1e-8 (8 of 1.9 M entries, fold-back faces) |
| spin spectra (16 lowest \|μ\|, HY & extrinsic) | 6e-15 / 7e-14 | — |
| Hodge ⋆0 / ⋆1, ⋆2 (sorted) | 7e-16 / 2e-14 | ⋆0 1e-13 |
| Hodge lowest +μ (60 / 50) | 5e-13 | 6e-13 |

Unit tests: `test_hodge_dirac` (structure, ±μ, D² = Hodge Laplacians, split = dense pencil, slicing, indefinite shift), `test_spin_dirac` (round-sphere spectra: intrinsic ±1 ×4, extrinsic 1 ×8 and −2 ×4), `test_graph_eigen` (icosahedron D − A spectrum 0, 5∓√5, 6; degree mass).

## Limits (say what the instrument cannot resolve)

- Harmonic 1-forms (μ = 0) are not in the split; use the pencil solve near σ = 0⁺ (σ exactly 0 is singular).
- Spin eigenmodes are on vertex spinors; the face-spinor pencil itself has spurious small eigenvalues (the cell doubling) and is not offered as an eigenproblem (`eigenProblemFor` throws).
- Hub runs: small meshes only. The bunny parity takes 26 s; the cortex solves are for fir through nsp.
- The WASM `k ≤ 1000` cap still applies; the Node addon takes any k < N.
