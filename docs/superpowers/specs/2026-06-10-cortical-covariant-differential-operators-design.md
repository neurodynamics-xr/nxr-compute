# Cortical covariant differential operators — frame transport + covariant gradient/divergence

**Date:** 2026-06-10
**Status:** Design — approved in discussion, pending spec review
**Scope:** C++ library + MEX. Additive. Builds on the embedded grid (`geometry::vertexGrid`)
and the operators/facet structure already on `main`.

---

## 1. Goal

Give the cortical-frame leadfield a **differential calculus that is free of frame-rotation
artifacts.** A leadfield is a 3-vector field; expressed in per-vertex cortical frames
(tangent₁, tangent₂, normal) it becomes physiologically interpretable, but the frames
rotate and tilt with the folding — so the *naïve* component-wise difference between two
vertices reports spurious change (the canonical case: two leadfields **parallel in Cartesian
space** on opposite sulcal walls read as **antiparallel** in local frames, purely because the
normal flips across the fold). This design supplies the **connection** that subtracts that
frame variation, so transport and gradients reflect true leadfield variation, not the moving
frame.

Two concrete capabilities:

1. **Transport** a 3-vector between any two vertex frames (e.g. across a sulcus) correctly.
2. **Differentiate** a cortical-frame leadfield field along the surface (covariant gradient /
   divergence) without spurious jumps.

## 2. The object: the flat full-frame connection `P_ij = Fⱼᵀ Fᵢ`

Per vertex, build the orthonormal 3-frame from the grid:

```
Fᵥ = [ e1ᵥ | e2ᵥ | nᵥ ]   columns,   e1ᵥ = Re(cᵥ),  e2ᵥ = Im(cᵥ),  nᵥ = e1ᵥ × e2ᵥ
```

where `cᵥ = e1ᵥ + i·e2ᵥ` is the existing complex grid (`geometry::vertexGrid`). The transport of
a frame-local 3-vector from `i` to `j` is the orthogonal `3×3`

```
P_ij = Fⱼᵀ Fᵢ
```

Its blocks are the tangential rotation (top-left `2×2`), the normal↔tangent tilt (the shape
operator, off-diagonals), and the normal alignment (bottom-right). It handles the rotation
and the tilt in one object.

**Key property — flat / path-independent.** Holonomy is identically zero:
`FᵢᵀFₖ · FₖᵀFⱼ · FⱼᵀFᵢ = I` exactly, for any `i,j,k`. So transport between **any** two
vertices (adjacent or on opposite sulcal walls) is the **direct endpoint product** `Fⱼᵀ Fᵢ`
— no path to choose, no ambiguity. This is exactly why it is the correct tool for artifact-free
comparison and for a consistent gradient/divergence: path-independence guarantees that
integrating or comparing over extended cortex gives the same answer regardless of route.

**What this is and is not.** This is the connection for **transport and first-order
differential analysis**, where flatness is the *virtue* (it removes the artifact and is
unambiguous). It is the same connection whose *Laplacian* `GᵀWG` is the (spectrally
degenerate) Ambient covariant — that degeneracy is irrelevant here because we use `G` as a
**gradient**, not an eigenbasis. The curvature-coupled *spectral* operator (the `α`-family /
shell / Dirac) is a separate, deliberately-out-of-scope question (§11); it does not arise for
transport, which is canonical and parameter-free.

## 3. Components

### 3a. Frame transport `P_ij`
- **`frameTransport(i, j) → 3×3`** for an arbitrary vertex pair (the sulcal-wall comparison).
  Computed directly as `Fⱼᵀ Fᵢ` from the two grid frames — flatness makes this exact for any
  pair, no path needed.
- The same per-edge transports, packed, back the gradient operator (§3c).

### 3b. Frame lift (local ↔ world)
- **`liftToWorld(Lloc) → Lworld`**: per vertex `Lworld[v] = Fᵥ · Lloc[v]` — the inverse of the
  grid transform `G·cᵀ`. This is the operationally minimal artifact-free comparison: lift both
  leadfields to Cartesian, compare there. Provided as a convenience because it is the cheapest
  correct recipe and `P_ij` factors through it (`P_ij = Fⱼᵀ·(Fᵢ·)`).
- **`liftToFrame(Lworld) → Lloc`**: the inverse, `Lloc[v] = Fᵥᵀ · Lworld[v]`.

### 3c. Covariant gradient operator `G` (the differential tool)
Per directed edge `e: i→j`, the **covariant difference** of a frame-local field `L`:

```
δ_e = L_j  −  P_ij · L_i              (measured in j's frame)
```

`G` is the sparse linear map `3N → 3E` with, for edge `e=(i,j)`, a `+I₃` block at vertex `j`
and a `−P_ij` block at vertex `i`. It is the discrete covariant exterior derivative `d0` lifted
from scalars (±1 incidence) to 3-vectors (the `3×3` frame transport). A field that is
**constant in Cartesian space** (uniform world dipole) has `G L = 0` exactly — the artifact is
gone — whereas the naïve local difference does not.

### 3d. Derived operators (consistency, not new math)
- **Covariant divergence** `= Gᵀ W` (W = per-edge cotan/weight diagonal).
- **`GᵀWG` equals the existing Ambient covariant Laplacian** (`operators().laplacian().covariant(Ambient)`)
  up to assembly convention — this is a *consistency check*, not a new operator, and ties the
  new gradient to the operator already on `main`.
- Per-edge directional derivative `= δ_e / edgeLength_e`, and per-edge change magnitude `‖δ_e‖`,
  are trivial post-products documented for the consumer (not separate operators).

## 4. C++ API (additive)

New free functions in `nxr::manifold::ops` (or a focused `…::differential` sub-namespace),
sourcing frames from the grid:

```cpp
// 3x3 transport between any two vertex frames (Fj^T Fi). Flat ⇒ exact for any pair.
Eigen::Matrix3d frameTransport(Manifold& m, int i, int j);

// Per-vertex frames stacked (column frames Fv), for lift + custom use.  [3 x 3 x nV] packed
//   as an [nV x 9] row-major array, or returned via a small accessor.
Eigen::MatrixXd vertexFrameMatrices(Manifold& m);   // [nV, 9] row-major Fv

// Covariant gradient operator G : 3N -> 3E (component-major blocks; see §6 layout).
// W defaults to identity (raw differences); pass cotan weights to weight it.
Eigen::SparseMatrix<double> covariantGradient(Manifold& m);

// Lift helpers (operate on an [nV,3] field).
Eigen::MatrixXd liftToWorld(Manifold& m, const Eigen::MatrixXd& Lloc);   // [nV,3] -> [nV,3]
Eigen::MatrixXd liftToFrame(Manifold& m, const Eigen::MatrixXd& Lworld); // [nV,3] -> [nV,3]
```

Optionally surfaced on the operators facet for discoverability:
`operators().gradient3D()` → the sparse `G` (cached like the other operators). Frame transport
and lifts are field/utility operations, not cached matrices, so they stay free functions /
gauge-facet helpers.

**No new geometry math:** every input is the grid (already computed) + the cotan weights
(already available). `frameTransport`/`G` are frame algebra over existing data.

## 5. MEX surface (thin, additive)

- **`nxr_compute('frameTransport', h, i, j)`** → `3×3` double (1-based `i,j`).
- **`nxr_compute('operators', h, 'gradient3D')`** → native sparse `G` (extends the existing
  `operators` string-dispatch with a `gradient3D` key; real sparse).
- **`nxr_compute('liftToWorld', h, Lloc)`** / **`('liftToFrame', h, Lworld)`** → `[nV×3]`.
  (Lifts take a field argument, so they are their own commands rather than the matrix-returning
  `operators` family.)

Marshalling reuses `eigenSparseToMx` (G) and the existing real-matrix helpers (lifts, 3×3).
The geometry/gauge consumer bundles are unchanged; only these thin commands are added.

## 6. Layout / conventions

- **Field representation:** `[nV × 3]` real, each row a frame-local leadfield vector
  `[a, b, c] = (tangent₁, tangent₂, normal)` component. World fields are `[nV × 3]` Cartesian.
- **Operator `G` internal layout:** component-major `3N` input `[a; b; c]` and `3E` output,
  matching the existing covariant Laplacian's `[a;b;c]` block layout so `GᵀWG` aligns with
  `operators().laplacian().covariant(Ambient)` for the consistency test. The MEX boundary
  accepts/returns `[nV×3]`/`[nE×3]` and reshapes; `G` itself is exposed in the component-major
  block layout (documented) so consumers can apply it to a reshaped field.
- **Edge orientation:** `G`'s rows follow the mesh's canonical halfedge/edge orientation
  (`he == he.edge().halfedge()`); the sign convention is documented so divergence `Gᵀ` composes
  correctly.

## 7. Error handling

- `frameTransport(i,j)` with out-of-range `i`/`j` → `Error(ErrorCode::InvalidInput)`.
- Lift/gradient field with wrong shape (`cols != 3`, `rows != nV`) → `Error(InvalidInput)` with
  the expected dimensions in the hint.
- Degenerate frame (a vertex where `e1×e2` is near-zero, i.e. a degenerate tangent basis) →
  the grid already guards this; `frameTransport` inherits the grid's validity. No new failure
  mode introduced.

## 8. Test plan

Native (`test/test_covariant_differential.cpp`) on the icosphere + the rhombus, plus a curved
two-frame fixture:

- **Frame orthogonality:** `Fᵥᵀ Fᵥ = I₃` per vertex; `P_ij` orthogonal (`P_ijᵀ P_ij = I`).
- **Flatness / path-independence:** for every interior triangle, `P_ki·P_jk·P_ij = I` to `< 1e-12`.
- **Artifact removal (the headline test):** set a **Cartesian-constant** leadfield
  `Lworld[v] = (1,0,0)` ∀v; lift to frames (`Lloc = Fᵥᵀ·Lworld`); assert `G·Lloc = 0`
  (`< 1e-12`) **while** the naïve local difference `Lloc[j] − Lloc[i]` is nonzero on curved
  edges. This proves the covariant gradient sees through the frame rotation.
- **Sulcal-wall transport:** two frames with opposed normals, a shared Cartesian vector →
  `frameTransport` maps one's local coords onto the other's exactly (`< 1e-12`).
- **Consistency with the existing operator:** `GᵀWG` (W = cotan weights) equals
  `operators().laplacian().covariant(CovariantCoupling::Ambient)` to `< 1e-9` (sign/convention
  reconciled) — ties the gradient to the operator already on `main`.
- **Lift round-trip:** `liftToFrame(liftToWorld(L)) == L` to `< 1e-12`.

MEX (`bindings/mex/test/test_covariant_differential.m`):
- `frameTransport(h,i,j)` is `3×3` orthogonal; equals `Fj'·Fi` from the grid the `geometry`
  command returns.
- `G = operators(h,'gradient3D')`: applying `G` to a Cartesian-constant leadfield (lifted to
  frames) gives ≈0; applying to a non-constant field gives the expected per-edge differences.
- `liftToWorld`/`liftToFrame` round-trip; `liftToWorld` of two sulcal-wall leadfields that are
  Cartesian-parallel yields equal world vectors (artifact removed end-to-end).

## 9. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | which connection | flat full-frame `Fⱼᵀ Fᵢ` (handles tangent rotation **and** normal tilt; path-independent) |
| 2 | path-dependence | **not** wanted — path-independence is what removes the artifact and keeps extended transport unambiguous; the intrinsic (path-dependent) connection handles only the tangent and re-injects curvature |
| 3 | gradient vs spectrum | use `G` as a **gradient** (first-order); the degenerate Ambient spectrum is irrelevant and the `α`-coupled spectral operator is out of scope |
| 4 | canonical / parameter-free | yes — transport has no free parameter (unlike the spectral-basis question) |
| 5 | minimal recipe | `liftToWorld` (= inverse grid transform) is the cheapest correct comparison; `frameTransport`/`G` formalize it for operators and along-surface differentiation |
| 6 | layout | `[nV×3]` fields at the boundary; component-major `3N` internally to match the covariant Laplacian (enables the `GᵀWG` consistency test) |

## 10. Out of scope / deferred

- The curvature-coupled **spectral** operator (`SurfaceAdapted(α)`), and the **shell/Koiter** and
  **extrinsic-Dirac** alternatives — separate design(s); they answer the spectral-basis question,
  not the transport/gradient one.
- Per-vertex gradient assembly (least-squares Jacobian from edge differences), higher-order
  covariant operators, and time/phase (chiral) analysis — downstream, built on `G` if needed.
- Re-cutting the geometry consumer bundles — unchanged.
