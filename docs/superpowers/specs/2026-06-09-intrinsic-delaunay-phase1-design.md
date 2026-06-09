# Intrinsic-Delaunay context normalization — Phase 1 (certified-PSD cotan + ambient covariant)

**Date:** 2026-06-09
**Status:** Design — approved in discussion, pending spec review
**Context:** `fixDelaunay` (extrinsic) is best-effort PSD. The *certified*-PSD path is
intrinsic Delaunay (geodesic edges, metric preserved), applied as a context-level
normalization. Phase 1 delivers the certified-PSD **cotan Laplacian + ambient
covariant** — the gauge-free, no-`φ_v` core. Phase 2 (connection L + grid via the
per-vertex signpost rotation `φ_v`) and Phase 3 (fully-intrinsic Topology/Geometry/DEC)
follow. See the investigation note in the conversation for the `φ_v` mechanics.

---

## 1. Goal

`nxr_compute('create', V, F, struct('intrinsicDelaunay', true))` builds a
`SignpostIntrinsicTriangulation` over the mesh and `flipToDelaunay()`s it. Operators
that are **gauge-free** (the cotan Laplacian and mass) are then assembled on the
intrinsic Delaunay geometry → **certified PSD** (all cotan weights ≥ 0, Bobenko–
Springborn). The **ambient `covariantLaplacian`** inherits PSD (its spectrum is the
cotan spectrum). Vertices are untouched (`flipToDelaunay` adds no Steiner points;
the intrinsic mesh is 1:1 with the input vertices, same indices), so every V-indexed
output aligns with the original vertex ordering and the leadfield correspondence.

Default (`intrinsicDelaunay` absent/false) is **byte-identical** to today.

## 2. Scope — what Phase 1 routes intrinsic, and what it does not

| Quantity | Under `intrinsicDelaunay` | Why |
|---|---|---|
| **cotan Laplacian** (`Geometry.operators.laplacian`, and the `cotanLaplacian` used everywhere via `assembleManifoldOperators`) | **intrinsic** (certified PSD) | gauge-free scalar; the Phase-1 payload |
| **mass** (lumped/Galerkin), `vertexDualAreas`, `totalArea` | **intrinsic** | pairs with the intrinsic cotan for a consistent `Kφ=λMφ` |
| **ambient `covariantLaplacian`** | **intrinsic** (inherits) | = frame-conjugate of `kron(I₃, cotanL)`; spectrum is `cotanL`'s |
| connection Laplacian (`Gauge.operators.laplacian`), product covariant | **original** (Phase 2) | needs the intrinsic gauge / `φ_v` to stay consistent with the grid |
| `grid`/frames, normals, curvature (extrinsic) | **original** | extrinsic; Phase 2 realizes the intrinsic grid |
| DEC `d0/d1/hodge`, the light `Topology`/`Geometry` bundle | **original** (Phase 3) | edge/face-indexed — intrinsic connectivity differs from the original |

**Documented consequence:** under normalization the cross-DEC identity
`cotanL == d0ᵀ·hodge1·d0` does **not** hold (cotan is intrinsic, DEC is original).
The existing `test_operators.m` identity assertion runs on a **non-normalized**
context (unchanged); normalized contexts get their own PSD test.

## 3. Architecture — `Manifold` becomes intrinsic-triangulation-aware

```cpp
// include/nxr/compute.h — forward-declare the GC types
namespace geometrycentral::surface {
  class IntrinsicGeometryInterface;
  class SignpostIntrinsicTriangulation;
}

class Manifold {
public:
    Manifold(const double* vertices, int nV,
             const int32_t* faces, int nF,
             bool intrinsicDelaunay = false);   // NEW param, default false
    // ...
    bool isIntrinsicDelaunay() const;            // intrinsicTri_ != nullptr
    // Geometry to assemble *intrinsic-interface* operators on (cotan, mass,
    // dual areas, connection, DEC, transport). Returns the intrinsic Delaunay
    // geometry when normalized, else the embedded VertexPositionGeometry (which
    // is itself an IntrinsicGeometryInterface). The embedded geometry() accessor
    // is unchanged and is still used for EXTRINSIC quantities (normals, frames).
    geometrycentral::surface::IntrinsicGeometryInterface& operatorGeometry();
private:
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh>            mesh_;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry>        geometry_;
    std::unique_ptr<geometrycentral::surface::SignpostIntrinsicTriangulation> intrinsicTri_;  // NEW, null unless normalized
};
```
Constructor: after building `mesh_` + `geometry_` as today, `if (intrinsicDelaunay) {
intrinsicTri_ = std::make_unique<SignpostIntrinsicTriangulation>(*mesh_, *geometry_);
intrinsicTri_->flipToDelaunay(); }`. `operatorGeometry()` returns `*intrinsicTri_` if
set, else `*geometry_`.

**`assembleManifoldOperators`** (`src/mesh_operators.cpp`): change the source of the
cotan Laplacian, mass, `vertexDualAreas`, and `totalArea` from `m.geometry()` to
`m.operatorGeometry()`. Keep `vertexNormals` (and any other extrinsic quantity) on
`m.geometry()` (the embedded interface — `operatorGeometry()`'s base type does not
expose normals). The `cotanLaplacian` view then binds to the intrinsic cache when
normalized; the owning `intrinsicTri_` keeps it alive (lifetime contract satisfied).

**Vertex-index alignment (verify):** the intrinsic mesh is a copy of the input with
1:1 vertex correspondence and matching indices, so `intrinsicTri_->cotanLaplacian` is
`V×V` aligned with the original vertex ordering. Confirm in the test (a normalized and
a raw context produce the same-size `V×V` operators on the same vertices).

## 4. MEX

`nxr_compute('create', V, F[, opts])` — `cmdCreate` reads an optional 3rd struct arg;
`opts.intrinsicDelaunay` (logical) is passed to the `Manifold` ctor. The handle then
yields certified-PSD `Geometry.operators.laplacian` and ambient `covariantLaplacian`.
All other commands are unchanged.

## 5. Test plan

Fixture: a **non-Delaunay** mesh whose raw cotan Laplacian has a **negative edge
weight** — the thin rhombus from `fixDelaunay` (`(0,0,0),(2,0,0),(1,0.2,0),(1,-0.2,0)`,
split along the long diagonal: the obtuse triangle gives a strongly negative diagonal
cotan weight, cot(157°) ≈ −2.1). The icosphere cannot exhibit this (it is Delaunay).

NOTE: the **certificate** is *non-negative weights* (`w_ij ≥ 0` ⟹ PSD, Bobenko–
Springborn), not "raw is indefinite" — a single negative weight on a tiny mesh need
NOT make the Laplacian indefinite (boundary weights can dominate), so we assert the
weight signs + that the *normalized* operator is PSD, which is the reliable claim.

- **Native** (`test/test_intrinsic_delaunay.cpp`):
  - Build a **raw** `Manifold` (intrinsicDelaunay=false) and a **normalized** one
    (true) from the rhombus.
  - `assembleManifoldOperators` on each. The **raw** `cotanLaplacian` has a positive
    off-diagonal entry (a negative cotan weight; `maxOffDiag > 1e-9`). The
    **normalized** one has all off-diagonals `≤ 1e-9` (all weights `≥ 0`) **and** min
    eigenvalue `≥ −1e-9` (**PSD — the certificate**).
  - both are `V×V` (= 4×4) on the same vertices; `isIntrinsicDelaunay()` true/false.
  - default path unchanged: on the icosphere, normalized `cotanLaplacian` equals the
    raw one (already Delaunay ⇒ `flipToDelaunay` is a no-op) — sanity that
    normalization doesn't perturb an already-Delaunay mesh, and that the
    `intrinsicDelaunay=false` path is byte-identical to before.
- **MATLAB** (`bindings/mex/test/test_intrinsic_delaunay.m`):
  - `hRaw = create(V,F)`, `hN = create(V,F, struct('intrinsicDelaunay',true))`.
  - `Geometry.operators.laplacian` (with `operators=true`): raw has a positive
    off-diagonal (negative weight); normalized has all off-diagonals `≤ 1e-9` and
    `min(eig(full(L))) ≥ −1e-9` (PSD).
  - ambient `Gauge.operators.covariantLaplacian` under normalization is symmetric and
    `min(eig) ≥ −1e-9` (PSD).
  - vertex count identical; both `V×V`.

## 6. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | trigger | create-time flag `intrinsicDelaunay`; default false ⇒ byte-identical |
| 2 | Phase-1 scope | cotan Laplacian + mass + ambient covariant intrinsic; rest original |
| 3 | flip routine | `flipToDelaunay` (no Steiner points; vertices/indices preserved) |
| 4 | architecture | `Manifold` holds `intrinsicTri_`; `operatorGeometry()` selects the active intrinsic interface; `geometry()` stays embedded for extrinsic quantities |
| 5 | certificate | intrinsic cotan weights ≥ 0 ⇒ PSD (Bobenko–Springborn); validated on a non-Delaunay fixture the icosphere can't provide |
| 6 | known gap under normalization | cross-DEC identity breaks (cotan intrinsic, DEC original) — documented; Phase 3 makes DEC intrinsic |

## 7. Deferred

- **Phase 2:** connection Laplacian + product covariant + intrinsic `grid`, via the
  per-vertex `φ_v = rescaled signpostAngle[intrinsicMesh.vertex(v).halfedge()]`
  (intrinsic grid `= e^{iφ_v}·input grid`; connection L from `intrinsicTri_`).
- **Phase 3:** fully-intrinsic `Topology`/`Geometry`/DEC bundle (one mesh everywhere).
