# Standalone `normalize` — extrinsic Delaunay edge-flip mesh fix

**Date:** 2026-06-09
**Status:** Design — approved in discussion, pending spec review
**Context:** The cortical-coordinate bundle's operators (cotan / connection / covariant
Laplacian) are indefinite on non-Delaunay meshes (raw signed cotan weights; ~11.4%
negative on FreeSurfer cortex). This adds a standalone mesh-cleanup utility; the
PSD-*certified* intrinsic-Delaunay context normalization is a separate larger follow-on.

---

## 1. Goal

`nxr_compute('normalize', V, F) → [V, F']` — a stateless function that returns a
**genuine repaired `(V, F')` manifold mesh** via extrinsic Delaunay edge flips
(geometry-central `fixDelaunay`). Same vertices (positions, count, indices) → new
face connectivity. Per-vertex correspondence (leadfield, `tess2mri`, annotations)
is preserved exactly.

## 2. What it is — and is not

- **Is:** a thin wrap over geometry-central's `fixDelaunay(mesh, geom)` (in
  `remeshing.h`), which flips edges *in place* (no vertex add / remove / move) to
  satisfy the extrinsic Delaunay criterion. `|V|, |E|, |F|` unchanged; `V`
  identical; `F` re-triangulated.
- **Is best-effort, not a PSD certificate.** Extrinsic Delaunay flips a non-planar
  quad's 3D diagonal, which slightly changes the piecewise-linear surface in curved
  regions and need not fully converge to all-Delaunay. It drives the negative cotan
  weights *way* down (clears the large majority) but does **not** guarantee zero
  negatives. The *certificate* is the intrinsic (geodesic-edge) `flipToDelaunay`,
  which is the separate follow-on and is **not** representable as a plain `(V,F')`.
- **Is not mesh repair** (holes / non-manifold / self-intersection). Input must be a
  valid manifold (`ManifoldSurfaceMesh` throws otherwise) — FreeSurfer surfaces are.

## 3. API

**Library** (`nxr::manifold`):
```cpp
struct DelaunayNormalization {
    Eigen::MatrixXi faces;  // [nF, 3] 0-based, re-triangulated; same vertex indices
    int flips;              // number of edge flips performed (0 ⇒ already Delaunay)
};
DelaunayNormalization normalizeDelaunay(
    const double* vertices, int nV, const int32_t* faces, int nF);
```
Builds `ManifoldSurfaceMesh(polygons)` + `VertexPositionGeometry` exactly as the
`Manifold` constructor does (`src/mesh_operators.cpp`), runs
`geometrycentral::surface::fixDelaunay(mesh, geom)`, then reads the new connectivity
via `mesh.getFaceVertexList()` into `faces` (0-based). Vertices are untouched, so the
caller keeps `V`.

**MEX** — stateless command (no handle), added to the dispatch chain:
```matlab
[V2, F2]          = nxr_compute('normalize', V, F);   % V2 == V (passthrough)
[V2, F2, nFlips]  = nxr_compute('normalize', V, F);   % optional flip count
```
- `V`/`F` are 1-based MATLAB inputs (parsed via `mxToVertexBuffer` / `mxToFaceBuffer`,
  which convert faces 1-based → 0-based).
- `V2` = the input vertices unchanged (`mxDuplicateArray`).
- `F2` = `faces + 1` (back to 1-based), `nF×3` double, column-major.
- `nFlips` = flip count.

## 4. Test plan

- **Native** (`test/test_geometry_bundle.cpp` or a new `test_normalize.cpp`):
  fixture = a thin rectangle (4 vertices: `(0,0,0),(3,0,0),(3,1,0),(0,1,0)`) split
  along the **long** diagonal `(0–2)` — the non-Delaunay diagonal. Assert:
  - `flips == 1` (the long diagonal is flipped to the short `(1–3)`).
  - result is a valid 4-vertex / 2-face / 5-edge manifold (counts unchanged).
  - the output faces use the **short** diagonal (vertices 1 and 3 now share an edge;
    no face contains both 0 and 2 as the shared diagonal). Concretely: after
    normalization, every edge's cotan weight `≥ 0` (assemble the cotan Laplacian on
    the normalized faces and check, or check the two new triangles are the
    short-diagonal split).
  - on the icosphere (already Delaunay): `flips == 0` and faces unchanged (as a set).
- **MATLAB** (`bindings/mex/test/test_normalize.m`):
  - thin-rectangle: `[V2,F2,n] = nxr_compute('normalize', V, F)` → `isequal(V2,V)`,
    `n == 1`, `size(F2) == size(F)`, and `F2` differs from `F`.
  - icosahedron: `n == 0`, `F2` equals `F` as a set (sorted rows).
  - 1-based round-trip: all `F2` entries in `[1, nV]`.

## 5. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | Operation | extrinsic Delaunay edge flips (`fixDelaunay`), vertex-preserving |
| 2 | Output | `[V (passthrough), F' (new, 1-based)]` + optional flip count |
| 3 | Statefulness | stateless command (no handle); pure `(V,F) → (V,F')` |
| 4 | Guarantee | best-effort (big reduction in negatives), **not** a PSD certificate |
| 5 | PSD certificate | deferred to the intrinsic-Delaunay context normalization (separate, geodesic, not a `(V,F')`) |
| 6 | Repair scope | Delaunay flips only; assumes valid manifold input (no hole/non-manifold repair) |

## 6. Deferred (separate follow-on)

The **intrinsic-Delaunay context normalization** (`create(..., intrinsicDelaunay=true)`
→ `SignpostIntrinsicTriangulation` + `flipToDelaunay`, all operators on the one
intrinsic geometry) for *certified* PSD — bigger, refactors the `Manifold` core,
geodesic edges. Spec separately.
