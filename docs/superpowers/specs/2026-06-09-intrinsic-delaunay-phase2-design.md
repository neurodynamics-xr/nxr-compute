# Intrinsic-Delaunay context normalization — Phase 2 (certified-PSD connection Laplacian + product covariant)

**Date:** 2026-06-09
**Status:** Design — approved in discussion, pending spec review
**Builds on:** Phase 1 (`2026-06-09-intrinsic-delaunay-phase1-design.md`) — the `intrinsicDelaunay`
context flag + `Manifold::operatorGeometry()`. Phase 1 made the cotan Laplacian + mass +
ambient covariant certified-PSD.

---

## 1. The probe result that defines Phase 2

A probe measured the per-vertex intrinsic↔input tangent-gauge rotation `φ_v` directly
(branch `probe/phi-v`, not merged): across three fixtures (1, 4, and 8 flips, including
flip-incident vertices), **`φ_v ≡ 0` to machine precision.** The mechanism:
`SignpostIntrinsicTriangulation` sets `signpostAngle[v.halfedge()] = 0` at construction
and the flip code maintains that invariant, so the intrinsic triangulation's
`halfedgeVectorsInVertex` gauge stays anchored to the **same reference direction** as the
input geometry. The intrinsic connection Laplacian is therefore **already consistent with
our existing input `grid` frames** — no `exp(iφ_v)` rotation, no grid realization.

So Phase 2 reduces to the Phase-1 pattern: route the connection Laplacian's weights/transport
through `operatorGeometry()` when normalized; the frames stay on the embedded `geometry()`
(consistent precisely because `φ_v = 0`).

## 2. Goal

Under `create(..., intrinsicDelaunay=true)`:
- **`Gauge.operators.laplacian`** (vertex connection Laplacian, Levi-Civita, any `nSym`) is
  assembled on the intrinsic Delaunay geometry → **certified PSD** (intrinsic weights ≥ 0;
  the connection Laplacian `Σ w_ij(…)` is Hermitian-PSD iff `w_ij ≥ 0`).
- **product `covariantLaplacian`** = `blkdiag(K, cotanL)` inherits PSD (both intrinsic now —
  `K` from this change, `cotanL` from Phase 1).
- The `grid` is **unchanged** (input grid; `φ_v = 0`), so the connection Laplacian's complex
  eigenvectors / the leadfield `z` decode consistently against it.

## 3. Change — reroute the connection-Laplacian *matrix* through `operatorGeometry()`

In `src/connection_laplacian.cpp`, `assembleConnectionLaplacian` currently binds a single
`auto& geometry = m.geometry();` and uses it for BOTH the `K` matrix assembly (the halfedge
walk over `edgeCotanWeights` × `transportVectorsAlongHalfedge`, plus `vertexIndices` and the
mesh iteration) AND the output frames (`vertexTangentBasis`). Split it:

```cpp
    auto& opGeom  = m.operatorGeometry();   // K assembly: weights, transport, vertex indices, halfedge/vertex iteration
    auto& embGeom = m.geometry();           // frames: vertexTangentBasis (extrinsic — stays embedded)
```
- The `K` assembly (the file-static halfedge-walk helper(s) and the diagonal/off-diagonal
  triplets) sources `edgeCotanWeights`, `transportVectorsAlongHalfedge`, `vertexIndices`
  from `opGeom`, and **iterates `opGeom.mesh`** (the intrinsic mesh when normalized). Pass
  `opGeom` into the helper(s) instead of `m.geometry()`.
- The frame block (`frameE1`/`frameE2` from `vertexTangentBasis`) sources from `embGeom`
  and iterates the embedded mesh (`m.mesh()`). Vertices are 1:1 with identical indices
  (Phase-1 invariant), so `frameE1(vi)` aligns with `K`'s row `vi`.

Non-normalized path is byte-identical (`opGeom == *geometry_ == embGeom.base`).

**`nSym` works for free:** the helper applies `transport^nSym`; routing the transport through
`opGeom` gives the intrinsic `transport^nSym`. All `nSym` get the intrinsic operator.

**Scope (Phase 2):** `ConnectionDomain::Vertex`, Levi-Civita (`assembleConnectionLaplacian`).
Deferred:
- **`assembleTrivialConnectionLaplacian`** (trivial gauge) — its `computeTrivialConnection`
  Poisson solve uses the DEC (`d0`, `★₁`), which is edge-indexed; routing it needs the
  intrinsic DEC (Phase 3). Under normalization, `Gauge.operators.laplacian` for the
  **trivial** gauge stays on the original mesh (documented). Euclidean/levi-civita are intrinsic.
- **`Face`/`EdgeCrouzeixRaviart`** domains — intrinsic faces/edges are re-triangulated, so
  these become intrinsic-indexed (analogous to the DEC Phase-3 caveat). Phase 2 validates the
  Vertex domain (what `Gauge.operators.laplacian` + the covariant use).

## 4. No MEX change

The `create(..., intrinsicDelaunay=true)` flag already exists (Phase 1). `buildGaugeOperators`
already calls `assembleConnectionLaplacian` (for euclidean/levi-civita) → it inherits the
reroute. The product `covariantLaplacian` already takes `K` + `cotanL` → both intrinsic now.
Only a new MATLAB test is added.

## 5. Test plan

- **Native** (`test/test_intrinsic_delaunay.cpp`, extend): on the non-Delaunay rhombus,
  - build raw + normalized `Manifold`; `assembleConnectionLaplacian({Vertex, nSym=1, Complex})`
    on each.
  - normalized `K_complex` is Hermitian (`‖K−Kᴴ‖<1e-9`) and **PSD**
    (`SelfAdjointEigenSolver<MatrixXcd>` min eig `≥ −1e-9`) — the certificate.
  - both `V×V` on the same vertices; `frameE1` identical raw vs normalized (frames unchanged,
    `φ_v=0`).
  - **icosphere no-op:** normalized `K_complex == raw K_complex` (`‖·‖<1e-9`) — already
    Delaunay ⇒ no flips ⇒ identical, and `φ_v=0` ⇒ no gauge drift.
- **MATLAB** (`bindings/mex/test/test_intrinsic_delaunay.m`, extend):
  - `Gauge.operators.laplacian` (levi-civita, `operators=true`) under normalization is
    Hermitian + `min(eig) ≥ −1e-9` (PSD).
  - **product** `covariantLaplacian` (`coupling='product'`) under normalization is symmetric +
    PSD.
  - **grid unchanged:** `Geometry.vertex.grid` is identical (`< 1e-12`) between the raw and
    normalized handles — the end-to-end confirmation that `φ_v = 0` (no grid realization).

## 6. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | `φ_v` | `≡ 0` (probe-verified, machine precision) ⇒ no grid realization, grid unchanged |
| 2 | reroute | `assembleConnectionLaplacian` K-matrix → `operatorGeometry()`; frames → embedded `geometry()` |
| 3 | covariant | product inherits (`K` + `cotanL` both intrinsic); ambient already PSD (Phase 1) |
| 4 | scope | Vertex domain, Levi-Civita; trivial-gauge connection + Face/Edge domains deferred (Phase 3) |
| 5 | MEX | no change (flag + builders already route through the rerouted function) |
| 6 | non-normalized | byte-identical (`operatorGeometry()` returns `*geometry_`) |

## 7. Deferred (Phase 3)

Fully-intrinsic `Topology`/`Geometry`/DEC bundle (one mesh everywhere) — makes the DEC, the
trivial connection, and the Face/Edge connection Laplacians intrinsic, and restores the
cross-DEC identity under normalization.
