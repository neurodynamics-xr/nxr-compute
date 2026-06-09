# Intrinsic-Delaunay Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Under `create(..., intrinsicDelaunay=true)`, the vertex connection Laplacian (Levi-Civita) and the product covariant become certified-PSD, by routing `assembleConnectionLaplacian`'s matrix assembly through `Manifold::operatorGeometry()`. The probe established `φ_v ≡ 0`, so the frames/grid stay unchanged (no gauge realization).

**Architecture:** Same pattern as Phase 1 — split `geometry` in `assembleConnectionLaplacian` into `operatorGeometry()` (intrinsic weights/transport/mesh-iteration for the `K` matrix) and the embedded `geometry()` (frames). No MEX change (the `intrinsicDelaunay` flag + `buildGaugeOperators` already route through this function).

**Tech Stack:** C++17, Eigen, geometry-central, MATLAB MEX. Build `bash scripts/build.sh Release`. Native binaries in `build/`. MATLAB tests via MATLAB MCP.

**Spec:** `docs/superpowers/specs/2026-06-09-intrinsic-delaunay-phase2-design.md`. Phase 1 is on `main`.

---

## Task 1: reroute `assembleConnectionLaplacian` through `operatorGeometry()`

**Files:** `src/connection_laplacian.cpp`, `test/test_intrinsic_delaunay.cpp` (extend)

- [ ] **Step 1: Read `src/connection_laplacian.cpp`** fully. `assembleConnectionLaplacian` binds `auto& geometry = m.geometry();` (~line 12) and uses it for: (a) the `K` matrix — the halfedge-walk helper(s) using `edgeCotanWeights`, `transportVectorsAlongHalfedge`, `vertexIndices` (for ConnectionDomain::Vertex), and (b) the output frames `frameE1`/`frameE2` from `vertexTangentBasis`. Identify the exact helper(s)/loops for each.

- [ ] **Step 2: Split the geometry source** for the `Vertex` domain:
```cpp
    auto& opGeom  = m.operatorGeometry();   // K matrix: weights, transport, vertex indices, mesh iteration
    auto& embGeom = m.geometry();           // frames: vertexTangentBasis (extrinsic)
```
Route the `K`-assembly accesses (`requireEdgeCotanWeights`/`edgeCotanWeights`,
`requireTransportVectorsAlongHalfedge`/`transportVectorsAlongHalfedge`,
`requireVertexIndices`/`vertexIndices`) and the **halfedge/vertex iteration** to `opGeom`
and **`opGeom.mesh`** (the intrinsic mesh when normalized). Route the frame block
(`requireVertexTangentBasis`/`vertexTangentBasis`, iterating the embedded mesh `m.mesh()`)
to `embGeom`. If the K assembly is in a file-static helper that takes a geometry/mesh,
pass `opGeom` (and `opGeom.mesh`) into it.

IMPORTANT (mesh-iteration alignment, same subtlety as Phase 1's dual-areas): when normalized,
`opGeom.mesh` is the intrinsic mesh — iterate IT for the K walk and index `opGeom.vertexIndices`/
weights/transport against it. Iterate the embedded `m.mesh()` for the frames. Vertices are 1:1
with identical indices, so `frameE1(vi)` aligns with `K` row `vi`.

For the **Face** and **EdgeCrouzeixRaviart** domains and `assembleTrivialConnectionLaplacian`
— LEAVE on `m.geometry()` (out of Phase 2 scope; documented). Only the **Vertex** domain of
`assembleConnectionLaplacian` is rerouted.

Non-normalized path must be byte-identical (`opGeom == *geometry_`).

- [ ] **Step 3: Extend the native test** `test/test_intrinsic_delaunay.cpp` — add a function (call it from `main()`):
```cpp
#include <complex>
// add near the other helpers:
static double minEigC(const Eigen::SparseMatrix<std::complex<double>>& K) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Eigen::MatrixXcd(K));
    return es.eigenvalues().minCoeff();
}

static void testConnectionLaplacian() {
    std::cout << "\n=== intrinsicDelaunay: connection Laplacian ===\n";
    namespace cl = nxr::manifold::ops::laplacian::connection;
    std::vector<double>  V = {0,0,0,  2,0,0,  1,0.2,0,  1,-0.2,0};
    std::vector<int32_t> F = {0,2,1,  0,1,3};
    Manifold mRaw(V.data(), 4, F.data(), 2, false);
    Manifold mN  (V.data(), 4, F.data(), 2, true);

    cl::ConnectionLaplacianOptions o;
    o.domain = cl::ConnectionDomain::Vertex; o.nSym = 1;
    o.format = cl::ConnectionLaplacianFormat::Complex;
    auto Kraw = cl::assembleConnectionLaplacian(mRaw, o);
    auto Kn   = cl::assembleConnectionLaplacian(mN,   o);

    EXPECT(Kn.K_complex.rows()==4, "connection L is V×V");
    // Hermitian + PSD under normalization (the certificate)
    auto herm = (Kn.K_complex - Eigen::SparseMatrix<std::complex<double>>(Kn.K_complex.adjoint())).norm();
    EXPECT(herm < 1e-9, "normalized connection L Hermitian");
    EXPECT(minEigC(Kn.K_complex) > -1e-9, "normalized connection L PSD (min eig >= 0)");
    // frames unchanged (phi_v = 0): raw frameE1 == normalized frameE1
    EXPECT((Kraw.frameE1 - Kn.frameE1).cwiseAbs().maxCoeff() < 1e-12, "frames unchanged (phi_v=0)");
}
```
(If `ConnectionLaplacian::K_complex` is empty for the Complex format on some path, confirm Complex format populates it — it does, per `connection_laplacian.cpp`.)

Add an **icosphere no-op** check too (in the existing icosphere test function or a new one): assemble the Vertex/Complex connection Laplacian on raw + normalized icosphere; assert `(Kraw.K_complex − Kn.K_complex).norm() < 1e-9` (no flips + φ_v=0 ⇒ identical).

- [ ] **Step 4: Build + run** — `bash scripts/build.sh Release 2>&1 | tail -8 && ./build/test_intrinsic_delaunay` → ALL PASSED. Regression: `./build/test_connection_laplacian` and `./build/test_geometry_bundle` must still pass (the reroute must not change the non-normalized connection Laplacian).
  - If the icosphere connection-L no-op fails (normalized ≠ raw on an already-Delaunay mesh), the reroute changed the non-normalized path or the mesh-iteration misaligned — investigate, report; do not weaken.

- [ ] **Step 5: Commit**
```bash
git add src/connection_laplacian.cpp test/test_intrinsic_delaunay.cpp
git commit -m "feat(connection): route vertex connection Laplacian through operatorGeometry (certified PSD)

phi_v ≡ 0 (probe-verified) so frames/grid are unchanged; only the K-matrix weights/
transport route through the intrinsic Delaunay geometry under normalization.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 1)
Read the whole `assembleConnectionLaplacian` + its file-static helpers in `src/connection_laplacian.cpp`. Confirm `IntrinsicGeometryInterface` exposes `.mesh`, `edgeCotanWeights`, `transportVectorsAlongHalfedge`, `vertexIndices` (it does — they're intrinsic-interface members). Confirm `vertexTangentBasis` is on the embedded interface only (so frames must use `m.geometry()`). Follow TDD. Report Status, build result, PASS lines, whether the icosphere connection-L no-op held, regression results, files, commit SHA.

---

## Task 2: MATLAB validation (no MEX code change)

**Files:** `bindings/mex/test/test_intrinsic_delaunay.m` (extend)

- [ ] **Step 1: Extend `test_intrinsic_delaunay.m`** — add (reusing the `hRaw`/`hN` handles already in the test):
```matlab
% --- Phase 2: connection Laplacian + product covariant certified-PSD ---
GlN = nxr_compute('gauge', hN, 'levi-civita', struct('operators',true));
K = GlN.operators.laplacian;                 % V×V complex connection Laplacian
assert(norm(K - K','fro') < 1e-9, 'normalized connection L Hermitian');
assert(min(eig(full(K))) > -1e-9, 'normalized connection L PSD');

% product covariant under normalization is symmetric + PSD
GpN = nxr_compute('gauge', hN, 'levi-civita', struct('operators',true,'coupling','product'));
Cp = GpN.operators.covariantLaplacian;
assert(norm(Cp - Cp','fro') < 1e-9, 'product covariant symmetric');
assert(min(eig(full(Cp))) > -1e-9, 'product covariant PSD under normalization');

% grid unchanged (phi_v = 0): normalized grid == raw grid
GgeoRaw = nxr_compute('geometry', hRaw);
GgeoN   = nxr_compute('geometry', hN);
assert(max(abs(GgeoRaw.vertex.grid(:) - GgeoN.vertex.grid(:))) < 1e-12, ...
       'grid unchanged under normalization (phi_v = 0)');
```
(Place these before the `destroy` calls; `hRaw`/`hN` already exist from the Phase-1 part of the test.)

- [ ] **Step 2: Build (no source change expected) + run** — `bash scripts/build.sh Release 2>&1 | tail -4`. Via MATLAB MCP run `bindings/mex/test/test_intrinsic_delaunay.m` → ALL TESTS PASSED. Also re-run `test_operators.m` and `test_bundle.m` (non-normalized connection Laplacian + covariant must be unchanged). If MATLAB MCP unavailable, report DONE_WITH_CONCERNS (native only — Task 1 already covers the native cert).

- [ ] **Step 3: Commit**
```bash
git add bindings/mex/test/test_intrinsic_delaunay.m
git commit -m "test(intrinsic): connection L + product covariant PSD + grid unchanged under normalization

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 2)
Read the current `bindings/mex/test/test_intrinsic_delaunay.m` to reuse `hRaw`/`hN`. Report Status, build result, MATLAB outputs (verbatim or unavailable), files, commit SHA.

---

## Self-Review
| Spec item | Task |
|---|---|
| connection L K-matrix → operatorGeometry(); frames → embedded | Task 1 |
| certified PSD connection L under normalization | Task 1 native + Task 2 MATLAB |
| product covariant inherits PSD | Task 2 |
| grid unchanged (φ_v=0) | Task 1 (frames) + Task 2 (grid) |
| non-normalized byte-identical | Task 1 (default + regression), Task 2 (test_operators/test_bundle) |
| trivial-gauge / Face / Edge deferred | scope note, not rerouted |

**Placeholders:** none. **Consistency:** reuses Phase-1's `operatorGeometry()`/`isIntrinsicDelaunay()` + the rhombus/icosphere fixtures.
