# Topology / Geometry / Gauge Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three MEX commands (`topology`, `geometry`, `gauge`) plus a `bundle` convenience that hand Brainstorm a halfedge-mesh coordinate system over the cortical surface, with the per-element frame encoded as a complex 3-vector (`grid = e1 + i·e2`) and the Gauge expressed as a transform of it.

**Architecture:** Algorithmic pieces (the complex `grid`, the 2-RoSy `curvature`, the trivial-gauge vertex rotation) become small focused functions in the `nxr_compute` library with native C++ tests. The MATLAB-facing structs are assembled and marshaled in the MEX binding (complex arrays via `mxCOMPLEX`, indices converted 0-based→1-based at the boundary), validated by MATLAB assert-scripts run through the MATLAB MCP.

**Tech Stack:** C++17, Eigen 3.x (incl. `MatrixXcd`/`VectorXcd`), geometry-central, MATLAB MEX (`nxr_compute.mexmaca64`). Build: `bash scripts/build.sh Release`. Native tests: `./build/Release/test_*.exe`. MATLAB tests: MATLAB MCP `run_matlab_file`.

**Spec:** `docs/superpowers/specs/2026-06-08-topology-geometry-gauge-bundle-design.md`

---

## File Map

| File | Responsibility | Change |
|---|---|---|
| `include/nxr/compute.h` | Library API | Declare `vertexGrid`, `faceGrid`, `VertexCurvature2RoSy`+`vertexCurvature`, `GaugeRotations`+`integrateTrivialGaugeRotations` |
| `src/geometry_grid.cpp` | grid + curvature assembly | **Create** — `vertexGrid`, `faceGrid`, `vertexCurvature` |
| `src/gauge.cpp` | trivial-gauge integration | **Create** — `integrateTrivialGaugeRotations` |
| `test/test_geometry_bundle.cpp` | native tests | **Create** |
| `CMakeLists.txt` | build | Register `test_geometry_bundle` |
| `bindings/mex/src/marshal.h` | mxArray converters | Add complex + 1-based-index converters |
| `bindings/mex/src/nxr_compute_mex.cpp` | MEX dispatch | Add `cmdTopology`, `cmdGeometry`, `cmdGauge`, `cmdBundle` + wiring |
| `bindings/mex/test/test_topology.m` | MATLAB test | **Create** |
| `bindings/mex/test/test_geometry_bundle.m` | MATLAB test | **Create** |
| `bindings/mex/test/test_gauge.m` | MATLAB test | **Create** |
| `bindings/mex/test/test_bundle.m` | MATLAB test | **Create** |

**Conventions to mirror (read before starting):**
- `src/vertex_frames.cpp` — `vertexFrames(Manifold&)` returns `VertexFrames{e1,e2,normals}` (V×3 each). `src/face_frames.cpp` — `frames(Manifold&)` returns `FaceFrames`.
- `src/curvatures.cpp` — `curvatures(Manifold&)` returns `CurvatureResult{gaussian,mean,kMin,kMax,principalDirMax}`.
- `src/direction_field.cpp` — `nxr::manifold::connection::computeTrivialConnection(m, dec, cache, singMap)` returns per-edge φ (`Eigen::VectorXd`, length nE).
- `bindings/mex/src/marshal.h` — `eigenMatrixToMx`, `eigenVectorToMx`, `getIntArg`, `getDoubleArg`, `getStringArg`, `mxToVertexIndices`, `mxToEigenVector`.
- `bindings/mex/test/test_vertex_frames.m` — the `local_icosahedron()` fixture + `create`/call/`destroy` flow to copy into each new MATLAB test.

**Out of scope (deferred per spec §10):** the `Operators` surface, `MeshData`, and `Gauge.face.rotation` for the `trivial` gauge (the field is emitted but left empty in v1 — see Task 6).

---

## Task 1: `vertexGrid` / `faceGrid` — the complex frame

**Files:**
- Modify: `include/nxr/compute.h` (namespace `nxr::manifold::geometry`, near `VertexFrames`/`vertexFrames`, ~line 1165)
- Create: `src/geometry_grid.cpp`
- Create: `test/test_geometry_bundle.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Declare the functions in `compute.h`**

In `namespace nxr::manifold::geometry`, immediately after the `VertexFrames vertexFrames(Manifold& m);` declaration, add:

```cpp
// ── Complex frame ("grid") ───────────────────────────────────
//
// The per-element tangent frame packed as a complex 3-vector
// c = e1 + i·e2 ∈ ℂ³ (real part = e1, imag part = e2). The unit
// normal is recovered as real(c) × imag(c) and is not stored
// separately. This is the representation Brainstorm's Geometry.*.grid
// fields use; see the bundle design spec §5.1.
Eigen::MatrixXcd vertexGrid(Manifold& m);   // [nV, 3] complex
Eigen::MatrixXcd faceGrid(Manifold& m);     // [nF, 3] complex
```

- [ ] **Step 2: Write the failing native test**

Create `test/test_geometry_bundle.cpp`:

```cpp
#include "nxr/compute.h"
#include <iostream>
#include <cmath>
#include <complex>

using nxr::manifold::Manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { \
    std::cerr << "  [FAIL] " << msg << "\n"; ++g_failures; } \
    else { std::cout << "  [PASS] " << msg << "\n"; } } while (0)

// Unit icosahedron (12 verts, 20 faces) — same fixture as the MATLAB tests.
static void makeIcosahedron(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double verts[12][3] = {
        {-1, t, 0},{1, t, 0},{-1,-t, 0},{1,-t, 0},
        {0,-1, t},{0, 1, t},{0,-1,-t},{0, 1,-t},
        {t, 0,-1},{t, 0, 1},{-t, 0,-1},{-t, 0, 1}};
    for (auto& r : verts) {
        double n = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        r[0]/=n; r[1]/=n; r[2]/=n;
    }
    V.assign(&verts[0][0], &verts[0][0]+36);
    int faces[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
    F.assign(&faces[0][0], &faces[0][0]+60);
}

static void testVertexGrid() {
    std::cout << "\n=== vertexGrid ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    Eigen::MatrixXcd c = nxr::manifold::geometry::vertexGrid(m);
    EXPECT(c.rows() == 12 && c.cols() == 3, "vertexGrid is [nV, 3]");

    // real(c)=e1, imag(c)=e2 must be unit-length and orthogonal; their cross
    // product must be a unit normal.
    double maxE1Err = 0, maxE2Err = 0, maxDot = 0, maxCrossErr = 0;
    for (int v = 0; v < 12; ++v) {
        Eigen::Vector3d e1 = c.row(v).real(), e2 = c.row(v).imag();
        maxE1Err = std::max(maxE1Err, std::abs(e1.norm() - 1.0));
        maxE2Err = std::max(maxE2Err, std::abs(e2.norm() - 1.0));
        maxDot   = std::max(maxDot, std::abs(e1.dot(e2)));
        maxCrossErr = std::max(maxCrossErr, std::abs(e1.cross(e2).norm() - 1.0));
    }
    EXPECT(maxE1Err < 1e-9, "real(c) unit length");
    EXPECT(maxE2Err < 1e-9, "imag(c) unit length");
    EXPECT(maxDot   < 1e-9, "real(c) ⟂ imag(c)");
    EXPECT(maxCrossErr < 1e-9, "real(c) × imag(c) is unit normal");
}

int main() {
    testVertexGrid();
    if (g_failures) { std::cerr << "\n" << g_failures << " failure(s)\n"; return 1; }
    std::cout << "\nALL PASSED\n"; return 0;
}
```

- [ ] **Step 3: Register the test in `CMakeLists.txt`**

After the `test_connection_laplacian` block (~line 163), add:

```cmake
    add_executable(test_geometry_bundle test/test_geometry_bundle.cpp)
    target_link_libraries(test_geometry_bundle PRIVATE nxr_compute)
    add_test(NAME test_geometry_bundle COMMAND test_geometry_bundle)
```

- [ ] **Step 4: Build to verify the test fails to link**

Run: `bash scripts/build.sh Release 2>&1 | grep -E "error:|vertexGrid|undefined"`
Expected: undefined-reference (link) error for `vertexGrid` — confirms the test compiles and binds to the not-yet-implemented symbol.

- [ ] **Step 5: Implement `src/geometry_grid.cpp`**

Create `src/geometry_grid.cpp`:

```cpp
#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace nxr::manifold::geometry {

// c = e1 + i·e2, built directly on the existing frame assembly so the
// tangent basis matches everything else (connection Laplacian, transport).
Eigen::MatrixXcd vertexGrid(Manifold& m) {
    VertexFrames vf = vertexFrames(m);
    int nV = m.nV();
    Eigen::MatrixXcd c(nV, 3);
    for (int v = 0; v < nV; ++v)
        for (int k = 0; k < 3; ++k)
            c(v, k) = std::complex<double>(vf.e1(v, k), vf.e2(v, k));
    return c;
}

Eigen::MatrixXcd faceGrid(Manifold& m) {
    FaceFrames ff = frames(m);
    int nF = m.nF();
    Eigen::MatrixXcd c(nF, 3);
    for (int f = 0; f < nF; ++f)
        for (int k = 0; k < 3; ++k)
            c(f, k) = std::complex<double>(ff.e1(f, k), ff.e2(f, k));
    return c;
}

} // namespace nxr::manifold::geometry
```

Add `src/geometry_grid.cpp` to the library sources in `CMakeLists.txt` (find the `add_library(nxr_compute ...)` source list and append `src/geometry_grid.cpp`).

- [ ] **Step 6: Build and run the native test**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_geometry_bundle`
Expected: `ALL PASSED` with 4 `[PASS]` lines under `=== vertexGrid ===`.

- [ ] **Step 7: Commit**

```bash
git add include/nxr/compute.h src/geometry_grid.cpp test/test_geometry_bundle.cpp CMakeLists.txt
git commit -m "feat(geometry): add complex-frame vertexGrid/faceGrid (c = e1 + i·e2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `vertexCurvature` — 2-RoSy deviatoric + mean

**Files:**
- Modify: `include/nxr/compute.h` (namespace `nxr::manifold::geometry`)
- Modify: `src/geometry_grid.cpp`
- Modify: `test/test_geometry_bundle.cpp`

- [ ] **Step 1: Declare in `compute.h`**

After the `faceGrid` declaration, add:

```cpp
// ── Curvature as a 2-RoSy complex + mean ─────────────────────
//
// The shape operator (3 real DOF) split into its trace and deviatoric
// parts. `deviatoric` is the traceless symmetric part encoded as a
// 2-RoSy complex number q: arg(q)/2 is the max principal direction
// angle in the vertex tangent basis, |q| = (κmax − κmin)/2. `mean`
// is (κmax + κmin)/2. See bundle design spec §5.2. Everything else
// (principal curvatures/directions, extrinsic Gaussian) derives from
// these; intrinsic Gaussian comes from the angle-defect (2π − angleSum).
struct VertexCurvature2RoSy {
    Eigen::VectorXcd deviatoric;  // [nV] q
    Eigen::VectorXd  mean;        // [nV] H
};
VertexCurvature2RoSy vertexCurvature(Manifold& m);
```

- [ ] **Step 2: Add the failing test to `test/test_geometry_bundle.cpp`**

Add this function and call it from `main()` (before the `if (g_failures)` line, after `testVertexGrid();`):

```cpp
static void testVertexCurvature() {
    std::cout << "\n=== vertexCurvature ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    auto cv = nxr::manifold::geometry::vertexCurvature(m);
    EXPECT(cv.deviatoric.size() == 12, "deviatoric length == nV");
    EXPECT(cv.mean.size() == 12, "mean length == nV");

    // Unit icosahedron approximates a unit sphere: nearly umbilic, so the
    // deviatoric magnitude is small relative to the mean, and the mean
    // (≈ principal curvature ≈ 1/R = 1) is positive (convex).
    double maxDev = cv.deviatoric.cwiseAbs().maxCoeff();
    double minMean = cv.mean.minCoeff();
    EXPECT(minMean > 0.5, "mean curvature positive (convex), ~1 on unit sphere");
    EXPECT(maxDev < 0.5 * minMean, "deviatoric small vs mean (near-umbilic)");

    // Extrinsic Gaussian K = H² − |q|² must be positive on a convex surface.
    bool allKpos = true;
    for (int v = 0; v < 12; ++v) {
        double K = cv.mean(v) * cv.mean(v) - std::norm(cv.deviatoric(v));
        if (K <= 0) allKpos = false;
    }
    EXPECT(allKpos, "extrinsic Gaussian H² − |q|² > 0 (convex)");

    // Determinism.
    auto cv2 = nxr::manifold::geometry::vertexCurvature(m);
    EXPECT((cv.deviatoric - cv2.deviatoric).cwiseAbs().maxCoeff() < 1e-15 &&
           (cv.mean - cv2.mean).cwiseAbs().maxCoeff() < 1e-15,
           "vertexCurvature deterministic");
}
```

- [ ] **Step 3: Build to verify failure**

Run: `bash scripts/build.sh Release 2>&1 | grep -E "error:|vertexCurvature|undefined"`
Expected: undefined-reference for `vertexCurvature`.

- [ ] **Step 4: Implement `vertexCurvature` in `src/geometry_grid.cpp`**

Add to `src/geometry_grid.cpp` inside `namespace nxr::manifold::geometry` (and add `#include <cmath>` at the top):

```cpp
// Build q from the validated principal-curvature data. We take the magnitude
// from (κmax − κmin)/2 and the phase from the SAME 3D max-principal-direction
// the existing curvatures() lift produces, projected back into the (e1,e2)
// tangent basis. Going through the 3D direction makes this independent of
// geometry-central's internal 2-RoSy power convention.
VertexCurvature2RoSy vertexCurvature(Manifold& m) {
    CurvatureResult cr = curvatures(m);
    VertexFrames    vf = vertexFrames(m);
    int nV = m.nV();

    VertexCurvature2RoSy out;
    out.deviatoric.resize(nV);
    out.mean.resize(nV);
    for (int v = 0; v < nV; ++v) {
        Eigen::RowVector3d dir = cr.principalDirMax.row(v);
        Eigen::RowVector3d e1  = vf.e1.row(v);
        Eigen::RowVector3d e2  = vf.e2.row(v);
        double a = dir.dot(e1);
        double b = dir.dot(e2);
        double theta = std::atan2(b, a);
        double mag   = 0.5 * (cr.kMax(v) - cr.kMin(v));
        out.deviatoric(v) = std::polar(mag, 2.0 * theta);
        out.mean(v)       = 0.5 * (cr.kMax(v) + cr.kMin(v));
    }
    return out;
}
```

- [ ] **Step 5: Build and run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_geometry_bundle`
Expected: `ALL PASSED`, including the 6 new `[PASS]` lines under `=== vertexCurvature ===`.

- [ ] **Step 6: Commit**

```bash
git add include/nxr/compute.h src/geometry_grid.cpp test/test_geometry_bundle.cpp
git commit -m "feat(geometry): add vertexCurvature (2-RoSy deviatoric q + mean H)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `integrateTrivialGaugeRotations` — vertex rotation field

**Files:**
- Modify: `include/nxr/compute.h` (namespace `nxr::manifold::connection`, near `computeTrivialConnection`, ~line 946)
- Create: `src/gauge.cpp`
- Modify: `test/test_geometry_bundle.cpp`
- Modify: `CMakeLists.txt` (add `src/gauge.cpp` to library sources)

- [ ] **Step 1: Declare in `compute.h`**

In `namespace nxr::manifold::connection`, after the `computeTrivialConnection(...)` declaration, add:

```cpp
// ── Trivial-gauge per-vertex rotation ────────────────────────
//
// Integrates the trivial-connection 1-form φ (from computeTrivialConnection)
// into a per-vertex unit-complex rotation r_v = exp(iθ_v) relative to the
// Levi-Civita vertex frame, by BFS over the vertex graph from a root,
// accumulating the Levi-Civita transport modulated by exp(i·sign·φ[edge]).
// The realized trivial gauge frame is r .* vertexGrid (broadcast over the
// 3 complex columns). See bundle design spec §6.
//
// Gauss-Bonnet: Σ singularityMap values must equal χ(mesh) (the caller's
// responsibility — not checked here).
struct GaugeRotations {
    Eigen::VectorXcd vertex;   // [nV] r_v = exp(iθ_v), |r_v| = 1
};
GaugeRotations integrateTrivialGaugeRotations(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap);
```

- [ ] **Step 2: Add the failing test**

Add to `test/test_geometry_bundle.cpp` and call from `main()`:

```cpp
static void testTrivialGaugeRotations() {
    std::cout << "\n=== integrateTrivialGaugeRotations ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    auto dec   = nxr::manifold::ops::assembleDECOperators(m);
    auto cache = nxr::manifold::ops::CholeskyCache{};
    // χ(sphere) = 2: two index-1 singularities (Gauss-Bonnet satisfied).
    std::map<int, double> sing = {{0, 1.0}, {1, 1.0}};

    auto g = nxr::manifold::connection::integrateTrivialGaugeRotations(m, dec, cache, sing);
    EXPECT(g.vertex.size() == 12, "rotation length == nV");

    // Every rotation is unit modulus (it's a pure tangent-plane rotation).
    double maxModErr = (g.vertex.cwiseAbs().array() - 1.0).abs().maxCoeff();
    EXPECT(maxModErr < 1e-9, "all rotations unit modulus");

    // The trivial gauge differs from Levi-Civita (rotations not all equal):
    // φ from prescribed singularities is non-trivial.
    double spread = (g.vertex.array() - g.vertex(0)).abs().maxCoeff();
    EXPECT(spread > 1e-6, "trivial gauge differs from Levi-Civita");

    // Determinism (same root, same BFS order).
    auto g2 = nxr::manifold::connection::integrateTrivialGaugeRotations(m, dec, cache, sing);
    EXPECT((g.vertex - g2.vertex).cwiseAbs().maxCoeff() < 1e-15,
           "rotation field deterministic");
}
```

- [ ] **Step 3: Build to verify failure**

Run: `bash scripts/build.sh Release 2>&1 | grep -E "error:|integrateTrivialGaugeRotations|undefined"`
Expected: undefined-reference for `integrateTrivialGaugeRotations`.

- [ ] **Step 4: Implement `src/gauge.cpp`**

Create `src/gauge.cpp`:

```cpp
#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <queue>
#include <vector>
#include <complex>

namespace nxr::manifold::connection {

using namespace geometrycentral;
using namespace geometrycentral::surface;

GaugeRotations integrateTrivialGaugeRotations(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap) {

    // 1. Trivial-connection 1-form φ (per edge). Sign convention: φ(e) is
    //    positive along e.halfedge() (see computeTrivialConnection docs).
    Eigen::VectorXd phi = computeTrivialConnection(m, dec, cache, singularityMap);

    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireVertexIndices();
    geom.requireEdgeIndices();
    geom.requireTransportVectorsAlongHalfedge();

    int nV = m.nV();
    GaugeRotations out;
    out.vertex.resize(nV);
    std::vector<char> visited(nV, 0);

    // 2. BFS over vertices from vertex 0; r[root] = 1.
    Vertex root = mesh.vertex(0);
    int rootIdx = static_cast<int>(geom.vertexIndices[root]);
    out.vertex(rootIdx) = std::complex<double>(1.0, 0.0);
    visited[rootIdx] = 1;

    std::queue<Vertex> q;
    q.push(root);
    while (!q.empty()) {
        Vertex u = q.front(); q.pop();
        int ui = static_cast<int>(geom.vertexIndices[u]);
        for (Halfedge he : u.outgoingHalfedges()) {
            Vertex w = he.tipVertex();
            int wi = static_cast<int>(geom.vertexIndices[w]);
            if (visited[wi]) continue;

            // Levi-Civita transport u-frame → w-frame along he, as a unit complex.
            Vector2 t = geom.transportVectorsAlongHalfedge[he];
            std::complex<double> rho(t.x, t.y);

            // Trivial correction exp(i·sign·φ[edge]); sign tracks edge orientation.
            double sign = (he == he.edge().halfedge()) ? 1.0 : -1.0;
            int e = static_cast<int>(geom.edgeIndices[he.edge()]);
            std::complex<double> corr = std::polar(1.0, sign * phi(e));

            std::complex<double> r = out.vertex(ui) * rho * corr;
            out.vertex(wi) = r / std::abs(r);   // renormalize against drift
            visited[wi] = 1;
            q.push(w);
        }
    }
    return out;
}

} // namespace nxr::manifold::connection
```

Add `src/gauge.cpp` to the `add_library(nxr_compute ...)` source list in `CMakeLists.txt`.

- [ ] **Step 5: Build and run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_geometry_bundle`
Expected: `ALL PASSED`, including 4 new `[PASS]` lines under `=== integrateTrivialGaugeRotations ===`.

If "trivial gauge differs from Levi-Civita" fails (spread ~0), φ integrated to a trivial field — check `computeTrivialConnection` returns non-zero (`phi.norm() > 0`). If unit-modulus fails, the per-step renormalization should prevent drift; investigate `transportVectorsAlongHalfedge` not being unit (it should be).

- [ ] **Step 6: Commit**

```bash
git add include/nxr/compute.h src/gauge.cpp test/test_geometry_bundle.cpp CMakeLists.txt
git commit -m "feat(gauge): add integrateTrivialGaugeRotations (per-vertex BFS of φ)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: MEX marshaling helpers — complex arrays + 1-based indices

**Files:**
- Modify: `bindings/mex/src/marshal.h`

- [ ] **Step 1: Add the converters**

In `bindings/mex/src/marshal.h`, after `eigenVectorToMx` (~line 104), add:

```cpp
// ── Complex dense conversion (interleaved-complex API, R2018a+) ──
inline mxArray* eigenComplexMatrixToMx(const Eigen::MatrixXcd& m) {
    mxArray* arr = mxCreateDoubleMatrix(m.rows(), m.cols(), mxCOMPLEX);
    mxComplexDouble* p = mxGetComplexDoubles(arr);
    mwSize idx = 0;
    for (Eigen::Index j = 0; j < m.cols(); ++j)        // column-major (MATLAB native)
        for (Eigen::Index i = 0; i < m.rows(); ++i, ++idx) {
            p[idx].real = m(i, j).real();
            p[idx].imag = m(i, j).imag();
        }
    return arr;
}

inline mxArray* eigenComplexVectorToMx(const Eigen::VectorXcd& v) {
    return eigenComplexMatrixToMx(v);  // N×1
}

// ── 1-based index column from 0-based values ─────────────────
// The single MEX-boundary conversion point: 0-based C++/geometry-central
// indices become 1-based MATLAB uint32; a negative value (geometry-central
// INVALID / "none") becomes the 0 sentinel.
inline mxArray* indexVectorToMx1Based(const std::vector<long>& idx0) {
    mxArray* arr = mxCreateNumericMatrix(idx0.size(), 1, mxUINT32_CLASS, mxREAL);
    uint32_t* p = static_cast<uint32_t*>(mxGetData(arr));
    for (size_t i = 0; i < idx0.size(); ++i)
        p[i] = (idx0[i] < 0) ? 0u : static_cast<uint32_t>(idx0[i] + 1);
    return arr;
}

inline mxArray* logicalVectorToMx(const std::vector<char>& b) {
    mxArray* arr = mxCreateLogicalMatrix(b.size(), 1);
    mxLogical* p = mxGetLogicals(arr);
    for (size_t i = 0; i < b.size(); ++i) p[i] = b[i] ? 1 : 0;
    return arr;
}

inline mxArray* scalarToMx(double x) {
    mxArray* arr = mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(arr) = x;
    return arr;
}
```

- [ ] **Step 2: Build to verify the header compiles**

Run: `bash scripts/build.sh Release 2>&1 | grep -E "error:|marshal.h"`
Expected: no errors (the helpers are header-only and unused until later tasks; a clean build means they compile).

- [ ] **Step 3: Commit**

```bash
git add bindings/mex/src/marshal.h
git commit -m "feat(mex): add complex-array + 1-based-index marshaling helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `topology` command

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`
- Create: `bindings/mex/test/test_topology.m`

- [ ] **Step 1: Add `cmdTopology`**

In `bindings/mex/src/nxr_compute_mex.cpp`, before the dispatch block (~line 1100), add. The `idxOrNone` lambda returns -1 for an invalid element so `indexVectorToMx1Based` maps it to the 0 sentinel.

```cpp
void cmdTopology(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    using namespace geometrycentral::surface;
    if (nrhs < 2) throw std::invalid_argument("nxr_compute('topology', handle)");
    ContextHolder& h = getHolder(prhs[1]);
    auto& mesh = h.ctx->mesh();
    auto& geom = h.ctx->geometry();
    geom.requireVertexIndices();
    geom.requireEdgeIndices();
    geom.requireFaceIndices();

    const int nV = h.ctx->nV(), nE = h.ctx->nE(), nF = h.ctx->nF();
    const int nH = static_cast<int>(mesh.nHalfedges());
    const int nC = static_cast<int>(mesh.nCorners());

    auto idxOrNone = [](size_t i) -> long {
        return (i == geometrycentral::INVALID_IND) ? -1L : static_cast<long>(i);
    };

    // Reverse lookups.
    std::vector<long> vHe(nV), eHe(nE), fHe(nF), cHe(nC);
    for (Vertex v : mesh.vertices()) vHe[v.getIndex()] = idxOrNone(v.halfedge().getIndex());
    for (Edge e : mesh.edges())      eHe[e.getIndex()] = idxOrNone(e.halfedge().getIndex());
    for (Face f : mesh.faces())      fHe[f.getIndex()] = idxOrNone(f.halfedge().getIndex());
    for (Corner c : mesh.corners())  cHe[c.getIndex()] = idxOrNone(c.halfedge().getIndex());

    // Halfedge table.
    std::vector<long> heTwin(nH), heNext(nH), heVert(nH), heEdge(nH), heFace(nH), heCorner(nH);
    std::vector<char> heOrient(nH), heInterior(nH);
    for (Halfedge he : mesh.halfedges()) {
        size_t i = he.getIndex();
        heTwin[i]  = idxOrNone(he.twin().getIndex());
        heNext[i]  = idxOrNone(he.next().getIndex());
        heVert[i]  = idxOrNone(he.vertex().getIndex());
        heEdge[i]  = idxOrNone(he.edge().getIndex());
        heFace[i]  = he.isInterior() ? idxOrNone(he.face().getIndex()) : -1L;
        heCorner[i]= he.isInterior() ? idxOrNone(he.corner().getIndex()) : -1L;
        heOrient[i]   = (he == he.edge().halfedge()) ? 1 : 0;
        heInterior[i] = he.isInterior() ? 1 : 0;
    }

    auto group = [](std::initializer_list<const char*> names) {
        return std::vector<const char*>(names);
    };
    // Build nested element-type structs.
    const char* topFields[] = {"schemaVersion","vertex","edge","face","corner","halfedge"};
    mxArray* s = mxCreateStructMatrix(1, 1, 6, topFields);
    mxSetField(s, 0, "schemaVersion", scalarToMx(1));

    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(nV)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(vHe));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(nE)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(eHe));
      mxSetField(s,0,"edge",g); }
    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(nF)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(fHe));
      mxSetField(s,0,"face",g); }
    { const char* f[] = {"count","halfedge"}; mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"count",scalarToMx(nC)); mxSetField(g,0,"halfedge",indexVectorToMx1Based(cHe));
      mxSetField(s,0,"corner",g); }
    { const char* f[] = {"count","twin","next","vertex","edge","face","corner","orientation","isInterior"};
      mxArray* g = mxCreateStructMatrix(1,1,9,f);
      mxSetField(g,0,"count",scalarToMx(nH));
      mxSetField(g,0,"twin",indexVectorToMx1Based(heTwin));
      mxSetField(g,0,"next",indexVectorToMx1Based(heNext));
      mxSetField(g,0,"vertex",indexVectorToMx1Based(heVert));
      mxSetField(g,0,"edge",indexVectorToMx1Based(heEdge));
      mxSetField(g,0,"face",indexVectorToMx1Based(heFace));
      mxSetField(g,0,"corner",indexVectorToMx1Based(heCorner));
      mxSetField(g,0,"orientation",logicalVectorToMx(heOrient));
      mxSetField(g,0,"isInterior",logicalVectorToMx(heInterior));
      mxSetField(s,0,"halfedge",g); }

    plhs[0] = s;
}
```

(Remove the unused `group` lambda if the compiler warns — it is illustrative; the inline `f[]` arrays are what build each sub-struct.)

- [ ] **Step 2: Wire into dispatch**

In the dispatch chain (~line 1108), after `else if (cmd == "destroy") ...`, add:

```cpp
        else if (cmd == "topology")      cmdTopology(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 3: Write the MATLAB test**

Create `bindings/mex/test/test_topology.m` (copy `local_icosahedron()` verbatim from `test_vertex_frames.m`):

```matlab
function test_topology
fprintf('[test_topology] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1); nF = size(F,1); nE = nV + nF - 2;   % Euler: V - E + F = 2
h = nxr_compute('create', V, F);
T = nxr_compute('topology', h);

assert(T.schemaVersion == 1, 'schemaVersion');
assert(T.vertex.count == nV, 'vertex.count');
assert(T.face.count   == nF, 'face.count');
assert(T.edge.count   == nE, 'edge.count');
assert(T.corner.count == 3*nF, 'corner.count == 3F');
assert(T.halfedge.count == 2*nE, 'halfedge.count == 2E');

% 1-based indices: every halfedge field in range [1, count] (closed mesh: no 0 sentinels).
H = T.halfedge.count;
assert(all(T.halfedge.twin >= 1 & T.halfedge.twin <= H), 'twin in 1..H');
assert(all(T.halfedge.next >= 1 & T.halfedge.next <= H), 'next in 1..H');
assert(all(T.halfedge.vertex >= 1 & T.halfedge.vertex <= nV), 'vertex in 1..nV');
assert(all(T.halfedge.edge >= 1 & T.halfedge.edge <= nE), 'edge in 1..nE');
assert(all(T.halfedge.face >= 1 & T.halfedge.face <= nF), 'face in 1..nF (closed)');
assert(isa(T.halfedge.twin, 'uint32'), 'indices are uint32');
assert(islogical(T.halfedge.orientation), 'orientation logical');

% twin is an involution: twin(twin(he)) == he (1-based).
tw = double(T.halfedge.twin);
assert(isequal(tw(tw), (1:H)'), 'twin is an involution');
% next is a 3-cycle on a triangle mesh: next^3 == identity.
nx = double(T.halfedge.next);
assert(isequal(nx(nx(nx)), (1:H)'), 'next^3 == identity (triangles)');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_topology\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
V = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0; ...
      0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t; ...
      t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
V = V ./ sqrt(sum(V.^2, 2));
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
     2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
     5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
```

- [ ] **Step 4: Build and run the MATLAB test**

Run: `bash scripts/build.sh Release 2>&1 | tail -5`
Then via MATLAB MCP `run_matlab_file` on `bindings/mex/test/test_topology.m`.
Expected: `ALL TESTS PASSED: test_topology`. If `twin is an involution` fails, the 1-based conversion or `idxOrNone` is wrong; if `face in 1..nF` fails on this closed mesh, a boundary code path is mis-firing.

- [ ] **Step 5: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_topology.m
git commit -m "feat(mex): add 'topology' command (1-based halfedge struct-of-arrays)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `geometry` command

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`
- Create: `bindings/mex/test/test_geometry_bundle.m`

- [ ] **Step 1: Add `cmdGeometry`**

Add before the dispatch block. Reads geometry-central caches for the light per-element scalars and the Task 1/2 library helpers for `grid` and `curvature`. Halfedge complex quantities (`transportAlong`, etc.) are read element-wise into `Eigen::VectorXcd`.

```cpp
void cmdGeometry(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;
    if (nrhs < 2) throw std::invalid_argument("nxr_compute('geometry', handle)");
    ContextHolder& h = getHolder(prhs[1]);
    Manifold& m = *h.ctx;
    auto& mesh = m.mesh();
    auto& geom = m.geometry();

    const int nV = m.nV(), nE = m.nE(), nF = m.nF();
    const int nH = static_cast<int>(mesh.nHalfedges());
    const int nC = static_cast<int>(mesh.nCorners());

    // ── per-element scalar caches ──
    geom.requireVertexDualAreas();
    geom.requireVertexGaussianCurvatures();   // pins angle sums dependency; angleSums below
    geom.requireEdgeLengths();
    geom.requireEdgeCotanWeights();
    geom.requireEdgeDihedralAngles();
    geom.requireFaceAreas();
    geom.requireHalfedgeCotanWeights();
    geom.requireCornerAngles();
    geom.requireCornerScaledAngles();
    geom.requireVertexAngleSums();
    geom.requireTransportVectorsAlongHalfedge();
    geom.requireTransportVectorsAcrossHalfedge();
    geom.requireHalfedgeVectorsInVertex();
    geom.requireHalfedgeVectorsInFace();

    auto vVec = [&](auto accessor, int n) {
        Eigen::VectorXd x(n);
        for (int i = 0; i < n; ++i) x(i) = accessor(i);
        return x;
    };

    Eigen::VectorXd vDualAreas(nV), vAngleSums(nV);
    for (Vertex v : mesh.vertices()) {
        int i = v.getIndex();
        vDualAreas(i) = geom.vertexDualAreas[v];
        vAngleSums(i) = geom.vertexAngleSums[v];
    }
    Eigen::VectorXd eLen(nE), eCot(nE), eDih(nE);
    for (Edge e : mesh.edges()) {
        int i = e.getIndex();
        eLen(i) = geom.edgeLengths[e];
        eCot(i) = geom.edgeCotanWeights[e];
        eDih(i) = geom.edgeDihedralAngles[e];
    }
    Eigen::VectorXd fArea(nF);
    Eigen::MatrixXd fCentroid(nF, 3);
    for (Face f : mesh.faces()) {
        int i = f.getIndex();
        fArea(i) = geom.faceAreas[f];
        Vector3 c{0,0,0}; int k = 0;
        for (Vertex v : f.adjacentVertices()) { c += geom.vertexPositions[v]; ++k; }
        c /= static_cast<double>(k);
        fCentroid(i,0)=c.x; fCentroid(i,1)=c.y; fCentroid(i,2)=c.z;
    }
    Eigen::VectorXd hCot(nH);
    Eigen::VectorXcd hVinV(nH), hVinF(nH), hTAlong(nH), hTAcross(nH);
    for (Halfedge he : mesh.halfedges()) {
        int i = he.getIndex();
        hCot(i) = geom.halfedgeCotanWeights[he];
        Vector2 a = geom.halfedgeVectorsInVertex[he];   hVinV(i)  = {a.x, a.y};
        Vector2 b = geom.halfedgeVectorsInFace[he];      hVinF(i)  = {b.x, b.y};
        Vector2 t = geom.transportVectorsAlongHalfedge[he];  hTAlong(i)  = {t.x, t.y};
        Vector2 u = geom.transportVectorsAcrossHalfedge[he]; hTAcross(i) = {u.x, u.y};
    }
    Eigen::VectorXd cAng(nC), cScaled(nC);
    for (Corner c : mesh.corners()) {
        int i = c.getIndex();
        cAng(i)    = geom.cornerAngles[c];
        cScaled(i) = geom.cornerScaledAngles[c];
    }
    geom.requireFaceAreas();
    double totalArea = fArea.sum();

    // ── library helpers ──
    Eigen::MatrixXcd vGrid = nxr::manifold::geometry::vertexGrid(m);
    Eigen::MatrixXcd fGrid = nxr::manifold::geometry::faceGrid(m);
    auto cv = nxr::manifold::geometry::vertexCurvature(m);

    // ── assemble nested struct ──
    const char* topF[] = {"schemaVersion","totalArea","vertex","edge","face","halfedge","corner"};
    mxArray* s = mxCreateStructMatrix(1,1,7,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));
    mxSetField(s,0,"totalArea",scalarToMx(totalArea));

    { const char* f[] = {"dualAreas","angleSums","curvature","meanCurvature","grid"};
      mxArray* g = mxCreateStructMatrix(1,1,5,f);
      mxSetField(g,0,"dualAreas",eigenVectorToMx(vDualAreas));
      mxSetField(g,0,"angleSums",eigenVectorToMx(vAngleSums));
      mxSetField(g,0,"curvature",eigenComplexVectorToMx(cv.deviatoric));
      mxSetField(g,0,"meanCurvature",eigenVectorToMx(cv.mean));
      mxSetField(g,0,"grid",eigenComplexMatrixToMx(vGrid));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"lengths","cotanWeights","dihedralAngles"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"lengths",eigenVectorToMx(eLen));
      mxSetField(g,0,"cotanWeights",eigenVectorToMx(eCot));
      mxSetField(g,0,"dihedralAngles",eigenVectorToMx(eDih));
      mxSetField(s,0,"edge",g); }
    { const char* f[] = {"areas","centroids","grid"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"areas",eigenVectorToMx(fArea));
      mxSetField(g,0,"centroids",eigenMatrixToMx(fCentroid));
      mxSetField(g,0,"grid",eigenComplexMatrixToMx(fGrid));
      mxSetField(s,0,"face",g); }
    { const char* f[] = {"cotanWeights","vectorsInVertex","vectorsInFace","transportAlong","transportAcross"};
      mxArray* g = mxCreateStructMatrix(1,1,5,f);
      mxSetField(g,0,"cotanWeights",eigenVectorToMx(hCot));
      mxSetField(g,0,"vectorsInVertex",eigenComplexVectorToMx(hVinV));
      mxSetField(g,0,"vectorsInFace",eigenComplexVectorToMx(hVinF));
      mxSetField(g,0,"transportAlong",eigenComplexVectorToMx(hTAlong));
      mxSetField(g,0,"transportAcross",eigenComplexVectorToMx(hTAcross));
      mxSetField(s,0,"halfedge",g); }
    { const char* f[] = {"angles","scaledAngles"};
      mxArray* g = mxCreateStructMatrix(1,1,2,f);
      mxSetField(g,0,"angles",eigenVectorToMx(cAng));
      mxSetField(g,0,"scaledAngles",eigenVectorToMx(cScaled));
      mxSetField(s,0,"corner",g); }

    plhs[0] = s;
}
```

- [ ] **Step 2: Wire into dispatch**

After the `topology` line, add:

```cpp
        else if (cmd == "geometry")      cmdGeometry(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 3: Write the MATLAB test**

Create `bindings/mex/test/test_geometry_bundle.m` (copy the `local_icosahedron()` helper from Task 5):

```matlab
function test_geometry_bundle
fprintf('[test_geometry_bundle] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1); nF = size(F,1); nE = nV + nF - 2;
h = nxr_compute('create', V, F);
G = nxr_compute('geometry', h);

% shapes
assert(isequal(size(G.vertex.grid), [nV 3]), 'vertex.grid nV x 3');
assert(~isreal(G.vertex.grid), 'vertex.grid is complex');
assert(isequal(size(G.vertex.curvature), [nV 1]), 'vertex.curvature nV x 1');
assert(~isreal(G.vertex.curvature), 'curvature complex');
assert(isequal(size(G.vertex.meanCurvature), [nV 1]), 'meanCurvature nV x 1');
assert(numel(G.edge.lengths) == nE, 'edge.lengths nE');
assert(numel(G.face.areas) == nF, 'face.areas nF');
assert(isequal(size(G.face.grid), [nF 3]), 'face.grid nF x 3');

% grid is a valid complex frame: real ⟂ imag, both unit, cross = unit normal
e1 = real(G.vertex.grid); e2 = imag(G.vertex.grid);
assert(max(abs(sqrt(sum(e1.^2,2)) - 1)) < 1e-9, 'real(grid) unit');
assert(max(abs(sqrt(sum(e2.^2,2)) - 1)) < 1e-9, 'imag(grid) unit');
assert(max(abs(sum(e1.*e2,2))) < 1e-9, 'real ⟂ imag');
nrm = cross(e1, e2, 2);
assert(max(abs(sqrt(sum(nrm.^2,2)) - 1)) < 1e-9, 'cross is unit normal');

% totalArea == sum(face.areas); dual areas also sum to total area
assert(abs(G.totalArea - sum(G.face.areas)) < 1e-9, 'totalArea == Σ faceAreas');
assert(abs(sum(G.vertex.dualAreas) - G.totalArea) < 1e-9, 'Σ dualAreas == totalArea');

% intrinsic Gaussian via angle defect integrates to 2πχ = 4π on a sphere
Kint = 2*pi - G.vertex.angleSums;
assert(abs(sum(Kint) - 4*pi) < 1e-6, 'Σ(2π − angleSum) == 2πχ == 4π');

% near-umbilic sphere: deviatoric small, mean positive, extrinsic K > 0
assert(max(abs(G.vertex.curvature)) < 0.5 * min(G.vertex.meanCurvature), 'deviatoric small');
Kext = G.vertex.meanCurvature.^2 - abs(G.vertex.curvature).^2;
assert(all(Kext > 0), 'extrinsic Gaussian H² − |q|² > 0');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_geometry_bundle\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
V = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0; ...
      0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t; ...
      t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
V = V ./ sqrt(sum(V.^2, 2));
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
     2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
     5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
```

- [ ] **Step 4: Build and run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5`
Then MATLAB MCP `run_matlab_file` on `bindings/mex/test/test_geometry_bundle.m`.
Expected: `ALL TESTS PASSED: test_geometry_bundle`.

- [ ] **Step 5: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_geometry_bundle.m
git commit -m "feat(mex): add 'geometry' command (element-grouped, complex grid + 2-RoSy curvature)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `gauge` command (euclidean / levi-civita / trivial)

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`
- Create: `bindings/mex/test/test_gauge.m`

- [ ] **Step 1: Add `cmdGauge`**

Add before the dispatch block. `face.rotation` is emitted empty in v1 (deferred per spec §10/§11). `vertex.rotation` is the realized rotation relative to Levi-Civita: identity for `euclidean`/`levi-civita`, integrated φ for `trivial`. (For `euclidean` the realized *frame* differs — that is handled by the consumer via `type`, per spec §5.3 — but the rotation field describes the relationship to the LC grid and is identity.)

```cpp
void cmdGauge(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    namespace conn = nxr::manifold::connection;
    if (nrhs < 3) throw std::invalid_argument(
        "nxr_compute('gauge', handle, type[, opts]) — type ∈ {euclidean,levi-civita,trivial}");
    ContextHolder& h = getHolder(prhs[1]);
    Manifold& m = *h.ctx;
    std::string type = getStringArg(prhs[2]);

    const int nV = m.nV();
    Eigen::VectorXcd rot = Eigen::VectorXcd::Ones(nV);   // identity by default
    std::vector<long> singVerts;
    Eigen::VectorXd   singIdx;
    std::string singSource = "none";

    if (type == "euclidean" || type == "levi-civita") {
        // rotation stays identity; nothing else to compute.
    } else if (type == "trivial") {
        if (nrhs < 4 || !mxIsStruct(prhs[3]))
            throw std::invalid_argument(
                "gauge('trivial') requires opts struct with singVerts, singValues");
        const mxArray* fv = mxGetField(prhs[3], 0, "singVerts");
        const mxArray* fi = mxGetField(prhs[3], 0, "singValues");
        if (!fv || !fi) throw std::invalid_argument("opts needs singVerts and singValues");
        std::vector<int> verts = mxToVertexIndices(fv);     // 1-based → 0-based
        Eigen::VectorXd  vals  = mxToEigenVector(fi);
        if (static_cast<size_t>(vals.size()) != verts.size())
            throw std::invalid_argument("singVerts and singValues length mismatch");
        std::map<int,double> sing;
        for (size_t i = 0; i < verts.size(); ++i) sing[verts[i]] = vals[i];

        conn::GaugeRotations gr =
            conn::integrateTrivialGaugeRotations(m, ensureDec(h), *h.cache, sing);
        rot = gr.vertex;

        singVerts.assign(verts.begin(), verts.end());       // 0-based; convert at marshal
        singIdx = vals;
        if (const mxArray* fs = mxGetField(prhs[3], 0, "source"))
            singSource = getStringArg(fs);
        else
            singSource = "manual";
    } else {
        throw std::invalid_argument("unknown gauge type '" + type + "'");
    }

    // ── assemble struct ──
    const char* topF[] = {"schemaVersion","type","vertex","face","singularity"};
    mxArray* s = mxCreateStructMatrix(1,1,5,topF);
    mxSetField(s,0,"schemaVersion",scalarToMx(1));
    mxSetField(s,0,"type",mxCreateString(type.c_str()));

    { const char* f[] = {"rotation"}; mxArray* g = mxCreateStructMatrix(1,1,1,f);
      mxSetField(g,0,"rotation",eigenComplexVectorToMx(rot));
      mxSetField(s,0,"vertex",g); }
    { const char* f[] = {"rotation"}; mxArray* g = mxCreateStructMatrix(1,1,1,f);
      mxSetField(g,0,"rotation",eigenComplexVectorToMx(Eigen::VectorXcd(0)));  // empty v1
      mxSetField(s,0,"face",g); }
    { const char* f[] = {"vertices","indices","source"};
      mxArray* g = mxCreateStructMatrix(1,1,3,f);
      mxSetField(g,0,"vertices",indexVectorToMx1Based(singVerts));
      mxSetField(g,0,"indices",eigenVectorToMx(singIdx));
      mxSetField(g,0,"source",mxCreateString(singSource.c_str()));
      mxSetField(s,0,"singularity",g); }

    plhs[0] = s;
}
```

- [ ] **Step 2: Wire into dispatch**

After the `geometry` line, add:

```cpp
        else if (cmd == "gauge")         cmdGauge(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 3: Write the MATLAB test**

Create `bindings/mex/test/test_gauge.m` (copy `local_icosahedron()`):

```matlab
function test_gauge
fprintf('[test_gauge] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1);
h = nxr_compute('create', V, F);

% euclidean: identity rotation
Ge = nxr_compute('gauge', h, 'euclidean');
assert(strcmp(Ge.type,'euclidean'), 'type euclidean');
assert(numel(Ge.vertex.rotation) == nV, 'rotation nV');
assert(max(abs(Ge.vertex.rotation - 1)) < 1e-12, 'euclidean rotation == 1');

% levi-civita: identity rotation vs the grid
Gl = nxr_compute('gauge', h, 'levi-civita');
assert(max(abs(Gl.vertex.rotation - 1)) < 1e-12, 'levi-civita rotation == 1');

% trivial: two index-1 singularities (Σ = χ = 2)
opts = struct('singVerts', uint32([1;2]), 'singValues', [1;1], 'source', 'manual');
Gt = nxr_compute('gauge', h, 'trivial', opts);
assert(strcmp(Gt.type,'trivial'), 'type trivial');
assert(max(abs(abs(Gt.vertex.rotation) - 1)) < 1e-9, 'trivial rotation unit modulus');
assert(numel(Gt.singularity.vertices) == 2, 'two singularities');
assert(all(Gt.singularity.vertices == uint32([1;2])), 'singularity verts 1-based round-trip');
assert(abs(sum(Gt.singularity.indices) - 2) < 1e-12, 'Σ indices == χ == 2');
assert(max(abs(Gt.vertex.rotation - 1)) > 1e-6, 'trivial differs from identity');

% realized trivial frame is still an orthonormal complex frame
G = nxr_compute('geometry', h);
c = Gt.vertex.rotation .* G.vertex.grid;        % broadcast V×1 over V×3
e1 = real(c); e2 = imag(c);
assert(max(abs(sqrt(sum(e1.^2,2)) - 1)) < 1e-9, 'realized real(c) unit');
assert(max(abs(sum(e1.*e2,2))) < 1e-9, 'realized real ⟂ imag');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_gauge\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
V = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0; ...
      0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t; ...
      t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
V = V ./ sqrt(sum(V.^2, 2));
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
     2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
     5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
```

- [ ] **Step 4: Build and run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5`
Then MATLAB MCP `run_matlab_file` on `bindings/mex/test/test_gauge.m`.
Expected: `ALL TESTS PASSED: test_gauge`.

- [ ] **Step 5: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_gauge.m
git commit -m "feat(mex): add 'gauge' command (euclidean/levi-civita/trivial transform)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `bundle` convenience + end-to-end leadfield round-trip

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`
- Create: `bindings/mex/test/test_bundle.m`

- [ ] **Step 1: Add `cmdBundle`**

Add before the dispatch block. It calls the three builders and packs their results into one struct. Refactor the three `cmdXxx` bodies' struct construction into helpers `buildTopologyStruct(h)`, `buildGeometryStruct(h)`, `buildGaugeStruct(h, type, opts...)` that return `mxArray*`, then have both the individual commands and `cmdBundle` call them. (Mechanical: move each `cmd`'s body — minus the `getHolder`/arg parsing — into an `mxArray* buildXxx(...)` and have the `cmd` wrapper parse args then `plhs[0] = buildXxx(...)`.)

```cpp
void cmdBundle(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw std::invalid_argument(
        "nxr_compute('bundle', handle, gaugeType[, opts])");
    ContextHolder& h = getHolder(prhs[1]);
    std::string type = getStringArg(prhs[2]);

    const char* f[] = {"Topology","Geometry","Gauge"};
    mxArray* s = mxCreateStructMatrix(1,1,3,f);
    mxSetField(s,0,"Topology", buildTopologyStruct(h));
    mxSetField(s,0,"Geometry", buildGeometryStruct(h));
    // Reuse the gauge opts (prhs[3]) if present.
    mxSetField(s,0,"Gauge", buildGaugeStruct(h, type, nrhs >= 4 ? prhs[3] : nullptr));
    plhs[0] = s;
}
```

(Adjust `buildGaugeStruct`'s signature to accept the optional opts `mxArray*`; `cmdGauge` passes `nrhs >= 4 ? prhs[3] : nullptr` likewise.)

- [ ] **Step 2: Wire into dispatch**

After the `gauge` line, add:

```cpp
        else if (cmd == "bundle")        cmdBundle(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 3: Write the end-to-end MATLAB test**

Create `bindings/mex/test/test_bundle.m`. This is the deliverable's acceptance test: the leadfield round-trip is exact and the gauge realization is consistent.

```matlab
function test_bundle
fprintf('[test_bundle] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nV = size(V,1);
h = nxr_compute('create', V, F);

opts = struct('singVerts', uint32([1;2]), 'singValues', [1;1], 'source', 'manual');
B = nxr_compute('bundle', h, 'levi-civita');
assert(isfield(B,'Topology') && isfield(B,'Geometry') && isfield(B,'Gauge'), 'three sub-structs');
assert(strcmp(B.Gauge.type,'levi-civita'), 'bundle gauge type');

% ── leadfield round-trip (the deliverable) ──
% Synthesize an unconstrained Cartesian leadfield block per vertex (nSensors×3).
rng(0); nSensors = 7;
c = B.Gauge.vertex.rotation .* B.Geometry.vertex.grid;   % realized frame (LC: rotation==1)
n = cross(real(c), imag(c), 2);
maxErr = 0;
for v = 1:nV
    Gv = randn(nSensors, 3);                  % Cartesian gain block
    cv = c(v,:); nv = n(v,:);
    % intrinsic readout
    Ltan = Gv * cv.';                         % nSensors×1 complex
    Ln   = Gv * nv.';                         % nSensors×1 real
    % a dipole and its intrinsic coords
    J  = randn(1,3);
    z  = sum(cv .* J, 2);  jn = sum(nv .* J, 2);
    % sensor reading two ways must agree
    direct   = Gv * J.';
    intrinsic= real(conj(z) .* Ltan) + jn .* Ln;
    maxErr = max(maxErr, max(abs(direct - intrinsic)));
end
assert(maxErr < 1e-9, sprintf('leadfield round-trip exact (err=%.2e)', maxErr));

% ── frame inverse is exact (lossless rotation) ──
J  = randn(nV,3);
z  = sum(c .* J, 2);  jn = sum(n .* J, 2);
Jr = real(conj(z) .* c) + jn .* n;
assert(max(abs(Jr(:) - J(:))) < 1e-9, 'Cartesian → intrinsic → Cartesian is identity');

% ── trivial bundle: Gauss-Bonnet input valid, realized frame orthonormal ──
Bt = nxr_compute('bundle', h, 'trivial', opts);
assert(abs(sum(Bt.Gauge.singularity.indices) - 2) < 1e-12, 'Σ indices == χ == 2');
ct = Bt.Gauge.vertex.rotation .* Bt.Geometry.vertex.grid;
assert(max(abs(sqrt(sum(real(ct).^2,2)) - 1)) < 1e-9, 'trivial realized frame unit');

nxr_compute('destroy', h);
fprintf('PASSED leadfield round-trip, err=%.2e\n', maxErr);
fprintf('ALL TESTS PASSED: test_bundle\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
V = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0; ...
      0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t; ...
      t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
V = V ./ sqrt(sum(V.^2, 2));
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
     2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
     5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
```

- [ ] **Step 4: Build and run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5`
Then MATLAB MCP `run_matlab_file` on `bindings/mex/test/test_bundle.m`.
Expected: `ALL TESTS PASSED: test_bundle` and a printed `leadfield round-trip, err < 1e-9`.

- [ ] **Step 5: Run the full native + MATLAB suites for regression**

Run: `./build/Release/test_geometry_bundle && ./build/Release/test_connection_laplacian`
Then MATLAB MCP `run_matlab_file` on each of `test_topology.m`, `test_geometry_bundle.m`, `test_gauge.m`, `test_bundle.m`, and the existing `test_vertex_frames.m` (must still pass).
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_bundle.m
git commit -m "feat(mex): add 'bundle' command + leadfield round-trip acceptance test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 hybrid delivery, `bundle` command | Task 8 |
| §3 index-base contract (1-based, single boundary, sentinel 0) | Task 4 (`indexVectorToMx1Based`), Task 5 (use + test) |
| §4.1 Topology struct | Task 5 |
| §4.2 Geometry struct (element-grouped, light) | Task 6 |
| §4.3 Gauge struct (transform; trivial carries data) | Task 7 |
| §5.1 complex `grid`, normal derived, lossless transform | Task 1 (assembly), Task 8 (round-trip) |
| §5.2 2-RoSy curvature + mean | Task 2 |
| §5.3 gauge realization (euclidean/LC/trivial) | Task 7, Task 8 (realize + assert) |
| §6 trivial-gauge vertex integration | Task 3 |
| §7 leadfield correspondence `Ltan = G·cᵀ` | Task 8 |
| §8 MEX command surface | Tasks 5–8 |
| §10 deferred `face.rotation` (emitted empty) | Task 7 |

**Placeholder scan:** None. (Task 8 Step 1 asks for a mechanical refactor — extracting each `cmd` body into a `buildXxx` helper — with explicit instructions, not an unfilled placeholder.)

**Type consistency:**
- `vertexGrid`/`faceGrid` return `Eigen::MatrixXcd` (Task 1) — consumed by `eigenComplexMatrixToMx` (Task 4, Task 6). ✓
- `VertexCurvature2RoSy{deviatoric (VectorXcd), mean (VectorXd)}` (Task 2) — marshaled as `vertex.curvature` (complex) + `vertex.meanCurvature` (real) (Task 6). ✓
- `GaugeRotations{vertex (VectorXcd)}` (Task 3) — used in `cmdGauge` (Task 7). ✓
- `integrateTrivialGaugeRotations(m, dec, cache, sing)` signature identical in Task 3 declaration, Task 3 definition, Task 7 call. ✓
- `indexVectorToMx1Based(std::vector<long>)` (Task 4) — fed `std::vector<long>` everywhere in Task 5/Task 7. ✓
- Build helpers `buildTopologyStruct`/`buildGeometryStruct`/`buildGaugeStruct` introduced in Task 8 Step 1 are refactors of Task 5/6/7 bodies — the refactor note instructs extracting them; ensure the individual `cmd` wrappers are updated to call them so there is one implementation each. ✓

**Open risk flagged for the implementer:** the geometry-central transport sign/orientation in Task 3 (`transportVectorsAlongHalfedge` direction and the `sign` on φ) is validated by the Task 3 native test (`unit modulus`, `differs from LC`) and the Task 8 trivial-frame orthonormality assert; the φ correctness itself is already covered by the existing `test_connection_laplacian`. If a future face-based consumer needs `Gauge.face.rotation`, add a dual-tree integration (the existing `propagateAngles` in `direction_field.cpp` is the face analogue).
