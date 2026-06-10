# Cortical Covariant Differential Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the cortical-frame leadfield an artifact-free differential calculus — a frame transport `P_ij = Fⱼᵀ Fᵢ`, local↔world lifts, and a covariant gradient operator `G` — so transport and gradients of a 3-vector field reflect true variation, not the rotating/tilting cortical frame.

**Architecture:** Additive C++ free functions in a new `nxr::manifold::differential` namespace (`src/covariant_differential.cpp`), all sourcing the per-vertex frame `Fᵥ = [e1|e2|n]` from the existing `geometry::vertexFrames`. The covariant gradient `G` is a sparse `3E×3N` block operator (the discrete `d0` lifted from ±1 to the `3×3` frame transport); its Laplacian `GᵀWG` equals the existing Ambient covariant (a built-in consistency check). Thin MEX commands expose all of it. No new geometry math.

**Tech Stack:** C++17, Eigen, geometry-central, MATLAB MEX. Build `bash scripts/build.sh Release`. Native tests in `build/`. MATLAB tests via the MATLAB MCP `run_matlab_file`.

**Spec:** `docs/superpowers/specs/2026-06-10-cortical-covariant-differential-operators-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/covariant_differential.cpp` | `frameTransport`, `vertexFrameMatrices`, `liftToWorld`, `liftToFrame`, `covariantGradient` |
| Modify `include/nxr/compute.h` | declare the `nxr::manifold::differential` namespace + its 5 functions |
| Create `test/test_covariant_differential.cpp` | native tests (orthogonality, flatness, lift round-trip, artifact removal, `GᵀWG`==ambient) |
| Modify `bindings/mex/src/nxr_compute_mex.cpp` | `frameTransport` / `liftToWorld` / `liftToFrame` commands + `operators … gradient3D` |
| Create `bindings/mex/test/test_covariant_differential.m` | MATLAB end-to-end (artifact removal, transport, lifts) |
| Modify `CMakeLists.txt` | register `src/covariant_differential.cpp` + `test_covariant_differential` |

**Layout convention (fixed across all tasks):** a frame-local field is `[nV×3]` real, each row `[a,b,c] = (tangent₁, tangent₂, normal)`. The operator `G` uses **component-major** `3N` input `[a;b;c]` and `3E` output, so a column-major `[nV×3]` matrix `L` flattens to the component-major vector via `Eigen::Map<VectorXd>(L.data(), 3N)` (column-major `[N×3]` == `[a;b;c]`). This makes `GᵀWG` align with the existing covariant Laplacian (also component-major `[a;b;c]`).

**Frame convention (fixed):** `Fᵥ = [e1ᵥ | e2ᵥ | nᵥ]` as **columns**, with `e1ᵥ = vf.e1.row(v)`, `e2ᵥ = vf.e2.row(v)`, `nᵥ = vf.normals.row(v)` from `geometry::vertexFrames(m)`. These are orthonormal and right-handed (`n = e1×e2`), so `Fᵥ` is a proper rotation. Transport `i→j` is `P_ij = Fⱼᵀ Fᵢ`.

---

## Task 1: `frameTransport` + `vertexFrameMatrices`

**Files:** Create `src/covariant_differential.cpp`; Modify `include/nxr/compute.h`; Create `test/test_covariant_differential.cpp`; Modify `CMakeLists.txt`.

- [ ] **Step 1: Declare the namespace + functions in `include/nxr/compute.h`.** Add immediately after the `} // namespace nxr::manifold::geometry` line (around line 1464):

```cpp
// ── Covariant differential operators on 3-vector cortical fields ──────────────
//
// Artifact-free transport + differentiation of a 3-vector field (e.g. a leadfield)
// expressed in per-vertex cortical frames Fv = [e1 | e2 | n] (columns), sourced from
// geometry::vertexFrames. The connection is the flat full-frame transport
// P_ij = Fj^T Fi — it accounts for the tangent rotation AND the normal tilt, and is
// path-independent (zero holonomy), which is exactly what removes the frame-rotation
// artifact (e.g. Cartesian-parallel leadfields on opposite sulcal walls reading as
// antiparallel in local frames). See the design doc.
namespace nxr::manifold::differential {

// 3x3 transport of a frame-local vector from vertex i's frame to vertex j's frame,
// P_ij = Fj^T Fi. Flat ⇒ exact for ANY pair (i, j) (adjacent or not). 0-based indices.
Eigen::Matrix3d frameTransport(Manifold& m, int i, int j);

// Per-vertex frames Fv stacked as [nV, 9], row v = Fv flattened ROW-major
// (Fv(0,0),Fv(0,1),Fv(0,2), Fv(1,0),...). Columns of Fv are [e1 | e2 | n].
Eigen::MatrixXd vertexFrameMatrices(Manifold& m);

// Lift a frame-local field to world (Cartesian): world[v] = Fv * Lloc[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToWorld(Manifold& m, const Eigen::MatrixXd& Lloc);

// Express a world (Cartesian) field in frames: Lloc[v] = Fv^T * Lworld[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToFrame(Manifold& m, const Eigen::MatrixXd& Lworld);

// Covariant gradient operator G : 3N -> 3E (component-major [a;b;c] blocks). For each
// oriented edge e: i->j, the covariant difference is δ_e = L_j - P_ij L_i. G has a +I3
// block at vertex j and a -P_ij block at vertex i. A Cartesian-constant field has G·L = 0.
Eigen::SparseMatrix<double> covariantGradient(Manifold& m);

} // namespace nxr::manifold::differential
```

- [ ] **Step 2: Write the failing test** `test/test_covariant_differential.cpp`:

```cpp
#include "nxr/compute.h"
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testFrameTransport() {
    std::cout << "\n=== covariant-differential: frame transport ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    // orthogonality: P_ij^T P_ij = I (P is a product of rotations)
    Eigen::Matrix3d P = differential::frameTransport(m, 0, 3);
    EXPECT((P.transpose() * P - Eigen::Matrix3d::Identity()).norm() < 1e-12, "P_ij orthogonal");

    // flatness / path-independence: around face {0,11,5}, P_k i · P_j k · P_i j = I
    int i = 0, j = 11, k = 5;
    Eigen::Matrix3d loop = differential::frameTransport(m, k, i)
                         * differential::frameTransport(m, j, k)
                         * differential::frameTransport(m, i, j);
    EXPECT((loop - Eigen::Matrix3d::Identity()).norm() < 1e-12, "holonomy around triangle is identity (flat)");

    // self-transport is identity
    EXPECT((differential::frameTransport(m, 4, 4) - Eigen::Matrix3d::Identity()).norm() < 1e-12,
           "P_ii = I");

    // vertexFrameMatrices: [nV,9], each row an orthonormal frame
    Eigen::MatrixXd Fm = differential::vertexFrameMatrices(m);
    EXPECT(Fm.rows() == 12 && Fm.cols() == 9, "vertexFrameMatrices [12,9]");
    Eigen::Matrix3d F0;  // row 0 reshaped row-major
    F0 << Fm(0,0),Fm(0,1),Fm(0,2), Fm(0,3),Fm(0,4),Fm(0,5), Fm(0,6),Fm(0,7),Fm(0,8);
    EXPECT((F0.transpose()*F0 - Eigen::Matrix3d::Identity()).norm() < 1e-12, "Fv orthonormal");
}

int main() {
    testFrameTransport();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
```

- [ ] **Step 3: Create `src/covariant_differential.cpp`** with the frame helpers (the lift + gradient bodies are added in Tasks 2–3; include them as declared-only is not needed since they're free functions — just implement Task-1 functions now):

```cpp
#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace nxr::manifold::differential {

// Build Fv = [e1 | e2 | n] (columns) for vertex v from precomputed frames.
static Eigen::Matrix3d frameOf(const geometry::VertexFrames& vf, int v) {
    Eigen::Matrix3d Fv;
    Fv.col(0) = vf.e1.row(v).transpose();
    Fv.col(1) = vf.e2.row(v).transpose();
    Fv.col(2) = vf.normals.row(v).transpose();
    return Fv;
}

Eigen::Matrix3d frameTransport(Manifold& m, int i, int j) {
    const int nV = m.nV();
    if (i < 0 || i >= nV || j < 0 || j >= nV)
        throw Error(ErrorCode::InvalidInput,
            "frameTransport: vertex index out of range",
            "Expected 0 <= i,j < " + std::to_string(nV) + ".");
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    return frameOf(vf, j).transpose() * frameOf(vf, i);   // P_ij = Fj^T Fi
}

Eigen::MatrixXd vertexFrameMatrices(Manifold& m) {
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    const int nV = m.nV();
    Eigen::MatrixXd out(nV, 9);
    for (int v = 0; v < nV; ++v) {
        Eigen::Matrix3d Fv = frameOf(vf, v);
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q)
                out(v, 3*p + q) = Fv(p, q);   // row-major flatten
    }
    return out;
}

} // namespace nxr::manifold::differential
```

- [ ] **Step 4: Register sources + test in `CMakeLists.txt`.** Append `src/covariant_differential.cpp` to the `add_library(nxr_compute ...)` source list. Add after the `test_facets` (or last) test registration (mirror the existing pattern, e.g. near line 190):

```cmake
    add_executable(test_covariant_differential test/test_covariant_differential.cpp)
    target_link_libraries(test_covariant_differential PRIVATE nxr_compute)
    add_test(NAME test_covariant_differential COMMAND test_covariant_differential)
```

- [ ] **Step 5: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -6 && ./build/test_covariant_differential`
Expected: `ALL PASSED` (frame-transport block). clangd "file not found" diagnostics on Eigen/nxr headers are IDE noise — trust `scripts/build.sh`.

- [ ] **Step 6: Commit**

```bash
git add include/nxr/compute.h src/covariant_differential.cpp test/test_covariant_differential.cpp CMakeLists.txt
git commit -m "feat(differential): frameTransport + vertexFrameMatrices (flat full-frame connection)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Before you begin (Task 1)
Read `include/nxr/compute.h` lines ~1346–1361 (`VertexFrames`, `vertexFrames`) — `vf.e1/e2/normals` are each `[nV×3]` real, with `normals == e1×e2` (right-handed), so `Fv` is a proper rotation and no cross-product is needed. Confirm `src/covariant_differential.cpp` is added to the `nxr_compute` library sources, not just the test exe. `Error`/`ErrorCode::InvalidInput` are the project error types (see `src/connection_laplacian.cpp`).

---

## Task 2: `liftToWorld` / `liftToFrame`

**Files:** Modify `src/covariant_differential.cpp`, `test/test_covariant_differential.cpp`.

- [ ] **Step 1: Write the failing test** — add to `test/test_covariant_differential.cpp` and call from `main()`:

```cpp
static void testLifts() {
    std::cout << "\n=== covariant-differential: lifts ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    // round-trip: liftToFrame(liftToWorld(L)) == L
    Eigen::MatrixXd Lloc = Eigen::MatrixXd::Random(12, 3);
    Eigen::MatrixXd back = differential::liftToFrame(m, differential::liftToWorld(m, Lloc));
    EXPECT((back - Lloc).cwiseAbs().maxCoeff() < 1e-12, "liftToFrame∘liftToWorld == identity");

    // artifact removal end-to-end: a Cartesian-CONSTANT field lifted to frames has
    // DIFFERENT local coords at different vertices, but the SAME world vector everywhere.
    Eigen::MatrixXd Lworld(12, 3);
    for (int v = 0; v < 12; ++v) Lworld.row(v) = Eigen::RowVector3d(1.0, 0.0, 0.0);
    Eigen::MatrixXd locC = differential::liftToFrame(m, Lworld);
    Eigen::MatrixXd worldBack = differential::liftToWorld(m, locC);
    EXPECT((worldBack - Lworld).cwiseAbs().maxCoeff() < 1e-12, "lift recovers constant world field");
    // local coords genuinely differ between two vertices on the curved surface
    EXPECT((locC.row(0) - locC.row(6)).cwiseAbs().maxCoeff() > 1e-3,
           "same Cartesian vector has different local coords (the artifact lifts away)");
}
```

- [ ] **Step 2: Implement the lifts** in `src/covariant_differential.cpp` (inside the namespace, after `vertexFrameMatrices`):

```cpp
Eigen::MatrixXd liftToWorld(Manifold& m, const Eigen::MatrixXd& Lloc) {
    const int nV = m.nV();
    if (Lloc.rows() != nV || Lloc.cols() != 3)
        throw Error(ErrorCode::InvalidInput,
            "liftToWorld: field must be [nV, 3]",
            "Expected [" + std::to_string(nV) + ", 3].");
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    Eigen::MatrixXd out(nV, 3);
    for (int v = 0; v < nV; ++v)
        out.row(v) = (frameOf(vf, v) * Lloc.row(v).transpose()).transpose();   // Fv * local
    return out;
}

Eigen::MatrixXd liftToFrame(Manifold& m, const Eigen::MatrixXd& Lworld) {
    const int nV = m.nV();
    if (Lworld.rows() != nV || Lworld.cols() != 3)
        throw Error(ErrorCode::InvalidInput,
            "liftToFrame: field must be [nV, 3]",
            "Expected [" + std::to_string(nV) + ", 3].");
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    Eigen::MatrixXd out(nV, 3);
    for (int v = 0; v < nV; ++v)
        out.row(v) = (frameOf(vf, v).transpose() * Lworld.row(v).transpose()).transpose();  // Fv^T * world
    return out;
}
```

- [ ] **Step 3: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -4 && ./build/test_covariant_differential`
Expected: `ALL PASSED`.

- [ ] **Step 4: Commit**

```bash
git add src/covariant_differential.cpp test/test_covariant_differential.cpp
git commit -m "feat(differential): liftToWorld / liftToFrame (local↔Cartesian frame lift)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `covariantGradient` operator `G`

**Files:** Modify `src/covariant_differential.cpp`, `test/test_covariant_differential.cpp`.

- [ ] **Step 1: Write the failing test** — add to `test/test_covariant_differential.cpp`, call from `main()`. It includes the headline artifact-removal test and the `GᵀWG`==Ambient consistency check:

```cpp
#include "nxr/facets.h"                 // for operators().laplacian().covariant(...)
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

static void testCovariantGradient() {
    std::cout << "\n=== covariant-differential: covariant gradient G ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV(), E = m.nE();   // 12, 30

    Eigen::SparseMatrix<double> G = differential::covariantGradient(m);
    EXPECT(G.rows() == 3*E && G.cols() == 3*N, "G is [3E, 3N] = [90, 36]");

    // HEADLINE: a Cartesian-constant field has zero covariant gradient.
    Eigen::MatrixXd Lworld(N, 3);
    for (int v = 0; v < N; ++v) Lworld.row(v) = Eigen::RowVector3d(0.3, -0.7, 0.2);
    Eigen::MatrixXd Lloc = differential::liftToFrame(m, Lworld);     // [N,3]
    // component-major 3N: column-major [N,3] flattens to [a;b;c]
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(Lloc.data(), 3*N);
    EXPECT((G * x).cwiseAbs().maxCoeff() < 1e-10, "G·(Cartesian-constant) = 0 (artifact removed)");
    // ...while the naive component-wise difference is NOT zero on this curved mesh:
    EXPECT((Lloc.row(0) - Lloc.row(2)).cwiseAbs().maxCoeff() > 1e-3, "naive local difference is nonzero");

    // CONSISTENCY: G^T W G == the existing Ambient covariant Laplacian (default LC gauge).
    // W is the edge cotan weight, replicated across the 3 component blocks.
    auto& geom = m.operatorGeometry(); geom.requireEdgeCotanWeights();
    Eigen::VectorXd wEdge(E);
    for (auto e : m.mesh().edges()) wEdge(e.getIndex()) = geom.edgeCotanWeights[e];
    Eigen::VectorXd wDiag(3*E);
    for (int p = 0; p < 3; ++p) wDiag.segment(p*E, E) = wEdge;
    Eigen::SparseMatrix<double> W(3*E, 3*E);
    { std::vector<Eigen::Triplet<double>> tw; tw.reserve(3*E);
      for (int k = 0; k < 3*E; ++k) tw.emplace_back(k, k, wDiag(k));
      W.setFromTriplets(tw.begin(), tw.end()); }
    Eigen::SparseMatrix<double> GtWG = G.transpose() * W * G;

    namespace cl = ops::laplacian::connection;
    Eigen::SparseMatrix<double> Camb =
        m.operators().laplacian().covariant(cl::CovariantCoupling::Ambient);
    EXPECT((Eigen::MatrixXd(GtWG) - Eigen::MatrixXd(Camb)).cwiseAbs().maxCoeff() < 1e-9,
           "G^T W G == Ambient covariant Laplacian (consistency)");
}
```

(Note: the test now needs `#include "nxr/facets.h"` at the top of the file — move the include up with the others. `m.operatorGeometry()`/`m.mesh().edges()` require the GC headers, already included for this test.)

- [ ] **Step 2: Implement `covariantGradient`** in `src/covariant_differential.cpp` (inside the namespace; add `#include <vector>` and `#include <Eigen/SparseCore>` at the top if not already present via compute.h):

```cpp
Eigen::SparseMatrix<double> covariantGradient(Manifold& m) {
    using namespace geometrycentral::surface;
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    auto& mesh = m.mesh();
    const int N = m.nV();
    const int E = m.nE();

    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(E) * 12);   // 3 (+I) + 9 (-P) per edge

    for (Edge e : mesh.edges()) {
        Halfedge he = e.halfedge();                       // canonical orientation
        const int i = static_cast<int>(he.tailVertex().getIndex());
        const int j = static_cast<int>(he.tipVertex().getIndex());
        const int eIdx = static_cast<int>(e.getIndex());

        // P_ij = Fj^T Fi  (transport i -> j)
        Eigen::Matrix3d Pij = frameOf(vf, j).transpose() * frameOf(vf, i);

        // δ_e[p] = L_j[p] - Σ_q P_ij[p,q] L_i[q],  component-major rows {E*p + eIdx}.
        for (int p = 0; p < 3; ++p) {
            T.emplace_back(E*p + eIdx, N*p + j, 1.0);                 // +I at vertex j
            for (int q = 0; q < 3; ++q)
                T.emplace_back(E*p + eIdx, N*q + i, -Pij(p, q));      // -P_ij at vertex i
        }
    }

    Eigen::SparseMatrix<double> G(3*E, 3*N);
    G.setFromTriplets(T.begin(), T.end());
    G.makeCompressed();
    return G;
}
```

- [ ] **Step 3: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -6 && ./build/test_covariant_differential`
Expected: `ALL PASSED`, including `G·(Cartesian-constant) = 0` and `G^T W G == Ambient covariant Laplacian`.
- If the consistency check fails by a global sign or constant factor, the assembly's edge orientation or weight convention is off — reconcile against `src/covariant_laplacian.cpp`'s Ambient branch (`L3[i,j] = cotanL[i,j]·(Fiᵀ Fj)`, with `cotanL[i,j] = −edgeCotanWeights[e]` off-diagonal). Do not weaken the tolerance.

- [ ] **Step 4: Regression** — `./build/test_facets && ./build/test_operators_facet && ./build/test_geometry_bundle` must still pass (this task is purely additive).

- [ ] **Step 5: Commit**

```bash
git add src/covariant_differential.cpp test/test_covariant_differential.cpp
git commit -m "feat(differential): covariantGradient G (artifact-free 3-vector gradient; GᵀWG == Ambient covariant)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Before you begin (Task 3)
Read `src/covariant_laplacian.cpp` (the `Ambient` branch) so the `GᵀWG` consistency convention lines up: it assembles `L3[i,j] = cotanL[i,j]·(Fiᵀ Fj)` in component-major `[a;b;c]` layout. Your `G` must use the same component-major layout (output rows `E*p + e`, input cols `N*q + v`) for the consistency test to hold. `Edge`/`Halfedge`/`tailVertex`/`tipVertex`/`edges()` come from `geometrycentral/surface/manifold_surface_mesh.h`.

---

## Task 4: MEX commands + MATLAB test

**Files:** Modify `bindings/mex/src/nxr_compute_mex.cpp`; Create `bindings/mex/test/test_covariant_differential.m`.

- [ ] **Step 1: Add the MEX handlers** in `bindings/mex/src/nxr_compute_mex.cpp` (near the other `cmd*` handlers; mirror the `cmdOperators` style — `getHolder`, `getStringArg`, `eigenSparseToMx`). Add a small `[nV×3]` matrix marshaller (MATLAB and Eigen are both column-major, so `Eigen::Map` is a direct view):

```cpp
// Read a real [rows x cols] MATLAB matrix into an Eigen::MatrixXd (column-major copy).
static Eigen::MatrixXd mxToEigenMatrix(const mxArray* a) {
    if (!a || !mxIsDouble(a) || mxIsComplex(a))
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "expected a real double matrix");
    const mwSize r = mxGetM(a), c = mxGetN(a);
    return Eigen::Map<const Eigen::MatrixXd>(mxGetPr(a), r, c);
}
static mxArray* eigenMatrixToMx(const Eigen::MatrixXd& M) {
    mxArray* out = mxCreateDoubleMatrix(M.rows(), M.cols(), mxREAL);
    Eigen::Map<Eigen::MatrixXd>(mxGetPr(out), M.rows(), M.cols()) = M;
    return out;
}

// nxr_compute('frameTransport', h, i, j) -> 3x3   (i,j are 1-based)
void cmdFrameTransport(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 4) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "frameTransport: expected nxr_compute('frameTransport', handle, i, j).");
    ContextHolder& h = getHolder(prhs[1]);
    int i = static_cast<int>(mxGetScalar(prhs[2])) - 1;   // 1-based -> 0-based
    int j = static_cast<int>(mxGetScalar(prhs[3])) - 1;
    plhs[0] = eigenMatrixToMx(nxr::manifold::differential::frameTransport(*h.ctx, i, j));
}

// nxr_compute('liftToWorld', h, Lloc[nV x 3]) -> [nV x 3]
void cmdLiftToWorld(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "liftToWorld: expected nxr_compute('liftToWorld', handle, field[nV x 3]).");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = eigenMatrixToMx(nxr::manifold::differential::liftToWorld(*h.ctx, mxToEigenMatrix(prhs[2])));
}

// nxr_compute('liftToFrame', h, Lworld[nV x 3]) -> [nV x 3]
void cmdLiftToFrame(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "liftToFrame: expected nxr_compute('liftToFrame', handle, field[nV x 3]).");
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = eigenMatrixToMx(nxr::manifold::differential::liftToFrame(*h.ctx, mxToEigenMatrix(prhs[2])));
}
```

Confirm `h.ctx` is the `Manifold` (it is — the holder field used by `cmdOperators`). Add `#include "nxr/compute.h"` if not already present (it is, via the existing includes).

- [ ] **Step 2: Extend `cmdOperators`** to serve the gradient. In the `family == "laplacian"` ladder's enclosing dispatch (the `cmdOperators` function added previously), add a new family branch:

```cpp
    } else if (family == "gradient3D") {
        plhs[0] = eigenSparseToMx(nxr::manifold::differential::covariantGradient(m));
```

(Place it alongside the `laplacian`/`mass`/`hodge`/`dec` branches; `gradient3D` takes no subtype. Update the family error message to include `gradient3D`.)

- [ ] **Step 3: Register the three new commands** in `mexFunction`'s dispatch ladder (near the other `else if (cmd == ...)` lines) and the help string:

```cpp
        else if (cmd == "frameTransport")              cmdFrameTransport(nlhs, plhs, nrhs, prhs);
        else if (cmd == "liftToWorld")                 cmdLiftToWorld(nlhs, plhs, nrhs, prhs);
        else if (cmd == "liftToFrame")                 cmdLiftToFrame(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 4: Write the MATLAB test** `bindings/mex/test/test_covariant_differential.m`:

```matlab
function test_covariant_differential
fprintf('[test_covariant_differential] starting\n');
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

t = (1+sqrt(5))/2;
V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; 2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; 5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
h = nxr_compute('create', V, F);
N = 12; E = 30;

% frame transport is orthogonal
P = nxr_compute('frameTransport', h, 1, 4);
assert(isequal(size(P),[3 3]) && norm(P'*P - eye(3),'fro') < 1e-12, 'frameTransport orthogonal');

% lifts round-trip
Lloc = randn(N,3);
back = nxr_compute('liftToFrame', h, nxr_compute('liftToWorld', h, Lloc));
assert(max(abs(back(:) - Lloc(:))) < 1e-12, 'lift round-trip');

% ARTIFACT REMOVAL end-to-end: Cartesian-constant field -> zero covariant gradient
Lworld = repmat([0.3 -0.7 0.2], N, 1);
Lloc_c = nxr_compute('liftToFrame', h, Lworld);          % [N,3] local coords (differ per vertex)
G = nxr_compute('operators', h, 'gradient3D');           % sparse [3E, 3N]
x = Lloc_c(:);                                           % column-major [N,3] -> [a;b;c] = component-major 3N
assert(max(abs(G*x)) < 1e-9, 'G·(Cartesian-constant) = 0 (artifact removed)');
% naive local difference is NOT zero (frames rotate)
assert(max(abs(Lloc_c(1,:) - Lloc_c(3,:))) > 1e-3, 'naive local difference nonzero');

% two Cartesian-parallel leadfields lift to the SAME world vector (sulcal-wall artifact gone)
assert(max(abs(nxr_compute('liftToWorld', h, Lloc_c) - Lworld), [], 'all') < 1e-12, ...
       'lift recovers the constant world field');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_covariant_differential\n');
end
```

- [ ] **Step 5: Build + run via MATLAB MCP**

Run: `bash scripts/build.sh Release 2>&1 | tail -4`.
Then run `bindings/mex/test/test_covariant_differential.m` via the MATLAB MCP `run_matlab_file`.
Expected: `ALL TESTS PASSED: test_covariant_differential`.
Also re-run `bindings/mex/test/test_operators_command.m` (the `operators` command gained a family — confirm the existing families still work) → `ALL TESTS PASSED`.
If the MATLAB MCP is unavailable, report DONE_WITH_CONCERNS (native already covers the math).

- [ ] **Step 6: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_covariant_differential.m
git commit -m "feat(mex): frameTransport / liftToWorld / liftToFrame + operators gradient3D

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Before you begin (Task 4)
Read `bindings/mex/src/nxr_compute_mex.cpp`: `getHolder` (holder field `ctx` is the `Manifold`), `cmdOperators` (the family dispatch you extend), `eigenSparseToMx`, and the `mexFunction` dispatch ladder + unknown-command help string. The `x = Lloc_c(:)` flatten in MATLAB is column-major, which equals the component-major `[a;b;c]` layout the C++ `G` expects — that's why no reshape is needed. clangd diagnostics are IDE noise.

---

## Self-Review

**Spec coverage** (`2026-06-10-cortical-covariant-differential-operators-design.md`):

| Spec section | Task |
|---|---|
| §2 `P_ij = Fⱼᵀ Fᵢ`, flat / path-independent | Task 1 (`frameTransport`, flatness test) |
| §3a `frameTransport(i,j)` any pair | Task 1 |
| §3b `liftToWorld` / `liftToFrame` | Task 2 |
| §3c covariant gradient `G` | Task 3 |
| §3d `GᵀWG` == Ambient covariant (consistency) | Task 3 (consistency test) |
| §4 C++ API (`nxr::manifold::differential`) | Tasks 1–3 |
| §5 MEX (`frameTransport`, `operators gradient3D`, lifts) | Task 4 |
| §6 layout (`[nV×3]` boundary; component-major `3N` internal) | Tasks 3–4 (`Map`/`(:)` flatten) |
| §7 error handling (range / shape) | Task 1 (`frameTransport`), Task 2 (lifts) |
| §8 tests (orthogonality, flatness, artifact removal, consistency, round-trip) | Tasks 1–4 |

**Placeholder scan:** none — every step has complete code and exact commands. The one "reconcile if the sign is off" note in Task 3 gives a concrete reconciliation target (`covariant_laplacian.cpp` Ambient branch), not a placeholder.

**Type consistency:** `frameTransport` / `vertexFrameMatrices` / `liftToWorld` / `liftToFrame` / `covariantGradient` and the static `frameOf` helper are used with identical signatures across tasks. Layout (`E*p+e` rows, `N*q+v` cols, component-major `[a;b;c]`) is consistent between the Task-3 assembly and both the native (`Map<VectorXd>(Lloc.data(),3N)`) and MATLAB (`Lloc_c(:)`) flattens. The `GᵀWG` weight `W` uses `edgeCotanWeights` to match the Ambient covariant's `cotanL`.

**Consistency note:** the `GᵀWG == Ambient covariant` test must run on a **default (Levi-Civita) manifold**, where `vertexFrames`-derived frames equal `gauge().grid()` (the covariant's frame source). All tests use a default `Manifold` — correct. If a future variant tests a trivial-gauge manifold, the frames would differ and that consistency check would not hold (documented in §2).
