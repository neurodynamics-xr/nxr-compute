# Intrinsic-Delaunay Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `create(V, F, struct('intrinsicDelaunay', true))` makes the cotan Laplacian + mass (and the ambient covariant) certified-PSD by assembling them on an intrinsic Delaunay triangulation. Default off ⇒ byte-identical.

**Architecture:** `Manifold` gains an optional `SignpostIntrinsicTriangulation` (built + `flipToDelaunay`'d at construction) and an `operatorGeometry()` accessor that returns the intrinsic interface when normalized; `assembleManifoldOperators` sources the cotan Laplacian / mass / dual areas from `operatorGeometry()` (normals stay on the embedded `geometry()`).

**Tech Stack:** C++17, Eigen, geometry-central, MATLAB MEX. Build `bash scripts/build.sh Release`. Native binaries in `build/`. MATLAB tests via MATLAB MCP.

**Spec:** `docs/superpowers/specs/2026-06-09-intrinsic-delaunay-phase1-design.md`

---

## Task 1: `Manifold` intrinsic-Delaunay support + operator reroute

**Files:** `include/nxr/compute.h`, `src/mesh_operators.cpp`, `test/test_intrinsic_delaunay.cpp` (create), `CMakeLists.txt`

- [ ] **Step 1: `compute.h` — forward decls + Manifold members**
Add forward declarations (near the existing GC forward decls) and extend the `Manifold` class:
```cpp
namespace geometrycentral { namespace surface {
  class IntrinsicGeometryInterface;
  class SignpostIntrinsicTriangulation;
}}
```
In `class Manifold`:
```cpp
    // add the trailing param (default false keeps every existing call identical)
    Manifold(const double* vertices, int nV,
             const int32_t* faces, int nF,
             bool intrinsicDelaunay = false);

    bool isIntrinsicDelaunay() const;
    // Geometry to assemble INTRINSIC-interface operators on (cotan, mass, dual
    // areas, …). Intrinsic Delaunay geometry when normalized, else the embedded
    // VertexPositionGeometry (itself an IntrinsicGeometryInterface). geometry()
    // is unchanged and still used for EXTRINSIC quantities (normals, frames).
    geometrycentral::surface::IntrinsicGeometryInterface& operatorGeometry();
```
And the private member:
```cpp
    std::unique_ptr<geometrycentral::surface::SignpostIntrinsicTriangulation> intrinsicTri_;
```

- [ ] **Step 2: `src/mesh_operators.cpp` — ctor + operatorGeometry()**
Add the include near the others:
```cpp
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h"
```
Extend the `Manifold` constructor (it currently builds `mesh_` + `geometry_`): add the `bool intrinsicDelaunay` param and, at the end:
```cpp
    if (intrinsicDelaunay) {
        intrinsicTri_ = std::make_unique<
            geometrycentral::surface::SignpostIntrinsicTriangulation>(*mesh_, *geometry_);
        intrinsicTri_->flipToDelaunay();
    }
```
Add the two new methods:
```cpp
bool Manifold::isIntrinsicDelaunay() const { return intrinsicTri_ != nullptr; }

geometrycentral::surface::IntrinsicGeometryInterface& Manifold::operatorGeometry() {
    if (intrinsicTri_) return *intrinsicTri_;
    return *geometry_;   // VertexPositionGeometry IS-A IntrinsicGeometryInterface (upcast)
}
```

- [ ] **Step 3: `src/mesh_operators.cpp` — reroute `assembleManifoldOperators`**
In `assembleManifoldOperators(Manifold& m, MassMatrixVariant variant)`, the body currently has `auto& geometry = m.geometry();` and uses it for the cotan Laplacian, mass (lumped/Galerkin), dual areas, totalArea, AND vertex normals. Split it:
```cpp
    auto& opGeom  = m.operatorGeometry();   // cotan, mass, dual areas, total area (intrinsic when normalized)
    auto& embGeom = m.geometry();           // vertex normals (extrinsic — embedded only)
```
Replace every `geometry.requireCotanLaplacian()/cotanLaplacian`,
`geometry.requireVertexLumpedMassMatrix()/vertexLumpedMassMatrix`,
`geometry.requireVertexGalerkinMassMatrix()/vertexGalerkinMassMatrix`,
`geometry.requireVertexDualAreas()/vertexDualAreas` with `opGeom.…`, and every
`geometry.requireVertexNormals()/vertexNormals` with `embGeom.…`. (Read the full
function and route each access; `totalArea` derives from the mass/areas → `opGeom`.)
Behavior when not normalized is unchanged (`opGeom == *geometry_`).

- [ ] **Step 4: Failing native test** — create `test/test_intrinsic_delaunay.cpp`:
```cpp
#include "nxr/compute.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

using nxr::manifold::Manifold;
namespace ops = nxr::manifold::ops;

static int g_failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { std::cerr << "  [FAIL] " << msg << "\n"; ++g_failures; } \
    else { std::cout << "  [PASS] " << msg << "\n"; } } while (0)

static double maxOffDiag(const Eigen::SparseMatrix<double>& L) {
    double m = -1e300;
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            if (it.row() != it.col()) m = std::max(m, it.value());
    return m;
}
static double minEig(const Eigen::SparseMatrix<double>& L) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(L));
    return es.eigenvalues().minCoeff();
}

static void testRhombus() {
    std::cout << "\n=== intrinsicDelaunay: non-Delaunay rhombus ===\n";
    std::vector<double>  V = {0,0,0,  2,0,0,  1,0.2,0,  1,-0.2,0};
    std::vector<int32_t> F = {0,2,1,  0,1,3};   // long-diagonal (0-1) split

    Manifold mRaw(V.data(), 4, F.data(), 2, /*intrinsicDelaunay=*/false);
    Manifold mN  (V.data(), 4, F.data(), 2, /*intrinsicDelaunay=*/true);
    EXPECT(!mRaw.isIntrinsicDelaunay(), "raw not normalized");
    EXPECT( mN.isIntrinsicDelaunay(),  "normalized flag set");

    auto Lraw = ops::assembleManifoldOperators(mRaw).cotanLaplacian;
    auto Ln   = ops::assembleManifoldOperators(mN).cotanLaplacian;
    EXPECT(Lraw.rows()==4 && Ln.rows()==4, "both cotan are 4x4 (same vertices)");

    // raw has a negative cotan weight (positive off-diagonal); normalized does not.
    EXPECT(maxOffDiag(Lraw) > 1e-9,  "raw cotan has a negative weight (non-Delaunay)");
    EXPECT(maxOffDiag(Ln)   < 1e-9,  "normalized cotan: all weights >= 0 (Delaunay)");
    // normalized is PSD (the certificate)
    EXPECT(minEig(Ln) > -1e-9,       "normalized cotan is PSD (min eig >= 0)");
}

static void testIcosphereNoOp() {
    std::cout << "\n=== intrinsicDelaunay: already-Delaunay icosphere (no-op) ===\n";
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<double> V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
                              0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    for (int i=0;i<12;++i){ double n=std::sqrt(V[3*i]*V[3*i]+V[3*i+1]*V[3*i+1]+V[3*i+2]*V[3*i+2]);
        V[3*i]/=n; V[3*i+1]/=n; V[3*i+2]/=n; }
    std::vector<int32_t> F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4,
        11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11,
        6,2,10, 8,6,7, 9,8,1};
    Manifold mRaw(V.data(), 12, F.data(), 20, false);
    Manifold mN  (V.data(), 12, F.data(), 20, true);
    auto Lraw = ops::assembleManifoldOperators(mRaw).cotanLaplacian;
    auto Ln   = ops::assembleManifoldOperators(mN).cotanLaplacian;
    double diff = (Eigen::MatrixXd(Lraw) - Eigen::MatrixXd(Ln)).cwiseAbs().maxCoeff();
    EXPECT(diff < 1e-9, "icosphere already Delaunay: normalized cotan == raw cotan");
}

int main() {
    testRhombus();
    testIcosphereNoOp();
    if (g_failures) { std::cerr << "\n" << g_failures << " failure(s)\n"; return 1; }
    std::cout << "\nALL PASSED\n"; return 0;
}
```
Register in `CMakeLists.txt`:
```cmake
    add_executable(test_intrinsic_delaunay test/test_intrinsic_delaunay.cpp)
    target_link_libraries(test_intrinsic_delaunay PRIVATE nxr_compute)
    add_test(NAME test_intrinsic_delaunay COMMAND test_intrinsic_delaunay)
```

- [ ] **Step 5: Build + run** — `bash scripts/build.sh Release 2>&1 | tail -8 && ./build/test_intrinsic_delaunay` → ALL PASSED. Also run a pre-existing native test to confirm no regression: `./build/test_eigen` and `./build/test_geometry_bundle`.
  - If the icosphere `normalized == raw` check fails with a non-trivial diff, the intrinsic mesh may reindex vertices — investigate (the SignpostIntrinsicTriangulation should be a 1:1 copy; if indices differ, that's a real finding to report, not to paper over).

- [ ] **Step 6: Commit**
```bash
git add include/nxr/compute.h src/mesh_operators.cpp test/test_intrinsic_delaunay.cpp CMakeLists.txt
git commit -m "feat(manifold): intrinsic-Delaunay context normalization (certified-PSD cotan + mass)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 1)
Read the `Manifold` ctor + `assembleManifoldOperators` in `src/mesh_operators.cpp` (to route each `geometry.` access correctly), confirm `SignpostIntrinsicTriangulation(ManifoldSurfaceMesh&, VertexPositionGeometry&)` + `flipToDelaunay()` signatures in `deps/geometry-central/include/geometrycentral/surface/signpost_intrinsic_triangulation.h`, and that `VertexPositionGeometry` upcasts to `IntrinsicGeometryInterface`. Follow TDD. Report Status, build result, the PASS lines, files, commit SHA, and whether the icosphere no-op (vertex-index alignment) held.

---

## Task 2: MEX `create` flag + MATLAB test

**Files:** `bindings/mex/src/nxr_compute_mex.cpp`, `bindings/mex/test/test_intrinsic_delaunay.m` (create)

- [ ] **Step 1: Extend `cmdCreate`** to accept an optional 3rd struct arg with `intrinsicDelaunay`. Read the current `cmdCreate` (it builds the `Manifold`/`ContextHolder`); add:
```cpp
    bool intrinsicDelaunay = false;
    if (nrhs >= 4 && mxIsStruct(prhs[3])) {
        const mxArray* f = mxGetField(prhs[3], 0, "intrinsicDelaunay");
        if (f && !mxIsEmpty(f))
            intrinsicDelaunay = mxIsLogical(f) ? mxGetLogicals(f)[0] : (mxGetScalar(f) != 0.0);
    }
```
and pass `intrinsicDelaunay` as the 5th arg to the `Manifold` constructor at the `make_shared`/`make_unique<Manifold>(...)` call. Keep the existing `nrhs` checks permissive enough to allow the optional 3rd arg (e.g. accept `nrhs == 3 || nrhs == 4`).

- [ ] **Step 2: Build** — `bash scripts/build.sh Release 2>&1 | tail -8` (clean + mexmaca64).

- [ ] **Step 3: Create `bindings/mex/test/test_intrinsic_delaunay.m`**:
```matlab
function test_intrinsic_delaunay
fprintf('[test_intrinsic_delaunay] starting\n');
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

% non-Delaunay rhombus (long-diagonal split, 1-based)
V = [0 0 0; 2 0 0; 1 0.2 0; 1 -0.2 0];
F = [1 3 2; 1 2 4];

hRaw = nxr_compute('create', V, F);
hN   = nxr_compute('create', V, F, struct('intrinsicDelaunay', true));

Graw = nxr_compute('geometry', hRaw, struct('operators',true));
GN   = nxr_compute('geometry', hN,   struct('operators',true));
Lraw = Graw.operators.laplacian;   % cotan
LN   = GN.operators.laplacian;
assert(isequal(size(Lraw), size(LN)) && size(LN,1)==4, 'both cotan 4x4');

offdiag = @(L) max(max(triu(L,1)));   % largest off-diagonal entry
assert(offdiag(Lraw) > 1e-9, 'raw cotan has a negative weight (non-Delaunay)');
assert(offdiag(LN)   < 1e-9, 'normalized cotan: all weights >= 0 (Delaunay)');
assert(min(eig(full(LN))) > -1e-9, 'normalized cotan PSD');

% ambient covariant under normalization is symmetric PSD
GaN = nxr_compute('gauge', hN, 'levi-civita', struct('operators',true,'coupling','ambient'));
C = GaN.operators.covariantLaplacian;
assert(norm(C - C','fro') < 1e-9, 'covariant symmetric');
assert(min(eig(full(C))) > -1e-9, 'covariant PSD under normalization');

nxr_compute('destroy', hRaw); nxr_compute('destroy', hN);
fprintf('ALL TESTS PASSED: test_intrinsic_delaunay\n');
end
```

- [ ] **Step 4: Run** — via MATLAB MCP (ToolSearch `select:mcp__plugin_brainstorm-dev_MATLAB__run_matlab_file`) run `bindings/mex/test/test_intrinsic_delaunay.m` → ALL TESTS PASSED. Also re-run `test_operators.m` (must still pass — the create-arg change and operator reroute must not break the non-normalized path). If MATLAB MCP unavailable, report DONE_WITH_CONCERNS (build + native only).

- [ ] **Step 5: Commit**
```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_intrinsic_delaunay.m
git commit -m "feat(mex): create(..., intrinsicDelaunay=true) → certified-PSD cotan + covariant

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 2)
Read `cmdCreate` to see exactly how the handle/`Manifold` is constructed and where to thread the flag + how `nrhs` is validated. Report Status, build result, the MATLAB outputs (verbatim or unavailable), files, commit SHA.

---

## Self-Review
| Spec item | Task |
|---|---|
| `intrinsicDelaunay` ctor flag; default byte-identical | Task 1 |
| `flipToDelaunay`, vertices preserved | Task 1 (icosphere no-op + 4x4 alignment) |
| cotan + mass from `operatorGeometry()` | Task 1 Step 3 |
| certified PSD (weights ≥0 + min eig ≥0) on non-Delaunay fixture | Task 1 + Task 2 tests |
| ambient covariant PSD under normalization | Task 2 |
| `create(..., intrinsicDelaunay=true)` MEX | Task 2 |
| non-normalized path unchanged | Task 1 (default), Task 2 (test_operators still passes) |

**Placeholders:** none. **Consistency:** `operatorGeometry()`/`isIntrinsicDelaunay()` declared T1, used in the reroute + tests; the rhombus fixture matches `fixDelaunay`'s.
