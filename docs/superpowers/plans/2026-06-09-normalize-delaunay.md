# Standalone `normalize` (Delaunay flip) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `nxr_compute('normalize', V, F) → [V, F']` — a stateless extrinsic Delaunay edge-flip mesh fix (geometry-central `fixDelaunay`), same vertices, new faces.

**Architecture:** A library function `normalizeDelaunay(V,F)` builds a `ManifoldSurfaceMesh`+`VertexPositionGeometry` (as the `Manifold` ctor does), runs `geometrycentral::surface::fixDelaunay`, reads back the re-triangulated faces. A stateless MEX `normalize` command marshals it.

**Tech Stack:** C++17, Eigen, geometry-central, MATLAB MEX. Build `bash scripts/build.sh Release`. Native binaries in `build/`. MATLAB tests via MATLAB MCP.

**Spec:** `docs/superpowers/specs/2026-06-09-normalize-delaunay-design.md`

---

## Task 1: `normalizeDelaunay` library function

**Files:** `include/nxr/compute.h`, `src/normalize_delaunay.cpp` (create), `test/test_normalize.cpp` (create), `CMakeLists.txt`

- [ ] **Step 1: Declare in `compute.h`** — in `namespace nxr::manifold` (near the `Manifold` class), add:
```cpp
// Extrinsic Delaunay edge-flip normalization (geometry-central fixDelaunay).
// Flips edges in place — same vertices (positions/count/indices), new faces.
// Best-effort PSD (reduces obtuse/negative cotan weights), not a certificate;
// the intrinsic-Delaunay certificate is a separate facility. Input must be a
// valid manifold (the underlying ManifoldSurfaceMesh throws otherwise).
struct DelaunayNormalization {
    Eigen::MatrixXi faces;  // [nF, 3] 0-based, re-triangulated; same vertex indices
    int flips = 0;          // edge flips performed (0 ⇒ already Delaunay)
};
DelaunayNormalization normalizeDelaunay(
    const double* vertices, int nV, const int32_t* faces, int nF);
```

- [ ] **Step 2: Failing native test** — create `test/test_normalize.cpp`:
```cpp
#include "nxr/compute.h"
#include <iostream>
#include <vector>
#include <set>

using nxr::manifold::normalizeDelaunay;

static int g_failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { \
    std::cerr << "  [FAIL] " << msg << "\n"; ++g_failures; } \
    else { std::cout << "  [PASS] " << msg << "\n"; } } while (0)

// True iff some face contains both a and b.
static bool edgePresent(const Eigen::MatrixXi& F, int a, int b) {
    for (int r = 0; r < F.rows(); ++r) {
        std::set<int> v{F(r,0), F(r,1), F(r,2)};
        if (v.count(a) && v.count(b)) return true;
    }
    return false;
}

static void testThinQuad() {
    std::cout << "\n=== normalizeDelaunay: thin quad (non-Delaunay) ===\n";
    // 0=(0,0) 1=(2,0) 2=(1,0.2) 3=(1,-0.2); split along the LONG diagonal 0–1.
    std::vector<double>  V = {0,0,0,  2,0,0,  1,0.2,0,  1,-0.2,0};
    std::vector<int32_t> F = {0,2,1,  0,1,3};   // 2 triangles, shared edge 0–1
    auto r = normalizeDelaunay(V.data(), 4, F.data(), 2);

    EXPECT(r.flips == 1, "exactly one flip");
    EXPECT(r.faces.rows() == 2 && r.faces.cols() == 3, "still 2 triangles");
    EXPECT(!edgePresent(r.faces, 0, 1), "long diagonal 0–1 removed");
    EXPECT(edgePresent(r.faces, 2, 3),  "short diagonal 2–3 present");
    // vertex indices stay in range
    EXPECT(r.faces.minCoeff() >= 0 && r.faces.maxCoeff() <= 3, "indices in [0,3]");
}

static void testAlreadyDelaunay() {
    std::cout << "\n=== normalizeDelaunay: already-Delaunay (icosahedron) ===\n";
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<double> V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
                              0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    for (int i=0;i<12;++i){ double n=std::sqrt(V[3*i]*V[3*i]+V[3*i+1]*V[3*i+1]+V[3*i+2]*V[3*i+2]);
        V[3*i]/=n; V[3*i+1]/=n; V[3*i+2]/=n; }
    std::vector<int32_t> F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4,
        11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11,
        6,2,10, 8,6,7, 9,8,1};
    auto r = normalizeDelaunay(V.data(), 12, F.data(), 20);
    EXPECT(r.flips == 0, "icosphere already Delaunay (0 flips)");
    EXPECT(r.faces.rows() == 20, "20 faces preserved");
}

int main() {
    testThinQuad();
    testAlreadyDelaunay();
    if (g_failures) { std::cerr << "\n" << g_failures << " failure(s)\n"; return 1; }
    std::cout << "\nALL PASSED\n"; return 0;
}
```
(Add `#include <cmath>` if needed.)

- [ ] **Step 3: Register the test in `CMakeLists.txt`** — mirror the other `add_executable`/`target_link_libraries`/`add_test` blocks:
```cmake
    add_executable(test_normalize test/test_normalize.cpp)
    target_link_libraries(test_normalize PRIVATE nxr_compute)
    add_test(NAME test_normalize COMMAND test_normalize)
```

- [ ] **Step 4: Build to confirm LINK failure** — `bash scripts/build.sh Release 2>&1 | grep -E "error:|normalizeDelaunay|undefined"` → undefined-reference.

- [ ] **Step 5: Implement `src/normalize_delaunay.cpp`** — mirror the `Manifold` ctor's mesh-building (`src/mesh_operators.cpp`) and read faces back via `getFaceVertexList()`:
```cpp
#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/remeshing.h"

namespace nxr::manifold {

using namespace geometrycentral;
using namespace geometrycentral::surface;

DelaunayNormalization normalizeDelaunay(
    const double* vertices, int nV, const int32_t* faces, int nF) {

    std::vector<std::vector<size_t>> polygons(nF);
    for (int i = 0; i < nF; ++i)
        polygons[i] = { static_cast<size_t>(faces[3*i]),
                        static_cast<size_t>(faces[3*i+1]),
                        static_cast<size_t>(faces[3*i+2]) };

    ManifoldSurfaceMesh mesh(polygons);
    VertexData<Vector3> positions(mesh);
    for (size_t i = 0; i < static_cast<size_t>(nV); ++i)
        positions[mesh.vertex(i)] = Vector3{ vertices[3*i], vertices[3*i+1], vertices[3*i+2] };
    VertexPositionGeometry geom(mesh, positions);

    DelaunayNormalization out;
    out.flips = static_cast<int>(fixDelaunay(mesh, geom));

    mesh.compress();  // ensure dense indexing before reading back
    std::vector<std::vector<size_t>> fvl = mesh.getFaceVertexList();
    out.faces.resize(static_cast<int>(fvl.size()), 3);
    for (size_t i = 0; i < fvl.size(); ++i)
        for (int k = 0; k < 3; ++k)
            out.faces(static_cast<int>(i), k) = static_cast<int>(fvl[i][k]);
    return out;
}

} // namespace nxr::manifold
```
VERIFY: `fixDelaunay` is in `geometrycentral/surface/remeshing.h` (confirmed). `getFaceVertexList()` is a `SurfaceMesh` member returning `std::vector<std::vector<size_t>>` (confirmed). Add `src/normalize_delaunay.cpp` to the `add_library(nxr_compute ...)` source list in `CMakeLists.txt`. If `mesh.compress()` isn't a valid call (check the API), drop it — flips don't change element counts so indexing stays dense.

- [ ] **Step 6: Build + run** — `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_normalize` → ALL PASSED (6 assertions). If "exactly one flip" fails with flips==0, the thin-quad diagonal orientation may already be Delaunay — try swapping the input split to `{0,2,1, 0,1,3}` vs `{0,1,2, 0,2,3}` and confirm which is the long-diagonal (0–1 shared) split; the long-diagonal one must flip. Do NOT weaken the test.

- [ ] **Step 7: Commit**
```bash
git add include/nxr/compute.h src/normalize_delaunay.cpp test/test_normalize.cpp CMakeLists.txt
git commit -m "feat(repair): add normalizeDelaunay (extrinsic Delaunay edge-flip fix)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 1)
Read the `Manifold` constructor in `src/mesh_operators.cpp` (mesh-building from V,F) and confirm `fixDelaunay` + `getFaceVertexList` signatures. Follow TDD. Report Status, build result, the test PASS lines, files, commit SHA.

---

## Task 2: MEX `normalize` command

**Files:** `bindings/mex/src/nxr_compute_mex.cpp`, `bindings/mex/test/test_normalize.m` (create)

- [ ] **Step 1: Add `cmdNormalize`** (stateless — mirror the stateless legacy path in an existing `cmd*` that parses V,F via `mxToVertexBuffer`/`mxToFaceBuffer`):
```cpp
void cmdNormalize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 3) throw std::invalid_argument(
        "nxr_compute('normalize', V, F) takes V and F");
    int nV = 0, nF = 0;
    std::vector<double>       Vbuf = mxToVertexBuffer(prhs[1], nV);  // [nV*3], row-major xyz
    std::vector<std::int32_t> Fbuf = mxToFaceBuffer(prhs[2], nF);    // [nF*3], 0-based

    auto r = nxr::manifold::normalizeDelaunay(Vbuf.data(), nV, Fbuf.data(), nF);

    plhs[0] = mxDuplicateArray(prhs[1]);   // V2 == V (passthrough, unchanged)
    // F2: nF×3 double, 1-based, column-major
    Eigen::MatrixXd Fd = r.faces.cast<double>().array() + 1.0;
    plhs[1] = eigenMatrixToMx(Fd);
    if (nlhs >= 3) {
        mxArray* nf = mxCreateDoubleMatrix(1,1,mxREAL);
        *mxGetPr(nf) = static_cast<double>(r.flips);
        plhs[2] = nf;
    }
}
```
VERIFY: `mxToVertexBuffer(prhs, nVout)` returns the vertex buffer in the same row-major xyz layout `normalizeDelaunay` expects (it must match the `Manifold` create path — read marshal.h). `eigenMatrixToMx` flattens column-major (Eigen default), matching MATLAB; `Fd` is `nF×3` so this yields the right MATLAB shape.

- [ ] **Step 2: Wire into dispatch** — add near the other stateless commands:
```cpp
        else if (cmd == "normalize")     cmdNormalize(nlhs, plhs, nrhs, prhs);
```
Also add `normalize` to the unknown-command help string if it enumerates commands.

- [ ] **Step 3: Create `bindings/mex/test/test_normalize.m`**:
```matlab
function test_normalize
fprintf('[test_normalize] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

% thin quad, split along the long diagonal (1-2 in 1-based) -> must flip
V = [0 0 0; 2 0 0; 1 0.2 0; 1 -0.2 0];
F = [1 3 2; 1 2 4];            % 1-based; shared edge 1-2 (the long diagonal)
[V2, F2, n] = nxr_compute('normalize', V, F);
assert(isequal(V2, V), 'vertices unchanged');
assert(n == 1, 'one flip');
assert(isequal(size(F2), size(F)), 'same #faces');
assert(all(F2(:) >= 1 & F2(:) <= 4), 'faces 1-based in range');
% long diagonal (1,2) gone; short diagonal (3,4) present
hasEdge = @(F,a,b) any(arrayfun(@(r) all(ismember([a b], F(r,:))), 1:size(F,1)));
assert(~hasEdge(F2,1,2), 'long diagonal removed');
assert(hasEdge(F2,3,4),  'short diagonal present');

% already-Delaunay icosahedron -> 0 flips
[Vi, Fi] = local_icosahedron();
[~, Fi2, ni] = nxr_compute('normalize', Vi, Fi);
assert(ni == 0, 'icosahedron already Delaunay');
assert(isequal(sortrows(sort(Fi2,2)), sortrows(sort(Fi,2))), 'faces unchanged as a set');

fprintf('ALL TESTS PASSED: test_normalize\n');
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

- [ ] **Step 4: Build + run** — `bash scripts/build.sh Release 2>&1 | tail -8` (clean + mexmaca64). Then via MATLAB MCP (ToolSearch `select:mcp__plugin_brainstorm-dev_MATLAB__run_matlab_file`) run `bindings/mex/test/test_normalize.m` → `ALL TESTS PASSED`. If the flip assertion fails, check the 1-based long-diagonal split matches the native test's fixture. If MATLAB MCP unavailable, report DONE_WITH_CONCERNS (build + native only).

- [ ] **Step 5: Commit**
```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_normalize.m
git commit -m "feat(mex): add stateless 'normalize' command (Delaunay edge-flip fix)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 2)
Read an existing stateless `cmd*` (one that parses raw V,F, not a handle) to mirror the `mxToVertexBuffer`/`mxToFaceBuffer` usage and the dispatch wiring. Confirm `eigenMatrixToMx` is the right real-matrix marshaler. Report Status, build result, the MATLAB output (verbatim or unavailable), files, commit SHA.

---

## Self-Review
| Spec item | Task |
|---|---|
| `normalizeDelaunay` via fixDelaunay, vertex-preserving | Task 1 |
| same V, new F (flips only) | Task 1 (impl), Task 2 (V2==V) |
| stateless `normalize` command, `[V2,F2,nFlips]` | Task 2 |
| 1-based round-trip at MEX boundary | Task 2 |
| non-Delaunay fixture flips; Delaunay fixture doesn't | Task 1 + Task 2 tests |

**Placeholders:** none. **Consistency:** `DelaunayNormalization`/`normalizeDelaunay` declared T1, used T2; the thin-quad fixture (long-diagonal split, shared edge between the two "end" vertices) is identical in the native (0-based 0–1) and MATLAB (1-based 1–2) tests.
