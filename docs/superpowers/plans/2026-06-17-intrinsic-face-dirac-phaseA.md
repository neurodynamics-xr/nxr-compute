# Intrinsic Face Dirac (`diracFaceIntrinsicD`) — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-order INTRINSIC face Dirac operator `D̃ᵢₙₜ` [4V×4F] to nxr-compute — the immersion (face-centroid) dual of `matrixFace`, exposed as `operators(h,'diracFaceIntrinsicD')`.

**Architecture:** Mirror the extrinsic `dirac::matrixFace` but build each per-vertex block from incident **face-centroid** differences `(C_{f_{k+1}} − C_{f_{k-1}})/(2Ã_v)` instead of face-normal differences — the dual of how `matrixIntrinsic` mirrors `matrix` using vertex positions instead of normals. Cache it like the other first-order Diracs, and expose it through the MEX + WASM `operators` dispatch.

**Tech Stack:** C++17 (Eigen, geometry-central), CMake build (`scripts/build.sh`); MATLAB MEX/WASM bindings. C++ validated by the local build + `test_dirac_face_operator`; MATLAB by `test_dirac_first_order.m` against a freshly built mex (via the brainstorm-dev MATLAB MCP).

**Commits:** repo `neurodynamics-xr/nxr-compute`; commit/push only when the user asks. Branch off `main`.

**Scope:** Phase A only (the operator + bindings + tests + build/release). Phase B (Brainstorm intrinsic face Helmholtz) is a separate later cycle.

---

### Task 1: C++ operator `matrixFaceIntrinsic` + caching + facet accessor

**Files:** `src/dirac_operator.cpp`, `include/nxr/compute.h`, `include/nxr/facets.h`, `src/facets.cpp`, `test/test_dirac_face_operator.cpp`.

- [ ] **Step 1: Write the failing C++ test** — in `test/test_dirac_face_operator.cpp`, add this function and call it from `main` (next to `testExtrinsicBlockFace`):

```cpp
static void testIntrinsicFaceDirac() {
    std::cout << "\n=== diracFaceIntrinsicD: D~_int (centroid immersion root) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV();   // 12
    const int Fn = m.nF();  // 20

    const Eigen::SparseMatrix<double>& Dt = m.operators().diracFaceIntrinsicD();
    EXPECT(Dt.rows() == 4*N && Dt.cols() == 4*Fn, "D~_int is [4V, 4F] = [48, 80]");
    EXPECT(Dt.norm() > 1e-6, "D~_int is nonzero on a curved mesh");

    // First-order: kills constant face-quaternion fields (telescoping cyclic centroid diffs).
    Eigen::VectorXd uf(4*Fn);
    for (int f = 0; f < Fn; ++f) { uf(4*f+0)=0.3; uf(4*f+1)=-0.7; uf(4*f+2)=0.2; uf(4*f+3)=0.5; }
    EXPECT((Dt * uf).cwiseAbs().maxCoeff() < 1e-10, "D~_int kills constant face fields");

    // Genuine quaternionic coupling (some 4x4 V-F block is non-scalar).
    Eigen::MatrixXd dD(Dt);
    bool coupled = false;
    for (int vi = 0; vi < N && !coupled; ++vi)
        for (int fj = 0; fj < Fn && !coupled; ++fj) {
            Eigen::Matrix4d blk = dD.block(4*vi, 4*fj, 4, 4);
            if (blk.norm() < 1e-12) continue;
            coupled = (blk - (blk.trace()/4.0)*Eigen::Matrix4d::Identity()).norm() > 1e-10;
        }
    EXPECT(coupled, "D~_int has genuine quaternionic coupling (non-scalar blocks)");

    // Cache stability: const-ref returns the same object on repeat.
    EXPECT(&m.operators().diracFaceIntrinsicD() == &Dt, "D~_int cached (same object)");
}
```
Add `testIntrinsicFaceDirac();` to `main()` alongside the existing calls.

- [ ] **Step 2: Build, expect FAIL** — `bash scripts/build.sh Release 2>&1 | tail -20`. Expected: compile error, `diracFaceIntrinsicD` is not a member of `OperatorsFacet`.

- [ ] **Step 3: Implement the operator** — in `src/dirac_operator.cpp`, after `matrixFace` (the closing `}` near line 226), add:

```cpp
Eigen::SparseMatrix<double> matrixFaceIntrinsic(Manifold& m) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    // INTRINSIC face-domain Dirac D̃_int : ℍ^F → ℍ^V  [4V × 4F]. Identical vertex-star
    // structure to matrixFace, but the per-block source is the IMMERSION of the dual
    // vertices (face CENTROIDS C) instead of the Gauss map (face normals N): the dual
    // mirror of how matrixIntrinsic uses vertex positions instead of normals. Geometry-
    // only (vertex positions + barycentric vertex dual areas). Closed-mesh v1.
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    if (mesh.hasBoundary())
        throw Error(ErrorCode::InvalidInput,
            "dirac::matrixFaceIntrinsic: open boundary unsupported (closed-mesh v1)",
            "Vertex stars must be closed; pass a closed mesh (e.g. a FreeSurfer hemisphere).");
    geom.requireVertexDualAreas();

    const int N  = m.nV();
    const int Fn = m.nF();
    auto vec3 = [](const Vector3& u) { return Eigen::Vector3d(u.x, u.y, u.z); };

    // dual immersion: per-face barycentric centroid
    std::vector<Eigen::Vector3d> C(Fn, Eigen::Vector3d::Zero());
    for (Face f : mesh.faces()) {
        Eigen::Vector3d c = Eigen::Vector3d::Zero();
        for (Vertex v : f.adjacentVertices()) c += vec3(geom.inputVertexPositions[v]);
        C[static_cast<int>(f.getIndex())] = c / 3.0;
    }

    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve(static_cast<size_t>(Fn) * 6 * 16);
    for (Vertex v : mesh.vertices()) {
        const int vi = static_cast<int>(v.getIndex());
        const double Av = geom.vertexDualAreas[v];
        if (Av <= 0.0)
            throw Error(ErrorCode::InvalidInput,
                "dirac::matrixFaceIntrinsic: degenerate (zero-area) vertex dual cell",
                "Vertex index " + std::to_string(vi) + "; fix mesh quality first.");
        std::vector<int> fid;
        for (Halfedge he : v.outgoingHalfedges()) fid.push_back(static_cast<int>(he.face().getIndex()));
        const int d = static_cast<int>(fid.size());
        const double s = -1.0 / (2.0 * Av);
        for (int k = 0; k < d; ++k) {
            Eigen::Vector3d dC = C[fid[(k + 1) % d]] - C[fid[(k - 1 + d) % d]];
            Eigen::Matrix4d B = s * leftMulImag(dC);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (B(a, b) != 0.0)
                        TD.emplace_back(4 * vi + a, 4 * fid[k] + b, B(a, b));
        }
    }
    Eigen::SparseMatrix<double> D(4 * N, 4 * Fn);
    D.setFromTriplets(TD.begin(), TD.end());
    D.makeCompressed();
    return D;
}
```

- [ ] **Step 4: Declare the operator** — in `include/nxr/compute.h`, next to `matrixFace(Manifold&)` (the `ops::dirac` declarations, ~line 456), add:
```cpp
Eigen::SparseMatrix<double> matrixFaceIntrinsic(Manifold& m);
```

- [ ] **Step 5: Add the cache slot + OperatorId + builder declaration** — in `include/nxr/compute.h`:
  - Extend the enum (line 86) to: `DiracD, DiracFaceD, DiracIntrinsicD, DiracFaceIntrinsicD`.
  - Add the cache member next to `cacheDiracIntrinsicD_` (~line 187):
    ```cpp
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDiracFaceIntrinsicD_; // first-order INTRINSIC face Dirac D̃_int [4V×4F]
    ```
  - Declare the private cached builder next to the other `dirac*MatrixCached_()` declarations:
    ```cpp
    const Eigen::SparseMatrix<double>& diracFaceIntrinsicMatrixCached_();
    ```

- [ ] **Step 6: Wire cache invalidation + builder + facet method** — in `src/facets.cpp`:
  - In the `has(...)` switch (~line 113) add: `case OperatorId::DiracFaceIntrinsicD: return (bool)cacheDiracFaceIntrinsicD_;`
  - In the `reset(...)` switch (~line 132) add: `case OperatorId::DiracFaceIntrinsicD: cacheDiracFaceIntrinsicD_.reset(); break;`
  - Add the cached builder next to `diracFaceMatrixCached_()` (~line 256), mirroring it:
    ```cpp
    const Eigen::SparseMatrix<double>& Manifold::diracFaceIntrinsicMatrixCached_() {
        if (!cacheDiracFaceIntrinsicD_)
            cacheDiracFaceIntrinsicD_ = std::make_unique<Eigen::SparseMatrix<double>>(
                ops::dirac::matrixFaceIntrinsic(*this));
        return *cacheDiracFaceIntrinsicD_;
    }
    ```
  - Add the facet accessor next to `diracFaceD()` (~line 451):
    ```cpp
    const Eigen::SparseMatrix<double>& OperatorsFacet::diracFaceIntrinsicD() const { return m_.diracFaceIntrinsicMatrixCached_(); }
    ```
  - In `include/nxr/facets.h`, declare it next to `diracFaceD()` (~line 174):
    ```cpp
    // diracFaceIntrinsicD(): first-order INTRINSIC face Dirac D̃_int [4V×4F], cached.
    // Centroid (immersion) dual of diracFaceD; geometry-only. Closed-mesh v1.
    const Eigen::SparseMatrix<double>& diracFaceIntrinsicD() const;
    ```

- [ ] **Step 7: Build + run, expect PASS** — `bash scripts/build.sh Release 2>&1 | tail -20 && ./build/Release/test_dirac_face_operator`. Expected: the new `[PASS]` lines for `D~_int` and overall `ALL PASSED`.

- [ ] **Step 8: Commit (when user asks)** `git commit -m "feat(dirac): intrinsic face Dirac D̃_int (centroid immersion root)"`

---

### Task 2: MEX + WASM bindings

**Files:** `bindings/mex/src/nxr_compute_mex.cpp`, `bindings/wasm/src/nxr_compute_wasm.cpp`, `bindings/wasm/js/index.d.ts`, `bindings/wasm/js/index.mjs`, `bindings/mex/test/test_dirac_first_order.m`.

- [ ] **Step 1: MEX dispatch** — in `nxr_compute_mex.cpp`, after the `diracIntrinsicD` arm (~line 1870), add:
```cpp
    } else if (family == "diracFaceIntrinsicD") {
        plhs[0] = eigenSparseToMx(m.operators().diracFaceIntrinsicD());   // [4V×4F], cached first-order INTRINSIC face D̃_int
```
Update the `else` error string (~line 1876) to append `|diracFaceIntrinsicD`, and add a line to the usage doc block (~line 1783):
```cpp
//   nxr_compute('operators', h, 'diracFaceIntrinsicD') % [4V×4F] first-order INTRINSIC
//                                               % face Dirac D̃_int (centroid immersion); cached
```

- [ ] **Step 2: WASM parity** — in `nxr_compute_wasm.cpp`, after the `diracIntrinsicD` arm (~line 349), add:
```cpp
            } else if (family == "diracFaceIntrinsicD") {
                return sparseToVal(m.operators().diracFaceIntrinsicD());
```
Append `|diracFaceIntrinsicD` to the wasm error string (~line 354). In `bindings/wasm/js/index.d.ts`, add `"diracFaceIntrinsicD"` to the no-arg `operators(...)` overload family union (~line 284). In `index.mjs`, add it to the no-arg families comment/list (~line 175) if the dispatch is name-driven (no code change if it forwards the string).

- [ ] **Step 3: Extend the MATLAB test** — in `bindings/mex/test/test_dirac_first_order.m`, after the `diracFaceD` block (before `nxr_compute('destroy', h)`), add:
```matlab
%% ---- face-domain (dual) INTRINSIC first-order D~_int : [4V x 4F] ----
DtI = nxr_compute('operators', h, 'diracFaceIntrinsicD');
assert(isequal(size(DtI), [4*nV, 4*nF]), 'diracFaceIntrinsicD size != [4V, 4F]');
assert(issparse(DtI) && isreal(DtI), 'diracFaceIntrinsicD must be real sparse');
assert(max(abs(DtI * Uf), [], 'all') < 1e-9, 'D~_int does not kill constant face fields');
assert(norm(DtI - nxr_compute('operators', h, 'diracFaceIntrinsicD'), 'fro') == 0, 'diracFaceIntrinsicD not stable across calls');
fprintf('  diracFaceIntrinsicD: [4V x 4F], kills constants, stable\n');
```

- [ ] **Step 4: Build the MEX + run the MATLAB test** — `bash scripts/build.sh Release` (builds the mex target too, or the dedicated mex build script). Then, via the brainstorm-dev MATLAB MCP, run `bindings/mex/test/test_dirac_first_order.m` against the freshly built `nxr_compute.mex*` on the build path. Expected: `test_dirac_first_order: ALL PASSED`.

- [ ] **Step 5: Commit (when user asks)** `git commit -m "feat(bindings): expose diracFaceIntrinsicD in MEX + WASM"`

---

### Task 3: Build, release, install

**Files:** none (build/release/install).

- [ ] **Step 1: Full local build + all dirac tests** — `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_dirac_operator && ./build/Release/test_dirac_face_operator`. Expected: `ALL PASSED` for both.

- [ ] **Step 2: Release (user-gated, outward)** — once the user approves: push the branch, open/merge a PR to `main`, then push a `v*` tag to trigger `publish-mex` (now hang-tolerant) and produce the multi-platform release zips including `nxr-compute-mex-r2023b` with the new operator.

- [ ] **Step 3: Install the fresh mex here** — download `mex-r2023b-macos-arm64` from the release/run; back up the current `~/.brainstorm/plugins/nxr-compute/nxr-compute-mex-r2023b/nxr_compute.mexmaca64` (`.bak-<date>`), copy the new one in; confirm in MATLAB: `h=nxr_compute('create',V,F); size(nxr_compute('operators',h,'diracFaceIntrinsicD'))` returns `[4V 4F]`.

- [ ] **Step 4: Hand off to Phase B** — the intrinsic face Dirac is now available to Brainstorm; Phase B (rewrite `bst_dirac_helmholtz_face` to use it + dual Poisson + dual reconstruction + comparison) is the next design→plan→implement cycle.

---

## Self-review notes

- **Spec coverage (Phase A):** centroid-immersion operator (Task 1 Step 3) ✓; declaration + cache + OperatorId + builder + facet accessor (Task 1 Steps 4–6) ✓; MEX + WASM bindings (Task 2) ✓; C++ test: size, kills-constants, quaternionic coupling, cache (Task 1 Step 1) ✓; MATLAB test (Task 2 Step 3) ✓; build/release/install (Task 3) ✓.
- **Naming consistency:** `ops::dirac::matrixFaceIntrinsic` (C++) → `OperatorId::DiracFaceIntrinsicD` → `cacheDiracFaceIntrinsicD_` → `diracFaceIntrinsicMatrixCached_()` → `OperatorsFacet::diracFaceIntrinsicD()` → MEX/WASM family string `'diracFaceIntrinsicD'`, used identically in tests.
- **Placeholder scan:** none; the only deferred item (the deep Galerkin-square-vs-DEC-2form comparison and planted-field recovery) is explicitly Phase B, not a Phase-A step.
- **Risk:** `leftMulImag`, `inputVertexPositions`, `vertexDualAreas` are all already used by `matrixFace`/`matrixIntrinsic` in the same file, so the new operator depends only on established helpers; line numbers are approximate — match on the adjacent code shown.
```
