# Face-domain (dual) Dirac Operator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `operators().diracFace(τ)` — the **face-domain** (Poincaré dual) relative-Dirac family `L̃(τ) = (1−τ)(K̃⊗I₄) + τ·Ẽ`, a real-symmetric `4F×4F` curvature-aware operator with a **face-supported** eigenbasis, built from **exact per-face normals** aggregated over vertex stars. For expanding a face-integrated current-flux leadfield (a genuinely face-defined field).

**Architecture:** A new geometry-only free function `ops::dirac::extrinsicBlockFace(Manifold&)` assembles the extrinsic block `Ẽ = D̃ᵀ⋆_V D̃` from the per-vertex-star quaternionic Dirac matrix (face normals + vertex dual areas). A new `OperatorsFacet::diracFace(τ)` blends `Ẽ` (cached, `OperatorId::DiracFace`) with the DEC 2-form Laplacian `K̃ = d₁⋆₁⁻¹d₁ᵀ` (from the existing `DECOperators`). The MEX `operators` command gains a `diracFace` family. The eigenbasis reuses the existing `solve::eigen`/`solve::normalize` path. This is the `V↔F` dual of the merged `operators().dirac(τ)`.

**Tech Stack:** C++17, Eigen sparse, geometry-central (face normals + vertex dual areas + DEC operators + vertex one-ring traversal), MATLAB MEX, existing operators-facet + eigensolver infrastructure.

**Reference design:** `docs/superpowers/specs/2026-06-10-face-domain-dirac-operator-design.md`. Sibling (vertex) implementation: `src/dirac_operator.cpp`, `2026-06-10-extrinsic-dirac-operator-design.md`.

---

## Environment notes (from the vertex-Dirac build)

- Build with `bash scripts/build.sh Release`. Native test binaries land in **`build/`** (NOT `build/Release/`) on this macOS setup.
- clangd/IDE "file not found" / "undeclared Eigen" diagnostics are KNOWN NOISE — ignore them; the build script is the source of truth.
- The eigensolver API is `nxr::manifold::solve::eigen(K, M, k)` and `solve::normalize(U, M)` (NOT `solveEigenmodes`/`normalizeEigenmodes`); `EigenResult` has `.eigenvectors` / `.eigenvalues`.
- `Error`/`ErrorCode` resolve unqualified inside `nxr::manifold::*`.
- `leftMulImag` is a file-static helper already in `src/dirac_operator.cpp` — reuse it (add the new function to that same file).

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/dirac_operator.cpp` | add `ops::dirac::extrinsicBlockFace(Manifold&)` — the `Ẽ = D̃ᵀ⋆_V D̃` assembly | Modify |
| `include/nxr/compute.h` | declare `extrinsicBlockFace`; add `OperatorId::DiracFace`; add `cacheDiracFace_` + `diracFaceExtrinsicBlockCached_()` / `diracFaceFamily_()` | Modify |
| `include/nxr/facets.h` | add `OperatorsFacet::diracFace(double tau)` | Modify |
| `src/facets.cpp` | `OperatorId::DiracFace` cache cases; `diracFaceExtrinsicBlockCached_`, `diracFaceFamily_`, `OperatorsFacet::diracFace` bodies | Modify |
| `CMakeLists.txt` | register `test_dirac_face_operator` | Modify |
| `test/test_dirac_face_operator.cpp` | native tests | **Create** |
| `bindings/mex/src/nxr_compute_mex.cpp` | `diracFace` family in `cmdOperators` + help | Modify |
| `bindings/mex/test/test_dirac_face_operator.m` | MATLAB test | **Create** |
| `CLAUDE.md` | operators row + face-Dirac note | Modify |

**Storage convention:** the `4F` dimension is **face-interleaved** — face `f`, quaternion component `c ∈ {0,1,2,3}` (`[w,x,y,z]`) is `4*f + c`. `K̃ ⊗ I₄` places scalar entry `(u,v)` at rows `4u..4u+3`, cols `4v..4v+3` (diagonal within the 4-block) — i.e. `kron(K̃, I₄)` (face-major).

---

### Task 1: Extrinsic block `Ẽ` (library)

**Files:**
- Modify: `include/nxr/compute.h` (declaration, next to `ops::dirac::extrinsicBlock`)
- Modify: `src/dirac_operator.cpp` (the assembly)
- Modify: `CMakeLists.txt` (register test)
- Create: `test/test_dirac_face_operator.cpp`

- [ ] **Step 1: Declare the free function** in `include/nxr/compute.h`. Find the existing `namespace dirac { Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m); }` block (inside `namespace nxr::manifold::ops`) and extend it to:

```cpp
namespace dirac {
Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m);
// Face-domain (Poincaré dual) extrinsic block: Ẽ = D̃ᵀ ⋆_V D̃, the [4F×4F]
// real-symmetric PSD operator from EXACT per-face normals aggregated over vertex
// stars (the dual of extrinsicBlock). Quaternion order [w,x,y,z]; index 4*f+c.
// Geometry-only (face normals + vertex dual areas). Closed-mesh v1: boundary
// vertices (open stars) are skipped — exact on closed cortical hemispheres.
Eigen::SparseMatrix<double> extrinsicBlockFace(Manifold& m);
}  // namespace dirac
```

- [ ] **Step 2: Write the failing test** — create `test/test_dirac_face_operator.cpp`:

```cpp
#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <Eigen/Eigenvalues>
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

static void testExtrinsicBlockFace() {
    std::cout << "\n=== diracFace: extrinsic block Ẽ ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int Fn = m.nF();   // 20

    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlockFace(m);
    EXPECT(E.rows() == 4*Fn && E.cols() == 4*Fn, "Ẽ is [4F, 4F] = [80, 80]");

    Eigen::SparseMatrix<double> asym = E - Eigen::SparseMatrix<double>(E.transpose());
    EXPECT(asym.norm() < 1e-10, "Ẽ is symmetric");

    Eigen::MatrixXd dense(E);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "Ẽ is positive-semidefinite");
    EXPECT(E.norm() > 1e-6, "Ẽ is nonzero on a curved mesh");

    // Per-face constant is in the kernel (telescoping around each vertex star).
    // x = (c,c,...,c) with c a fixed quaternion → Ẽ x ≈ 0.
    Eigen::VectorXd x(4*Fn);
    for (int f = 0; f < Fn; ++f) { x(4*f+0)=0.3; x(4*f+1)=-0.7; x(4*f+2)=0.2; x(4*f+3)=0.5; }
    EXPECT((E * x).cwiseAbs().maxCoeff() < 1e-10, "Ẽ·(face-constant) = 0 (telescoping kernel)");
}

// A closed octahedron has flat faces but is curved as a whole — keep the curved
// check on the icosphere. For the flat-kernel check, a single planar fan: all
// face normals equal ⇒ every N_{k+1}-N_{k-1} = 0 ⇒ Ẽ = 0.
static void flatPatch(std::vector<double>& V, std::vector<int32_t>& F) {
    V = { 0,0,0,  1,0,0,  0.5,1,0,  -0.5,1,0,  -1,0,0,  -0.5,-1,0,  0.5,-1,0 };
    F = { 0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1 };
}

static void testFlatKernelFace() {
    std::cout << "\n=== diracFace: flat-region kernel ===\n";
    std::vector<double> V; std::vector<int32_t> F; flatPatch(V, F);
    Manifold m(V.data(), 7, F.data(), 6);
    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlockFace(m);
    // All face normals identical ⇒ Ẽ ≡ 0. (Only vertex 0 is interior here; its
    // star's face normals are all +z, so its rows vanish too.)
    EXPECT(E.norm() < 1e-10, "Ẽ vanishes on a flat patch (pure kernel)");
}

int main() {
    testExtrinsicBlockFace();
    testFlatKernelFace();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
```

- [ ] **Step 3: Register the test** in `CMakeLists.txt`, after the `test_dirac_operator` block:

```cmake
    add_executable(test_dirac_face_operator test/test_dirac_face_operator.cpp)
    target_link_libraries(test_dirac_face_operator PRIVATE nxr_compute)
    add_test(NAME test_dirac_face_operator COMMAND test_dirac_face_operator)
```

- [ ] **Step 4: Build to verify the test FAILS** (link error — `extrinsicBlockFace` undefined):

Run: `bash scripts/build.sh Release 2>&1 | tail -20`
Expected: FAIL — undefined reference to `nxr::manifold::ops::dirac::extrinsicBlockFace`.

- [ ] **Step 5: Implement `extrinsicBlockFace`** — append to `src/dirac_operator.cpp`, inside `namespace nxr::manifold::ops::dirac`, after `extrinsicBlock` (it reuses the file-static `leftMulImag`):

```cpp
Eigen::SparseMatrix<double> extrinsicBlockFace(Manifold& m) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireFaceNormals();        // EXACT per-face normals (the Gauss map at faces)
    geom.requireVertexDualAreas();    // ⋆_V measure

    const int N  = m.nV();
    const int Fn = m.nF();

    auto vec3 = [](const Vector3& u) { return Eigen::Vector3d(u.x, u.y, u.z); };

    // Rectangular real operator D̃ : ℍ^F → ℍ^V  [4V × 4F].
    // For each INTERIOR vertex v with cyclically ordered incident faces f_0..f_{d-1}:
    //   block(v, f_k) = -leftMulImag(N_{f_{k+1}} - N_{f_{k-1}}) / (2 Ã_v).
    // Boundary vertices (open star) are skipped — exact on closed meshes.
    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve(static_cast<size_t>(Fn) * 6 * 16);

    for (Vertex v : mesh.vertices()) {
        if (v.isBoundary()) continue;
        const int vi = static_cast<int>(v.getIndex());
        const double Av = geom.vertexDualAreas[v];
        if (Av <= 0.0)
            throw Error(ErrorCode::InvalidInput,
                "dirac::extrinsicBlockFace: degenerate (zero-area) vertex dual cell",
                "Vertex index " + std::to_string(vi) + "; fix mesh quality first.");

        // Incident faces in cyclic order (one per outgoing halfedge around v).
        std::vector<int> fid;
        std::vector<Eigen::Vector3d> nrm;
        for (Halfedge he : v.outgoingHalfedges()) {
            Face f = he.face();
            fid.push_back(static_cast<int>(f.getIndex()));
            nrm.push_back(vec3(geom.faceNormals[f]));
        }
        const int d = static_cast<int>(fid.size());
        const double s = -1.0 / (2.0 * Av);
        for (int k = 0; k < d; ++k) {
            Eigen::Vector3d dN = nrm[(k + 1) % d] - nrm[(k - 1 + d) % d];
            Eigen::Matrix4d B = s * leftMulImag(dN);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (B(a, b) != 0.0)
                        TD.emplace_back(4 * vi + a, 4 * fid[k] + b, B(a, b));
        }
    }
    Eigen::SparseMatrix<double> D(4 * N, 4 * Fn);
    D.setFromTriplets(TD.begin(), TD.end());

    // Vertex dual-area mass ⋆_V : [4V × 4V] diagonal (Ã_v on each of the 4 rows).
    std::vector<Eigen::Triplet<double>> TW;
    TW.reserve(static_cast<size_t>(4) * N);
    for (Vertex v : mesh.vertices()) {
        const int vi = static_cast<int>(v.getIndex());
        const double Av = geom.vertexDualAreas[v];
        for (int a = 0; a < 4; ++a) TW.emplace_back(4 * vi + a, 4 * vi + a, Av);
    }
    Eigen::SparseMatrix<double> WV(4 * N, 4 * N);
    WV.setFromTriplets(TW.begin(), TW.end());

    // Materialise D̃ᵀ before the chain (Eigen sparse aliasing — see extrinsicBlock).
    Eigen::SparseMatrix<double> E = (Eigen::SparseMatrix<double>(D.transpose()) * WV * D).pruned();
    E.makeCompressed();
    return E;
}
```

(If `geom.faceNormals` / `requireFaceNormals()` or `geom.vertexDualAreas` / `requireVertexDualAreas()` don't compile as written, confirm the exact GC accessor — grep `src/curvatures.cpp` / `src/normals.cpp` / `src/mesh_operators.cpp` for `faceNormals` / `vertexDualAreas` usage — and adapt minimally. `Vector3` has `.x/.y/.z`.)

- [ ] **Step 6: Build and run the test:**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_dirac_face_operator`
Expected: `ALL PASSED`.

- [ ] **Step 7: Commit:**

```bash
git add include/nxr/compute.h src/dirac_operator.cpp CMakeLists.txt test/test_dirac_face_operator.cpp
git commit -m "feat(dirac): face-domain extrinsic block Ẽ = D̃ᵀ⋆_V D̃ from face normals"
```

---

### Task 2: `operators().diracFace(τ)` family + cache

**Files:**
- Modify: `include/nxr/compute.h` (`OperatorId::DiracFace`; `cacheDiracFace_`; helper decls)
- Modify: `include/nxr/facets.h` (`OperatorsFacet::diracFace`)
- Modify: `src/facets.cpp` (cache cases + bodies)
- Modify: `test/test_dirac_face_operator.cpp` (facet tests)

- [ ] **Step 1: Extend `OperatorId`** in `include/nxr/compute.h` — append `DiracFace`:

```cpp
enum class OperatorId {
    LaplacianCotan, LaplacianGraph, LaplacianConnection, LaplacianCovariant,
    Dec, MassLumped, MassGalerkin, Gradient3D, Dirac, DiracFace
};
```

- [ ] **Step 2: Add the cache slot + helper declarations** in `include/nxr/compute.h`. After the `cacheDirac_;` member:

```cpp
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDiracFace_;  // face extrinsic block Ẽ
```

After the `diracFamily_(double tau);` declaration:

```cpp
    // Face-domain relative-Dirac extrinsic block Ẽ (4F×4F), cached (OperatorId::DiracFace).
    const Eigen::SparseMatrix<double>& diracFaceExtrinsicBlockCached_();
    // Assemble L̃(τ) = (1−τ)(K̃⊗I₄) + τ·Ẽ by value, K̃ = d₁⋆₁⁻¹d₁ᵀ. τ ∈ [0,1].
    Eigen::SparseMatrix<double> diracFaceFamily_(double tau);
```

- [ ] **Step 3: Add `diracFace` to the facet** in `include/nxr/facets.h`, after the `dirac(double tau)` declaration:

```cpp
    // diracFace(tau): the FACE-domain (dual) relative-Dirac family
    // L̃(τ) = (1−τ)(K̃⊗I₄) + τ·Ẽ, a [4F×4F] real symmetric sparse matrix, by value.
    // τ=0 ⇒ DEC 2-form Laplacian ⊗ I₄; τ=1 ⇒ pure extrinsic face Dirac Ẽ. τ ∈ [0,1].
    Eigen::SparseMatrix<double> diracFace(double tau) const;
```

- [ ] **Step 4: Write the failing facet tests** — append to `test/test_dirac_face_operator.cpp` before `main()`:

```cpp
// K̃ = d₁ ⋆₁⁻¹ d₁ᵀ, the DEC 2-form Laplacian (independent oracle for the τ=0 anchor).
static Eigen::SparseMatrix<double> twoFormLaplacian(Manifold& m) {
    const ops::DECOperators& dec = m.operators().dec();
    Eigen::SparseMatrix<double> d1t = dec.d1.transpose();
    return dec.d1 * dec.hodge1Inverse * d1t;     // [F×E][E×E][E×F] = [F×F]
}

// K ⊗ I₄ in the face-interleaved 4f+c layout (= kron(K, I₄)).
static Eigen::SparseMatrix<double> kron4(const Eigen::SparseMatrix<double>& K) {
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(K.nonZeros()) * 4);
    for (int k = 0; k < K.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it)
            for (int c = 0; c < 4; ++c)
                T.emplace_back(4 * static_cast<int>(it.row()) + c,
                               4 * static_cast<int>(it.col()) + c, it.value());
    Eigen::SparseMatrix<double> out(4 * K.rows(), 4 * K.cols());
    out.setFromTriplets(T.begin(), T.end());
    return out;
}

static void testDiracFaceFamily() {
    std::cout << "\n=== diracFace: family L̃(τ) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int Fn = m.nF();

    // HEADLINE anchor: diracFace(0) == K̃ ⊗ I₄.
    Eigen::SparseMatrix<double> L0 = m.operators().diracFace(0.0);
    Eigen::SparseMatrix<double> anchor = kron4(twoFormLaplacian(m));
    EXPECT((L0 - anchor).norm() < 1e-12, "diracFace(0) == K̃ ⊗ I₄ (intrinsic anchor)");

    for (double tau : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        Eigen::SparseMatrix<double> L = m.operators().diracFace(tau);
        bool shape = (L.rows() == 4*Fn && L.cols() == 4*Fn);
        Eigen::SparseMatrix<double> asym = L - Eigen::SparseMatrix<double>(L.transpose());
        EXPECT(shape && asym.norm() < 1e-10,
               std::string("diracFace(") + std::to_string(tau) + ") is [4F×4F] symmetric");
    }

    EXPECT((m.operators().diracFace(1.0) - ops::dirac::extrinsicBlockFace(m)).norm() < 1e-12,
           "diracFace(1) == extrinsicBlockFace");

    double tau = 0.4;
    Eigen::SparseMatrix<double> blend =
        (1.0 - tau) * m.operators().diracFace(0.0) + tau * m.operators().diracFace(1.0);
    EXPECT((m.operators().diracFace(tau) - blend).norm() < 1e-12, "diracFace(τ) is the convex blend");

    Eigen::MatrixXd dense(m.operators().diracFace(0.5));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "diracFace(0.5) is PSD");

    bool threw = false;
    try { m.operators().diracFace(1.5); } catch (const std::exception&) { threw = true; }
    EXPECT(threw, "diracFace(τ>1) throws");
    bool threwNeg = false;
    try { m.operators().diracFace(-0.1); } catch (const std::exception&) { threwNeg = true; }
    EXPECT(threwNeg, "diracFace(τ<0) throws");
}

static void testDiracFaceCache() {
    std::cout << "\n=== diracFace: cache lifecycle ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    EXPECT(!m.isOperatorCached(OperatorId::DiracFace), "DiracFace not cached initially");
    m.operators().diracFace(0.5);
    EXPECT(m.isOperatorCached(OperatorId::DiracFace), "diracFace(τ>0) caches Ẽ");
    m.releaseOperator(OperatorId::DiracFace);
    EXPECT(!m.isOperatorCached(OperatorId::DiracFace), "releaseOperator(DiracFace) clears Ẽ");
    m.operators().diracFace(0.0);
    EXPECT(!m.isOperatorCached(OperatorId::DiracFace), "diracFace(0) does not build Ẽ");
}

static void testDiracFaceEigenbasis() {
    std::cout << "\n=== diracFace: eigenbasis ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto& geom = m.geometry(); geom.requireFaceAreas();
    const int Fn = m.nF();

    Eigen::SparseMatrix<double> L = m.operators().diracFace(0.5);
    // Face-area mass B̃ = diag(A_f) ⊗ I₄ (SPD, commutes with right-ℍ-mult).
    std::vector<Eigen::Triplet<double>> TB; TB.reserve(4*Fn);
    for (auto f : m.mesh().faces())
        for (int c = 0; c < 4; ++c)
            TB.emplace_back(4*static_cast<int>(f.getIndex())+c,
                            4*static_cast<int>(f.getIndex())+c, geom.faceAreas[f]);
    Eigen::SparseMatrix<double> B(4*Fn, 4*Fn); B.setFromTriplets(TB.begin(), TB.end());

    const int k = 8;
    solve::EigenResult er = solve::eigen(L, B, k);
    Eigen::MatrixXd Phi = solve::normalize(er.eigenvectors, B);
    Eigen::MatrixXd gram = Phi.transpose() * (B * Phi);
    EXPECT((gram - Eigen::MatrixXd::Identity(gram.rows(), gram.cols())).cwiseAbs().maxCoeff() < 1e-9,
           "ΦᵀB̃Φ ≈ I (B-orthonormal face eigenbasis)");
    EXPECT(er.eigenvalues.allFinite() && er.eigenvalues(0) <= er.eigenvalues(k-1),
           "eigenvalues finite & ascending");
    auto groupSpread = [&](int g) {
        double lo = er.eigenvalues.segment(4*g, 4).minCoeff();
        double hi = er.eigenvalues.segment(4*g, 4).maxCoeff();
        return (hi - lo) / (1.0 + std::abs(hi));
    };
    EXPECT(groupSpread(0) < 1e-4 && groupSpread(1) < 1e-4,
           "eigenvalues form 4-fold quaternionic multiplets");
}
```

Add the three calls in `main()` before the summary:

```cpp
    testDiracFaceFamily();
    testDiracFaceCache();
    testDiracFaceEigenbasis();
```

(Verify `dec.hodge1Inverse` and `dec.d1` are the exact `DECOperators` field names — grep `include/nxr/compute.h` for `struct DECOperators`; the vertex hodge view exposes `h1inv()` so the inverse exists. Verify `solve::eigen`/`solve::normalize` as in the vertex test.)

- [ ] **Step 5: Build to verify failure** (compile/link — `diracFace`/`DiracFace`/helpers undefined):

Run: `bash scripts/build.sh Release 2>&1 | tail -20`

- [ ] **Step 6: Wire the cache cases** in `src/facets.cpp`. In `isOperatorCached`, after the `OperatorId::Dirac` case:

```cpp
        case OperatorId::DiracFace:           return (bool)cacheDiracFace_;
```

In `releaseOperator`, after the `OperatorId::Dirac` case:

```cpp
        case OperatorId::DiracFace:           cacheDiracFace_.reset();           break;
```

- [ ] **Step 7: Implement the helpers** in `src/facets.cpp`, after `diracFamily_`:

```cpp
const Eigen::SparseMatrix<double>& Manifold::diracFaceExtrinsicBlockCached_() {
    if (!cacheDiracFace_)
        cacheDiracFace_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::extrinsicBlockFace(*this));
    return *cacheDiracFace_;
}

// L̃(τ) = (1−τ)(K̃ ⊗ I₄) + τ·Ẽ, K̃ = d₁⋆₁⁻¹d₁ᵀ (DEC 2-form Laplacian). Builds each
// term only when its coefficient is nonzero — τ=0 never assembles Ẽ; τ=1 never
// builds K̃. Same NaN-safe guard + exact-fast-path reasoning as diracFamily_.
Eigen::SparseMatrix<double> Manifold::diracFaceFamily_(double tau) {
    if (!(tau >= 0.0 && tau <= 1.0))
        throw Error(ErrorCode::InvalidInput, "diracFace: tau must be in [0,1]",
                    "Got tau=" + std::to_string(tau) + ".");
    const int Fn = nF();

    Eigen::SparseMatrix<double> K4;
    if (tau < 1.0) {
        const ops::DECOperators& dec = decOperators();
        Eigen::SparseMatrix<double> d1t = dec.d1.transpose();
        Eigen::SparseMatrix<double> Ktilde = dec.d1 * dec.hodge1Inverse * d1t;  // [F×F]
        std::vector<Eigen::Triplet<double>> T;
        T.reserve(static_cast<size_t>(Ktilde.nonZeros()) * 4);
        for (int k = 0; k < Ktilde.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(Ktilde, k); it; ++it)
                for (int c = 0; c < 4; ++c)
                    T.emplace_back(4 * static_cast<int>(it.row()) + c,
                                   4 * static_cast<int>(it.col()) + c, it.value());
        K4.resize(4 * Fn, 4 * Fn);
        K4.setFromTriplets(T.begin(), T.end());
    }
    if (tau == 0.0) { K4.makeCompressed(); return K4; }
    const auto& E = diracFaceExtrinsicBlockCached_();
    if (tau == 1.0) { Eigen::SparseMatrix<double> out = E; out.makeCompressed(); return out; }
    Eigen::SparseMatrix<double> L = (1.0 - tau) * K4 + tau * E;
    L.makeCompressed();
    return L;
}
```

- [ ] **Step 8: Implement the facet accessor** in `src/facets.cpp`, next to `OperatorsFacet::dirac`:

```cpp
Eigen::SparseMatrix<double> OperatorsFacet::diracFace(double tau) const { return m_.diracFaceFamily_(tau); }
```

- [ ] **Step 9: Build and run:**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_dirac_face_operator`
Expected: `ALL PASSED`.

- [ ] **Step 10: Regression sweep:**

Run: `./build/test_dirac_operator && ./build/test_operators_facet && ./build/test_facets && ./build/test_intrinsic_delaunay`
Expected: all `ALL PASSED`.

- [ ] **Step 11: Commit:**

```bash
git add include/nxr/compute.h include/nxr/facets.h src/facets.cpp test/test_dirac_face_operator.cpp
git commit -m "feat(dirac): operators().diracFace(τ) family + OperatorId::DiracFace cache"
```

---

### Task 3: MEX surface `operators(h,'diracFace',τ)`

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp`
- Create: `bindings/mex/test/test_dirac_face_operator.m`

- [ ] **Step 1: Write the failing MATLAB test** — create `bindings/mex/test/test_dirac_face_operator.m` (mirror the harness of `bindings/mex/test/test_dirac_operator.m`):

```matlab
function test_dirac_face_operator
fprintf('[test_dirac_face_operator] starting\n');
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found', mexext);
addpath(hits(1).folder); clear nxr_compute

[V, F] = local_icosahedron();
nF = size(F,1);
h = nxr_compute('create', V, F);

% τ=0 anchor: diracFace(0) == kron(K̃, I4), K̃ = d1 * h1inv * d1'.
decs  = nxr_compute('operators', h, 'dec');               % struct .d0 .d1
h1inv = nxr_compute('operators', h, 'hodge', 'h1inv');
Ktilde = decs.d1 * h1inv * decs.d1';
anchor = kron(Ktilde, speye(4));
L0 = nxr_compute('operators', h, 'diracFace', 0);
assert(isequal(size(L0), [4*nF, 4*nF]), 'diracFace size wrong');
assert(norm(L0 - anchor, 'fro') < 1e-9, 'diracFace(0) != kron(K̃, I4)');

% τ=0.5 symmetric; real eigenvalues; 4-fold multiplets. Use a GENERALIZED problem
% against the face-area mass B̃ = diag(A_f)⊗I4 — L̃(0.5) has an exact 4-fold zero
% (the face-constant kernel), so plain eigs(L,'smallestabs') shift-inverts a
% singular matrix; the mass-paired solve is robust (mirrors the vertex test).
L = nxr_compute('operators', h, 'diracFace', 0.5);
assert(norm(L - L', 'fro') < 1e-9, 'diracFace(0.5) not symmetric');
v1 = V(F(:,1),:); v2 = V(F(:,2),:); v3 = V(F(:,3),:);
Af = 0.5 * sqrt(sum(cross(v2-v1, v3-v1, 2).^2, 2));    % per-face areas from coords
B  = kron(spdiags(Af, 0, nF, nF), speye(4));
d  = sort(real(eigs(L, B, 8, 'smallestabs')));
assert(all(abs(imag(eigs(L, B, 8, 'smallestabs'))) < 1e-8), 'eigenvalues not real');
assert(d(4) - d(1) < 1e-4*(1+abs(d(4))), 'first multiplet not 4-fold');
assert(d(8) - d(5) < 1e-4*(1+abs(d(8))), 'second multiplet not 4-fold');

% τ out of range errors.
threw = false;
try, nxr_compute('operators', h, 'diracFace', 2.0); catch, threw = true; end
assert(threw, 'diracFace(2.0) did not error');

nxr_compute('destroy', h);
fprintf('test_dirac_face_operator: ALL PASSED\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; ...
      0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
V = V ./ sqrt(sum(V.^2, 2));
F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; ...
     2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; ...
     4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
     5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
```

- [ ] **Step 2: Run to verify failure** (build, then MATLAB MCP `run_matlab_file` on the `.m`): expected error `family must be ...` (no `diracFace`).

- [ ] **Step 3: Add the `diracFace` branch** in `cmdOperators`, after the `dirac` branch:

```cpp
    } else if (family == "diracFace") {
        if (nrhs < 4)
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators diracFace: expected a scalar tau, "
                "nxr_compute('operators', h, 'diracFace', tau).");
        double tau = getDoubleArg(prhs[3]);
        plhs[0] = eigenSparseToMx(m.operators().diracFace(tau));   // [4F×4F], caches Ẽ
```

- [ ] **Step 4: Update the trailing family error message** to append `|diracFace`:

```cpp
            "operators: family must be laplacian|mass|hodge|dec|gradient3D|dirac|diracFace.");
```

- [ ] **Step 5: Update the help comment** above `cmdOperators`, after the `dirac` line:

```cpp
//   nxr_compute('operators', h, 'diracFace', tau)  % [4F×4F] FACE-domain (dual)
//                                                  % relative-Dirac, tau in [0,1]
```

- [ ] **Step 6: Rebuild and run the MATLAB test** (MATLAB MCP `run_matlab_file`). Expected: `test_dirac_face_operator: ALL PASSED`.

- [ ] **Step 7: Commit:**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_dirac_face_operator.m
git commit -m "feat(mex): operators(h,'diracFace',τ) returns the face-domain Dirac family"
```

---

### Task 4: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Extend the `operators` command row** — append `diracFace` to the family list (find the `` `nxr_compute('operators', h, family[, subtype])` `` row), same terse style:

```
; `diracFace` (the FACE-domain dual relative-Dirac family L̃(τ) = (1−τ)K̃⊗I₄ + τ·Ẽ, real `[4F×4F]`, `τ∈[0,1]`; exact face normals aggregated over vertex stars, face-supported eigenbasis)
```

- [ ] **Step 2: Add a short note** after the "Extrinsic Dirac operator" paragraph: `diracFace(τ)` is the `V↔F` Poincaré dual — `ψ` on faces, the Gauss map sampled at **exact face normals** (`e₁×e₂`, no vertex averaging) aggregated over **vertex stars** (the only cell that keeps the quaternionic coupling — the edge-only assembly degenerates to a scalar curvature²-Laplacian). Extrinsic block `Ẽ = D̃ᵀ⋆_V D̃` (cached `OperatorId::DiracFace`, gauge-independent); intrinsic anchor `K̃ = d₁⋆₁⁻¹d₁ᵀ` (DEC 2-form Laplacian); face-interleaved `4f+c` storage (`kron(K̃,I₄)`); 4-fold multiplets; `diracFace(0)` byte-matches `K̃⊗I₄`. For a face-integrated current-flux leadfield (genuinely face-defined). Closed-cortex v1 (boundary vertices / open stars skipped). Design: `docs/superpowers/specs/2026-06-10-face-domain-dirac-operator-design.md`.

- [ ] **Step 3: Commit:**

```bash
git add CLAUDE.md
git commit -m "docs(dirac): document operators().diracFace(τ) face-domain Dirac"
```

---

## Final verification (after all tasks)

- [ ] Full native sweep:

```bash
./build/test_dirac_face_operator
./build/test_dirac_operator
./build/test_operators_facet
./build/test_facets
./build/test_intrinsic_delaunay
./build/test_connection_laplacian
```

All must print `ALL PASSED` / `OK`.

- [ ] MEX: `bindings/mex/test/test_dirac_face_operator.m` passes via MATLAB MCP, and `bindings/mex/test/test_dirac_operator.m` + `test_bundle.m` still pass (no `operators`-command regression).

- [ ] Dispatch a final whole-branch code review (subagent) covering all four commits before finishing the branch.
