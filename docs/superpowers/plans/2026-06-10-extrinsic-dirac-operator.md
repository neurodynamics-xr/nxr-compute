# Extrinsic (Relative) Dirac Operator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `operators().dirac(τ)` — the Liu/Jacobson/Crane (SGP 2017) relative-Dirac family `L(τ) = (1−τ)(cotanL⊗I₄) + τ·E`, a real-symmetric `4V×4V` curvature-aware operator with a built-in eigenbasis, assembled from cortex geometry alone.

**Architecture:** A new geometry-only free function `ops::dirac::extrinsicBlock(Manifold&)` assembles the extrinsic Galerkin block `E = Dᵀ⋆_F D` from the per-face quaternionic Dirac matrix (Gauss map + face areas). A new `OperatorsFacet::dirac(τ)` sub-family blends `E` (cached, `OperatorId::Dirac`) with the existing cached cotan Laplacian `⊗ I₄`, returning by value. The MEX `operators` command gains a `dirac` family taking a numeric `τ`. The eigenbasis reuses the existing `solveEigenmodes`/`normalizeEigenmodes` path unchanged.

**Tech Stack:** C++17, Eigen sparse, geometry-central (vertex frames/normals + face areas), MATLAB MEX, existing nxr-compute operators-facet + eigensolver infrastructure.

**Reference design:** `docs/superpowers/specs/2026-06-10-extrinsic-dirac-operator-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/dirac_operator.cpp` | `ops::dirac::extrinsicBlock(Manifold&)` — the `E = Dᵀ⋆_F D` assembly | **Create** |
| `include/nxr/compute.h` | Declare `ops::dirac::extrinsicBlock`; add `OperatorId::Dirac`; add `cacheDirac_` slot + `diracExtrinsicBlockCached_()` / `diracFamily_()` private helpers | Modify |
| `include/nxr/facets.h` | Add `OperatorsFacet::dirac(double tau)` | Modify |
| `src/facets.cpp` | `OperatorId::Dirac` cases in `isOperatorCached`/`releaseOperator`; `diracExtrinsicBlockCached_`, `diracFamily_`, `OperatorsFacet::dirac` bodies | Modify |
| `CMakeLists.txt` | Add `src/dirac_operator.cpp` to the library; register `test_dirac_operator` | Modify |
| `test/test_dirac_operator.cpp` | Native tests: block shape/symmetry/PSD/kernel; `τ=0` anchor; blend; cache lifecycle; eigenbasis B-orthonormality | **Create** |
| `bindings/mex/src/nxr_compute_mex.cpp` | `dirac` family in `cmdOperators` (numeric `τ` arg) + help string | Modify |
| `bindings/mex/test/test_dirac_operator.m` | MATLAB: `operators(h,'dirac',0)` == `kron(I₄,cotanL)`; `eigs(L,B)` real; multiplets | **Create** |
| `CLAUDE.md` | Operators-command table row for `dirac`; storage-convention note | Modify |

**Storage convention (used throughout):** the `4V` index for vertex `v`, quaternion component `c ∈ {0,1,2,3}` (order `[w,x,y,z]`) is `4*v + c`. `cotanL ⊗ I₄` places scalar entry `(u,v)` at rows `4u..4u+3`, cols `4v..4v+3` (diagonal within the 4-block).

---

### Task 1: Extrinsic Galerkin block `E` (library)

**Files:**
- Create: `src/dirac_operator.cpp`
- Modify: `include/nxr/compute.h` (declaration, after `graphLaplacian` at line 399)
- Modify: `CMakeLists.txt` (add source at line 73 area; register test)
- Create: `test/test_dirac_operator.cpp`

- [ ] **Step 1: Declare the free function** in `include/nxr/compute.h`. The line `graphLaplacian(Manifold& m);` (line 399) sits inside `namespace nxr::manifold::ops`; insert this block on the next line, still inside that `ops` namespace:

```cpp
// Extrinsic block of the relative Dirac family (Liu/Jacobson/Crane, SGP 2017):
// E = Dᵀ ⋆_F D, the [4V×4V] real-symmetric PSD Galerkin form of the relative
// Dirac operator D_N (the shape-operator / Gauss-map energy). Quaternion order
// [w,x,y,z]; index 4*v+c. Geometry-only (vertex normals + face areas) — the one
// operator assembled directly rather than wrapped from geometry-central.
namespace dirac {
Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m);
}  // namespace dirac
```

- [ ] **Step 2: Write the failing test** — create `test/test_dirac_operator.cpp`:

```cpp
#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
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

// A single flat triangle fan in the z=0 plane (Gauss map constant → D_N kernel).
static void flatPatch(std::vector<double>& V, std::vector<int32_t>& F) {
    V = { 0,0,0,  1,0,0,  0.5,1,0,  -0.5,1,0,  -1,0,0,  -0.5,-1,0,  0.5,-1,0 };
    F = { 0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1 };
}

static void testExtrinsicBlock() {
    std::cout << "\n=== dirac: extrinsic block E ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV();   // 12

    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlock(m);
    EXPECT(E.rows() == 4*N && E.cols() == 4*N, "E is [4V, 4V] = [48, 48]");

    Eigen::SparseMatrix<double> asym = E - Eigen::SparseMatrix<double>(E.transpose());
    EXPECT(asym.norm() < 1e-10, "E is symmetric");

    // PSD: smallest eigenvalue of dense(E) >= -tol
    Eigen::MatrixXd dense(E);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "E is positive-semidefinite");

    // On a curved mesh E is NOT all-zero (shape operator is nontrivial)
    EXPECT(E.norm() > 1e-6, "E is nonzero on a curved mesh");
}

static void testFlatKernel() {
    std::cout << "\n=== dirac: flat-region kernel ===\n";
    std::vector<double> V; std::vector<int32_t> F; flatPatch(V, F);
    Manifold m(V.data(), 7, F.data(), 6);
    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlock(m);
    // Flat: Gauss map constant ⇒ N_r − N_q ≡ 0 ⇒ D ≡ 0 ⇒ E ≡ 0 (entirely kernel).
    EXPECT(E.norm() < 1e-10, "E vanishes on a flat patch (pure kernel)");
}

int main() {
    testExtrinsicBlock();
    testFlatKernel();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
```

- [ ] **Step 3: Register the test + source** in `CMakeLists.txt`. Add to the library sources after `src/covariant_differential.cpp` (line 73):

```cmake
  src/dirac_operator.cpp
```

And after the `test_covariant_differential` block (line 206-208):

```cmake
    add_executable(test_dirac_operator test/test_dirac_operator.cpp)
    target_link_libraries(test_dirac_operator PRIVATE nxr_compute)
    add_test(NAME test_dirac_operator COMMAND test_dirac_operator)
```

- [ ] **Step 4: Run the test to verify it fails** (link error — `extrinsicBlock` undefined):

Run: `bash scripts/build.sh Release 2>&1 | tail -20`
Expected: FAIL — undefined reference to `nxr::manifold::ops::dirac::extrinsicBlock`.

- [ ] **Step 5: Implement `src/dirac_operator.cpp`:**

```cpp
#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <array>
#include <vector>

namespace nxr::manifold::ops::dirac {

// 4×4 real matrix of LEFT-multiplication by a purely imaginary quaternion
// v = x·i + y·j + z·k, in component order [w,x,y,z]. Antisymmetric.
static Eigen::Matrix4d leftMulImag(const Eigen::Vector3d& v) {
    const double x = v.x(), y = v.y(), z = v.z();
    Eigen::Matrix4d L;
    L << 0, -x, -y, -z,
         x,  0, -z,  y,
         y,  z,  0, -x,
         z, -y,  x,  0;
    return L;
}

Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m) {
    using namespace geometrycentral::surface;

    // Gauss map: per-vertex unit normals from the embedded geometry. The Dirac
    // extrinsic term is genuinely extrinsic, so it always uses the TRUE embedding
    // (geometry()), independent of intrinsicDelaunay normalization.
    geometry::VertexFrames vf = geometry::vertexFrames(m);   // vf.normals = N [nV,3]
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireFaceAreas();

    const int N = m.nV();
    const int Fn = m.nF();

    // Rectangular real Dirac matrix D : 4F × 4V.
    // Per face f=ijk, area A: for each cyclic (p,q,r), block on column p is
    //   D_{f,p} = −leftMulImag(N_r − N_q) / (2A).
    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve(static_cast<size_t>(Fn) * 3 * 16);

    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        std::array<int,3> vid{};
        int c = 0;
        for (Vertex v : f.adjacentVertices()) vid[c++] = static_cast<int>(v.getIndex());

        for (int s = 0; s < 3; ++s) {
            const int p = vid[s];
            const int q = vid[(s + 1) % 3];
            const int r = vid[(s + 2) % 3];
            Eigen::Vector3d Nr = vf.normals.row(r).transpose();
            Eigen::Vector3d Nq = vf.normals.row(q).transpose();
            Eigen::Matrix4d B = (-1.0 / (2.0 * A)) * leftMulImag(Nr - Nq);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (B(a, b) != 0.0)
                        TD.emplace_back(4 * fi + a, 4 * p + b, B(a, b));
        }
    }
    Eigen::SparseMatrix<double> D(4 * Fn, 4 * N);
    D.setFromTriplets(TD.begin(), TD.end());

    // Face-area 2-form mass ⋆_F : 4F × 4F diagonal (A_f on each of the 4 rows).
    std::vector<Eigen::Triplet<double>> TW;
    TW.reserve(static_cast<size_t>(4) * Fn);
    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        for (int a = 0; a < 4; ++a) TW.emplace_back(4 * fi + a, 4 * fi + a, A);
    }
    Eigen::SparseMatrix<double> WF(4 * Fn, 4 * Fn);
    WF.setFromTriplets(TW.begin(), TW.end());

    Eigen::SparseMatrix<double> E = (Eigen::SparseMatrix<double>(D.transpose()) * WF * D).pruned();
    E.makeCompressed();
    return E;
}

}  // namespace nxr::manifold::ops::dirac
```

- [ ] **Step 6: Run the test to verify it passes:**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_dirac_operator`
Expected: `ALL PASSED` (4 PASS lines for block + 1 for flat kernel).

- [ ] **Step 7: Commit:**

```bash
git add include/nxr/compute.h src/dirac_operator.cpp CMakeLists.txt test/test_dirac_operator.cpp
git commit -m "feat(dirac): extrinsic Galerkin block E = DᵀWF D from the Gauss map"
```

---

### Task 2: `operators().dirac(τ)` family + cache (facet)

**Files:**
- Modify: `include/nxr/compute.h` (`OperatorId::Dirac`; `cacheDirac_`; `diracExtrinsicBlockCached_`/`diracFamily_` decls)
- Modify: `include/nxr/facets.h` (`OperatorsFacet::dirac`)
- Modify: `src/facets.cpp` (cache cases + bodies)
- Modify: `test/test_dirac_operator.cpp` (add facet tests)

- [ ] **Step 1: Extend the `OperatorId` enum** in `include/nxr/compute.h` (line 83-86):

```cpp
enum class OperatorId {
    LaplacianCotan, LaplacianGraph, LaplacianConnection, LaplacianCovariant,
    Dec, MassLumped, MassGalerkin, Gradient3D, Dirac
};
```

- [ ] **Step 2: Add the cache slot + private helper declarations** in `include/nxr/compute.h`. After `cacheGradient3D_;` (line 181):

```cpp
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDirac_;  // extrinsic block E
```

And after the `gradient3DCached_();` declaration (line 204):

```cpp
    // Relative-Dirac extrinsic block E (4V×4V), cached (OperatorId::Dirac).
    const Eigen::SparseMatrix<double>& diracExtrinsicBlockCached_();
    // Assemble L(τ) = (1−τ)(cotanL⊗I₄) + τ·E by value. τ ∈ [0,1] (validated).
    Eigen::SparseMatrix<double> diracFamily_(double tau);
```

- [ ] **Step 3: Add `dirac` to the facet** in `include/nxr/facets.h`. After the `gradient3D()` declaration (line 149):

```cpp
    // dirac(tau): the relative-Dirac family L(τ) = (1−τ)(cotanL⊗I₄) + τ·E, a
    // [4V×4V] real symmetric sparse matrix, returned BY VALUE (τ-dependent blend).
    // τ=0 ⇒ block cotan-Laplacian; τ=1 ⇒ pure relative Dirac E. τ ∈ [0,1].
    Eigen::SparseMatrix<double> dirac(double tau) const;
```

- [ ] **Step 4: Write the failing facet tests** — append to `test/test_dirac_operator.cpp` before `main()`:

```cpp
#include <Eigen/Eigenvalues>

// cotanL ⊗ I₄ built independently (the τ=0 anchor oracle).
static Eigen::SparseMatrix<double> kron4(const Eigen::SparseMatrix<double>& Lc) {
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(Lc.nonZeros()) * 4);
    for (int k = 0; k < Lc.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Lc, k); it; ++it)
            for (int c = 0; c < 4; ++c)
                T.emplace_back(4 * static_cast<int>(it.row()) + c,
                               4 * static_cast<int>(it.col()) + c, it.value());
    Eigen::SparseMatrix<double> K(4 * Lc.rows(), 4 * Lc.cols());
    K.setFromTriplets(T.begin(), T.end());
    return K;
}

static void testDiracFamily() {
    std::cout << "\n=== dirac: family L(τ) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV();

    // HEADLINE anchor: dirac(0) == cotanL ⊗ I₄ byte-for-byte.
    Eigen::SparseMatrix<double> L0 = m.operators().dirac(0.0);
    Eigen::SparseMatrix<double> anchor = kron4(m.operators().laplacian().cotan());
    EXPECT((L0 - anchor).norm() < 1e-12, "dirac(0) == cotanL ⊗ I4 (intrinsic anchor)");

    // Shape + symmetry across the family.
    for (double tau : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        Eigen::SparseMatrix<double> L = m.operators().dirac(tau);
        bool shape = (L.rows() == 4*N && L.cols() == 4*N);
        Eigen::SparseMatrix<double> asym = L - Eigen::SparseMatrix<double>(L.transpose());
        EXPECT(shape && asym.norm() < 1e-10,
               std::string("dirac(") + std::to_string(tau) + ") is [4V×4V] symmetric");
    }

    // dirac(1) == extrinsic block E.
    EXPECT((m.operators().dirac(1.0) - ops::dirac::extrinsicBlock(m)).norm() < 1e-12,
           "dirac(1) == extrinsicBlock");

    // Convex blend identity: dirac(τ) == (1−τ)dirac(0) + τ·dirac(1).
    double tau = 0.4;
    Eigen::SparseMatrix<double> blend =
        (1.0 - tau) * m.operators().dirac(0.0) + tau * m.operators().dirac(1.0);
    EXPECT((m.operators().dirac(tau) - blend).norm() < 1e-12, "dirac(τ) is the convex blend");

    // PSD for τ<1.
    Eigen::MatrixXd dense(m.operators().dirac(0.5));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "dirac(0.5) is PSD");

    // Out-of-range τ throws (Error derives from std::runtime_error — catch the base).
    bool threw = false;
    try { m.operators().dirac(1.5); } catch (const std::exception&) { threw = true; }
    EXPECT(threw, "dirac(τ>1) throws InvalidInput");
}

static void testDiracCache() {
    std::cout << "\n=== dirac: cache lifecycle ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    EXPECT(!m.isOperatorCached(OperatorId::Dirac), "Dirac not cached initially");
    m.operators().dirac(0.5);                       // builds E
    EXPECT(m.isOperatorCached(OperatorId::Dirac), "dirac(τ>0) caches E");
    // A second different-τ call reuses the cached E and still re-blends correctly.
    double tau = 0.9;
    Eigen::SparseMatrix<double> expect =
        (1.0 - tau) * kron4(m.operators().laplacian().cotan())
        + tau * ops::dirac::extrinsicBlock(m);
    EXPECT((m.operators().dirac(tau) - expect).norm() < 1e-12, "re-blend with cached E is correct");
    m.releaseOperator(OperatorId::Dirac);
    EXPECT(!m.isOperatorCached(OperatorId::Dirac), "releaseOperator(Dirac) clears E");
    // dirac(0) does NOT force the extrinsic build (τ=0 ⇒ pure intrinsic).
    m.operators().dirac(0.0);
    EXPECT(!m.isOperatorCached(OperatorId::Dirac), "dirac(0) does not build E");
}
```

Also add these calls into `main()` (before the summary line):

```cpp
    testDiracFamily();
    testDiracCache();
```

- [ ] **Step 5: Run to verify failure** (link/compile error — `dirac` / `Dirac` / helpers undefined):

Run: `bash scripts/build.sh Release 2>&1 | tail -20`
Expected: FAIL — `OperatorId::Dirac` unhandled / `diracFamily_` undefined / `OperatorsFacet::dirac` undefined.

- [ ] **Step 6: Wire the cache cases** in `src/facets.cpp`. In `isOperatorCached` (after line 108):

```cpp
        case OperatorId::Dirac:               return (bool)cacheDirac_;
```

In `releaseOperator` (after line 122):

```cpp
        case OperatorId::Dirac:               cacheDirac_.reset();               break;
```

- [ ] **Step 7: Implement the cache-fill + family helpers** in `src/facets.cpp`, after `gradient3DCached_()` (line 175):

```cpp
const Eigen::SparseMatrix<double>& Manifold::diracExtrinsicBlockCached_() {
    if (!cacheDirac_)
        cacheDirac_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::extrinsicBlock(*this));
    return *cacheDirac_;
}

// L(τ) = (1−τ)(cotanL ⊗ I₄) + τ·E. Builds each term only when its coefficient is
// nonzero — τ=0 never assembles E; τ=1 never builds the intrinsic block.
Eigen::SparseMatrix<double> Manifold::diracFamily_(double tau) {
    if (tau < 0.0 || tau > 1.0)
        throw Error(ErrorCode::InvalidInput, "dirac: tau must be in [0,1]",
                    "Got tau=" + std::to_string(tau) + ".");
    const int N = nV();

    Eigen::SparseMatrix<double> L4;
    if (tau < 1.0) {
        const auto& cotanL = cotanLaplacianCached_();    // sources operatorGeometry()
        std::vector<Eigen::Triplet<double>> T;
        T.reserve(static_cast<size_t>(cotanL.nonZeros()) * 4);
        for (int k = 0; k < cotanL.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(cotanL, k); it; ++it)
                for (int c = 0; c < 4; ++c)
                    T.emplace_back(4 * static_cast<int>(it.row()) + c,
                                   4 * static_cast<int>(it.col()) + c, it.value());
        L4.resize(4 * N, 4 * N);
        L4.setFromTriplets(T.begin(), T.end());
    }
    if (tau == 0.0) { L4.makeCompressed(); return L4; }
    const auto& E = diracExtrinsicBlockCached_();
    if (tau == 1.0) return E;
    Eigen::SparseMatrix<double> L = (1.0 - tau) * L4 + tau * E;
    L.makeCompressed();
    return L;
}
```

- [ ] **Step 8: Implement the facet accessor** in `src/facets.cpp`, next to `OperatorsFacet::gradient3D` (line 306):

```cpp
Eigen::SparseMatrix<double> OperatorsFacet::dirac(double tau) const { return m_.diracFamily_(tau); }
```

- [ ] **Step 9: Add the eigenbasis B-orthonormality test** — append to `test/test_dirac_operator.cpp` before `main()`, and call `testDiracEigenbasis();` in `main()`:

```cpp
static Eigen::SparseMatrix<double> galerkinKron4(Manifold& m) {
    return kron4(m.operators().mass().galerkin());
}

static void testDiracEigenbasis() {
    std::cout << "\n=== dirac: eigenbasis ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    Eigen::SparseMatrix<double> L = m.operators().dirac(0.5);
    Eigen::SparseMatrix<double> B = galerkinKron4(m);   // M_Galerkin ⊗ I₄
    const int k = 8;
    solve::EigenResult er = solve::solveEigenmodes(L, B, k);
    Eigen::MatrixXd Phi = solve::normalizeEigenmodes(er.eigenvectors, B);

    Eigen::MatrixXd gram = Phi.transpose() * (B * Phi);
    EXPECT((gram - Eigen::MatrixXd::Identity(gram.rows(), gram.cols())).cwiseAbs().maxCoeff() < 1e-9,
           "ΦᵀBΦ ≈ I (B-orthonormal eigenbasis)");
    // eigenvalues real & ascending (solver contract) and finite.
    EXPECT(er.eigenvalues.allFinite() && er.eigenvalues(0) <= er.eigenvalues(k-1),
           "eigenvalues finite & ascending");

    // Quaternionic structure: L(τ) commutes with right-ℍ-multiplication, so each
    // distinct eigenvalue has real multiplicity divisible by 4. With k=8 the
    // spectrum is two 4-fold multiplets — check each group is (near-)constant,
    // tolerance loose (shift-invert + Galerkin mass perturb the exact degeneracy).
    auto groupSpread = [&](int g) {
        double lo = er.eigenvalues.segment(4*g, 4).minCoeff();
        double hi = er.eigenvalues.segment(4*g, 4).maxCoeff();
        return (hi - lo) / (1.0 + std::abs(hi));
    };
    EXPECT(groupSpread(0) < 1e-4 && groupSpread(1) < 1e-4,
           "eigenvalues form 4-fold quaternionic multiplets");
}
```

(Confirm the `solve::` namespace qualifiers compile — `solveEigenmodes`/`normalizeEigenmodes`/`EigenResult` live in `nxr::manifold::solve`; `grep -n "normalizeEigenmodes" include/nxr/compute.h` to verify the exact namespace and adjust the qualifier if needed.)

- [ ] **Step 10: Run all native tests:**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_dirac_operator`
Expected: `ALL PASSED`.

- [ ] **Step 11: Regression sweep** (the facet/operators canaries must stay green):

Run: `./build/Release/test_operators_facet && ./build/Release/test_facets && ./build/Release/test_covariant_differential && ./build/Release/test_intrinsic_delaunay`
Expected: all `ALL PASSED` / `OK`.

- [ ] **Step 12: Commit:**

```bash
git add include/nxr/compute.h include/nxr/facets.h src/facets.cpp test/test_dirac_operator.cpp
git commit -m "feat(dirac): operators().dirac(τ) family + OperatorId::Dirac cache"
```

---

### Task 3: MEX surface `operators(h,'dirac',τ)`

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp` (`cmdOperators` + help string)
- Create: `bindings/mex/test/test_dirac_operator.m`

- [ ] **Step 1: Write the failing MATLAB test** — create `bindings/mex/test/test_dirac_operator.m`:

```matlab
function test_dirac_operator
    % Icosphere fixture (same as native tests).
    t = (1 + sqrt(5)) / 2;
    V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; ...
          0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
    F = [0 11 5; 0 5 1; 0 1 7; 0 7 10; 0 10 11; 1 5 9; 5 11 4; 11 10 2; ...
         10 7 6; 7 1 8; 3 9 4; 3 4 2; 3 2 6; 3 6 8; 3 8 9; 4 9 5; ...
         2 4 11; 6 2 10; 8 6 7; 9 8 1] + 1;   % MATLAB 1-based
    h = nxr_compute('create', V, F);
    nV = size(V,1);

    % τ=0 anchor: dirac(0) == kron(I4, cotanL).
    L0 = nxr_compute('operators', h, 'dirac', 0);
    Lc = nxr_compute('operators', h, 'laplacian', 'cotan');
    anchor = kron(speye(4), Lc);
    assert(norm(L0 - anchor, 'fro') < 1e-10, 'dirac(0) != cotanL kron I4');
    assert(isequal(size(L0), [4*nV, 4*nV]), 'dirac size wrong');

    % Generalized eigenproblem with Galerkin mass ⊗ I4 gives real eigenvalues.
    L = nxr_compute('operators', h, 'dirac', 0.5);
    Mg = nxr_compute('operators', h, 'mass', 'galerkin');
    B = kron(speye(4), Mg);
    assert(norm(L - L', 'fro') < 1e-9, 'dirac(0.5) not symmetric');
    d = eigs(L, B, 8, 'smallestabs');
    assert(all(abs(imag(d)) < 1e-8), 'eigenvalues not real');

    % τ out of range errors.
    threw = false;
    try, nxr_compute('operators', h, 'dirac', 2.0); catch, threw = true; end
    assert(threw, 'dirac(2.0) did not error');

    nxr_compute('destroy', h);
    disp('test_dirac_operator: ALL PASSED');
end
```

- [ ] **Step 2: Run to verify failure:**

Build the MEX (`bash scripts/build.sh Release`), then in MATLAB run `test_dirac_operator` via the MATLAB MCP `run_matlab_file`.
Expected: error — `operators: family must be laplacian|mass|hodge|dec|gradient3D` (no `dirac`).

- [ ] **Step 3: Add the `dirac` branch** in `cmdOperators` (`bindings/mex/src/nxr_compute_mex.cpp`), after the `gradient3D` branch (line 1823-1824):

```cpp
    } else if (family == "dirac") {
        if (nrhs < 4 || !mxIsNumeric(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 1)
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators dirac: expected a scalar tau, "
                "nxr_compute('operators', h, 'dirac', tau).");
        double tau = mxGetScalar(prhs[3]);
        plhs[0] = eigenSparseToMx(m.operators().dirac(tau));   // [4V×4V], caches E on the handle
```

- [ ] **Step 4: Update the family error message** in the trailing `else` (line 1826-1828):

```cpp
    } else {
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators: family must be laplacian|mass|hodge|dec|gradient3D|dirac.");
    }
```

- [ ] **Step 5: Update the help/usage comment** above `cmdOperators` (line 1758-1772). Add a line documenting the dirac form:

```cpp
//   nxr_compute('operators', h, 'dirac', tau)   % [4V×4V] relative-Dirac family,
//                                               % tau in [0,1] (0=cotan⊗I4, 1=D_N)
```

(If there is a separate dispatch help string elsewhere — `grep -n "gradient3D" bindings/mex/src/nxr_compute_mex.cpp` — add `dirac` there too for consistency.)

- [ ] **Step 6: Rebuild and run the MATLAB test:**

Run: `bash scripts/build.sh Release` then MATLAB MCP `run_matlab_file` on `bindings/mex/test/test_dirac_operator.m`.
Expected: `test_dirac_operator: ALL PASSED`.

- [ ] **Step 7: Commit:**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_dirac_operator.m
git commit -m "feat(mex): operators(h,'dirac',τ) returns the relative-Dirac family"
```

---

### Task 4: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `dirac` to the `operators` command row.** In the MEX command table, extend the `operators` row's family list to mention `dirac` (the relative-Dirac family `L(τ)`, real `4V×4V`, τ∈[0,1]). Find it with `grep -n "operators', h, family" CLAUDE.md` (the row beginning `` `nxr_compute('operators', h, family[, subtype])` ``) and append, in the same terse style:

```
; `dirac` (the relative-Dirac family L(τ) = (1−τ)cotan⊗I₄ + τ·D_N, real `[4V×4V]`, `τ∈[0,1]`; extrinsic, geometry-only, Liu/Jacobson/Crane SGP 2017)
```

- [ ] **Step 2: Add a short operators-surface note** near the covariant/operators discussion (the paragraph describing `Gauge.operators` / `covariantLaplacian`), one or two sentences: the `dirac(τ)` family is the curvature-aware extrinsic operator; `τ=0` byte-matches `cotanL⊗I₄`, `τ=1` is the pure relative Dirac `E = Dᵀ⋆_F D` from the Gauss map; quaternion storage is vertex-interleaved `[w,x,y,z]` at index `4v+c`; eigenbasis via the existing `solveEigenmodes` against `M_Galerkin⊗I₄`. Reference `docs/superpowers/specs/2026-06-10-extrinsic-dirac-operator-design.md`.

- [ ] **Step 3: Commit:**

```bash
git add CLAUDE.md
git commit -m "docs(dirac): document operators().dirac(τ) relative-Dirac family"
```

---

## Final verification (after all tasks)

- [ ] Full native sweep:

```bash
./build/Release/test_dirac_operator
./build/Release/test_operators_facet
./build/Release/test_facets
./build/Release/test_covariant_differential
./build/Release/test_intrinsic_delaunay
./build/Release/test_connection_laplacian
```

All must print `ALL PASSED` / `OK`.

- [ ] MEX: `bindings/mex/test/test_dirac_operator.m` passes via MATLAB MCP, and the pre-existing `bindings/mex/test/test_bundle.m` still passes (no regression in the `operators` command).

- [ ] Dispatch a final whole-branch code review (subagent) covering the four commits before finishing the branch.
