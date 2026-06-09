# Covariant (3-frame) Laplacian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `Gauge.operators.covariantLaplacian` — a 3N×3N real symmetric sparse Laplacian on the full 3-frame `[a;b;c] = [Re z; Im z; normal]`, in two couplings (`product` block-diagonal, `ambient` frame-coupled).

**Architecture:** A pure linear-algebra library function `assembleCovariantLaplacian(coupling, K, gaugeGrid, cotanL)` builds the 3N×3N real matrix from pieces the caller already has (the gauge connection Laplacian `K`, the realized gauge grid, the cotan Laplacian). The MEX `buildGaugeOperators` computes those pieces and attaches `covariantLaplacian`, gated by the existing `operators` flag, with a `coupling` option.

**Tech Stack:** C++17, Eigen, MATLAB MEX. Build `bash scripts/build.sh Release`. Native binaries in `build/`. MATLAB tests via MATLAB MCP.

**Spec:** `docs/superpowers/specs/2026-06-08-vector-laplacian-3d-design.md`

**Layout (hard rule):** component-major block `[a; b; c]`, each an N-block. Rows/cols `0..N−1` = a (Re z), `N..2N−1` = b (Im z), `2N..3N−1` = c (normal).

---

## Task 1: `assembleCovariantLaplacian` library function

**Files:** `include/nxr/compute.h`, `src/covariant_laplacian.cpp` (create), `test/test_geometry_bundle.cpp`, `CMakeLists.txt`

- [ ] **Step 1: Declare in `compute.h`** — in `namespace nxr::manifold::ops::laplacian::connection` (where the other connection-Laplacian declarations live), add:

```cpp
// Coupling for the 3-frame covariant Laplacian (see covariant-laplacian spec).
enum class CovariantCoupling {
    Product,   // blkdiag(real-expand(K), cotanL) — tangent ⊕ normal decoupled
    Ambient    // frame-conjugate of kron(I3, cotanL): L3[i,j] = cotanL[i,j]·(Fiᵀ Fj)
};

// 3N×3N real symmetric Laplacian on the 3-frame [a;b;c] = [Re z; Im z; normal],
// component-major block layout. Pure linear algebra over pieces the caller holds:
//   K        = gauge connection Laplacian (V×V complex Hermitian) — used by Product
//              (and equals the Ambient tangent block for the matching gauge)
//   gaugeGrid= realized gauge frame, V×3 complex c = e1 + i·e2 (normal = Re×Im) — used by Ambient
//   cotanL   = scalar cotan Laplacian (V×V real)
// See spec §3–4. Symmetric PSD.
Eigen::SparseMatrix<double> assembleCovariantLaplacian(
    CovariantCoupling coupling,
    const Eigen::SparseMatrix<std::complex<double>>& K,
    const Eigen::MatrixXcd& gaugeGrid,
    const Eigen::SparseMatrix<double>& cotanL);
```

- [ ] **Step 2: Failing native test** — append to `test/test_geometry_bundle.cpp`, call from `main()`. It assembles the real pieces from the icosphere and checks the exact identities.

```cpp
static void testCovariantLaplacian() {
    std::cout << "\n=== covariantLaplacian ===\n";
    namespace cl = nxr::manifold::ops::laplacian::connection;
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = 12;

    auto ops = nxr::manifold::ops::assembleManifoldOperators(m);   // cotanLaplacian
    Eigen::SparseMatrix<double> L = ops.cotanLaplacian;
    Eigen::MatrixXcd grid = nxr::manifold::geometry::vertexGrid(m); // V×3 complex (LC frame)

    cl::ConnectionLaplacianOptions o; o.domain = cl::ConnectionDomain::Vertex;
    o.nSym = 1; o.format = cl::ConnectionLaplacianFormat::Complex;
    auto K = cl::assembleConnectionLaplacian(m, o).K_complex;       // V×V complex

    auto Lp = cl::assembleCovariantLaplacian(cl::CovariantCoupling::Product,  K, grid, L);
    auto La = cl::assembleCovariantLaplacian(cl::CovariantCoupling::Ambient,  K, grid, L);

    EXPECT(Lp.rows()==3*N && Lp.cols()==3*N, "covariantLaplacian is 3N×3N");

    // symmetric
    auto symErr = [](const Eigen::SparseMatrix<double>& A){
        return (A - Eigen::SparseMatrix<double>(A.transpose())).norm(); };
    EXPECT(symErr(Lp) < 1e-9, "product symmetric");
    EXPECT(symErr(La) < 1e-9, "ambient symmetric");

    // product == blkdiag(real-expand(K), cotanL), exact.
    // real-expand(K): aa=ReK, ab=-ImK, ba=ImK, bb=ReK; cc=L.
    {
        Eigen::SparseMatrix<double> ReK = K.real().cast<double>();   // see note: build from K
        Eigen::SparseMatrix<double> ImK = K.imag().cast<double>();
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(3*N, 3*N);
        D.block(0,0,N,N)   =  Eigen::MatrixXd(ReK);
        D.block(0,N,N,N)   = -Eigen::MatrixXd(ImK);
        D.block(N,0,N,N)   =  Eigen::MatrixXd(ImK);
        D.block(N,N,N,N)   =  Eigen::MatrixXd(ReK);
        D.block(2*N,2*N,N,N)=  Eigen::MatrixXd(L);
        double err = (Eigen::MatrixXd(Lp) - D).cwiseAbs().maxCoeff();
        EXPECT(err < 1e-9, "product == blkdiag(real-expand(K), cotanL)");
    }

    // ambient world-form identity: Fbd · La · Fbdᵀ == kron(I3, L), exact.
    // Fbd is 3N×3N, component-major: block (p-comp, q-comp) is diag over v of F_v[p][q],
    // F_v columns = e1=Re(grid_v), e2=Im(grid_v), n=e1×e2.
    {
        Eigen::MatrixXd Fbd = Eigen::MatrixXd::Zero(3*N, 3*N);
        for (int v=0; v<N; ++v) {
            Eigen::Vector3d e1 = grid.row(v).real(), e2 = grid.row(v).imag();
            Eigen::Vector3d nrm = e1.cross(e2);
            Eigen::Matrix3d Fv; Fv.col(0)=e1; Fv.col(1)=e2; Fv.col(2)=nrm;
            for (int p=0;p<3;++p) for (int q=0;q<3;++q) Fbd(p*N+v, q*N+v) = Fv(p,q);
        }
        Eigen::MatrixXd kronI3L = Eigen::MatrixXd::Zero(3*N,3*N);
        kronI3L.block(0,0,N,N)     = Eigen::MatrixXd(L);
        kronI3L.block(N,N,N,N)     = Eigen::MatrixXd(L);
        kronI3L.block(2*N,2*N,N,N) = Eigen::MatrixXd(L);
        double err = (Fbd * Eigen::MatrixXd(La) * Fbd.transpose() - kronI3L).cwiseAbs().maxCoeff();
        EXPECT(err < 1e-9, "ambient world-form == kron(I3, cotanL)");
    }

    // ambient ≠ product on the curved mesh
    EXPECT((Eigen::MatrixXd(La) - Eigen::MatrixXd(Lp)).cwiseAbs().maxCoeff() > 1e-6,
           "ambient differs from product (curvature coupling)");

    // PSD
    auto minEig = [](const Eigen::SparseMatrix<double>& A){
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(A));
        return es.eigenvalues().minCoeff(); };
    EXPECT(minEig(Lp) > -1e-9, "product PSD");
    EXPECT(minEig(La) > -1e-9, "ambient PSD");
}
```
NOTE for the implementer: `K.real()`/`K.imag()` on a complex sparse may need `K.real().eval()` or building Re/Im by iterating triplets — use whatever compiles cleanly to get the real and imaginary sparse parts (mirror how `connection_laplacian.cpp` splits a complex sparse into real/imag if it does). Ensure `<Eigen/Eigenvalues>` is included (Task added it earlier).

- [ ] **Step 3: Build to confirm link failure** — `bash scripts/build.sh Release 2>&1 | grep -E "error:|assembleCovariantLaplacian|undefined"` → undefined-reference.

- [ ] **Step 4: Implement `src/covariant_laplacian.cpp`**

```cpp
#include "nxr/compute.h"
#include <vector>
#include <complex>

namespace nxr::manifold::ops::laplacian::connection {

Eigen::SparseMatrix<double> assembleCovariantLaplacian(
    CovariantCoupling coupling,
    const Eigen::SparseMatrix<std::complex<double>>& K,
    const Eigen::MatrixXcd& gaugeGrid,
    const Eigen::SparseMatrix<double>& cotanL) {

    const int N = static_cast<int>(cotanL.rows());
    std::vector<Eigen::Triplet<double>> T;

    if (coupling == CovariantCoupling::Product) {
        // tangent 2N block = [[ReK, -ImK],[ImK, ReK]]
        for (int k = 0; k < K.outerSize(); ++k)
            for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(K, k); it; ++it) {
                int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
                double re = it.value().real(), im = it.value().imag();
                T.emplace_back(i,       j,       re);   // aa
                T.emplace_back(i,       N + j,  -im);   // ab
                T.emplace_back(N + i,   j,       im);   // ba
                T.emplace_back(N + i,   N + j,   re);   // bb
            }
        // normal block cc = cotanL
        for (int k = 0; k < cotanL.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(cotanL, k); it; ++it)
                T.emplace_back(2*N + static_cast<int>(it.row()),
                               2*N + static_cast<int>(it.col()), it.value());
    } else { // Ambient: L3[i,j] (3×3) = cotanL[i,j] · (Fiᵀ Fj)
        // precompute per-vertex 3×3 frame F_v (columns e1, e2, n)
        std::vector<Eigen::Matrix3d> Fv(N);
        for (int v = 0; v < N; ++v) {
            Eigen::Vector3d e1 = gaugeGrid.row(v).real();
            Eigen::Vector3d e2 = gaugeGrid.row(v).imag();
            Eigen::Vector3d nrm = e1.cross(e2);
            Fv[v].col(0) = e1; Fv[v].col(1) = e2; Fv[v].col(2) = nrm;
        }
        for (int k = 0; k < cotanL.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(cotanL, k); it; ++it) {
                int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
                double w = it.value();
                Eigen::Matrix3d M = Fv[i].transpose() * Fv[j];   // (Fiᵀ Fj)
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q)
                        T.emplace_back(p*N + i, q*N + j, w * M(p,q));
            }
    }

    Eigen::SparseMatrix<double> L3(3*N, 3*N);
    L3.setFromTriplets(T.begin(), T.end());
    L3.makeCompressed();
    return L3;
}

} // namespace
```
Add `src/covariant_laplacian.cpp` to the `add_library(nxr_compute ...)` source list in CMakeLists.txt.

- [ ] **Step 5: Build + run** — `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_geometry_bundle` → ALL PASSED incl. the new covariantLaplacian assertions. (binaries in `build/`.) If "ambient world-form" fails, the frame column order or the (p,q) block placement is off — recheck `M = Fiᵀ Fj` and the `p*N+i` indexing.

- [ ] **Step 6: Commit**
```bash
git add include/nxr/compute.h src/covariant_laplacian.cpp test/test_geometry_bundle.cpp CMakeLists.txt
git commit -m "feat(ops): add assembleCovariantLaplacian (3-frame product/ambient Laplacian)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 1)
Confirm in compute.h: the `connection` namespace path, `ConnectionLaplacianOptions`/`ConnectionDomain`/`ConnectionLaplacianFormat`/`assembleConnectionLaplacian` (returns a struct with `.K_complex`), `assembleManifoldOperators` (`.cotanLaplacian`), and `geometry::vertexGrid`. Confirm how to extract real/imag sparse parts of a complex sparse (check `connection_laplacian.cpp`). Follow TDD. Report Status, build result, the covariantLaplacian PASS lines, files, commit SHA.

---

## Task 2: wire `covariantLaplacian` + `coupling` into the MEX gauge operators

**Files:** `bindings/mex/src/nxr_compute_mex.cpp`, `bindings/mex/test/test_operators.m`

- [ ] **Step 1: Add a `coupling` parser** near `readOperatorsFlag`:
```cpp
static nxr::manifold::ops::laplacian::connection::CovariantCoupling
parseCoupling(const mxArray* opts) {
    namespace cl = nxr::manifold::ops::laplacian::connection;
    if (opts && mxIsStruct(opts)) {
        const mxArray* f = mxGetField(opts, 0, "coupling");
        if (f && mxIsChar(f)) {
            std::string s = getStringArg(f);
            if (s == "product") return cl::CovariantCoupling::Product;
            if (s == "ambient") return cl::CovariantCoupling::Ambient;
            throw std::invalid_argument("coupling must be 'product' or 'ambient'");
        }
    }
    return cl::CovariantCoupling::Ambient;   // default
}
```

- [ ] **Step 2: Extend `buildGaugeOperators`** to also assemble `covariantLaplacian`. It already computes `K` (the gauge connection Laplacian, `Eigen::SparseMatrix<std::complex<double>>`). Add: the cotan Laplacian (`ensureOps(h).cotanLaplacian`), the realized gauge grid, the coupling, then call the library fn and add the field. The struct becomes `{laplacian, covariantLaplacian}`.

Realized gauge grid: `vertexGrid(m)` for euclidean/levi-civita; for trivial, multiply each row by the per-vertex rotation from `integrateTrivialGaugeRotations` (the same `gr.vertex` used for `Gauge.vertex.rotation`). Compute it once:
```cpp
    // ... after K is obtained and (for trivial) sing parsed ...
    Eigen::MatrixXcd grid = nxr::manifold::geometry::vertexGrid(*h.ctx);
    if (type == "trivial") {
        auto gr = nxr::manifold::connection::integrateTrivialGaugeRotations(*h.ctx, ensureDec(h), *h.cache, sing);
        for (int v = 0; v < grid.rows(); ++v) grid.row(v) *= gr.vertex(v);  // rotation .* grid
    }
    const auto& cotanL = ensureOps(h).cotanLaplacian;
    namespace cl = nxr::manifold::ops::laplacian::connection;
    Eigen::SparseMatrix<double> covL = cl::assembleCovariantLaplacian(coupling, K, grid, cotanL);

    const char* f[] = {"laplacian","covariantLaplacian"};
    mxArray* s = mxCreateStructMatrix(1,1,2,f);
    mxSetField(s,0,"laplacian", eigenComplexSparseToMx(K));
    mxSetField(s,0,"covariantLaplacian", eigenSparseToMx(covL));
    return s;
```
`buildGaugeOperators` needs the `coupling` — change its signature to
`buildGaugeOperators(ContextHolder& h, const std::string& type, const mxArray* opts, CovariantCoupling coupling)` and pass `parseCoupling(opts)` from the caller (`buildGaugeStruct`'s operators-attach block). NOTE: for `trivial`, the `sing` map is already parsed in this function for `K`; reuse it for the grid rotation (don't re-parse). If `integrateTrivialGaugeRotations` was not already called here, call it once and reuse.

(If `buildGaugeStruct` calls `buildGaugeOperators(h, type, opts)`, update that call to pass `parseCoupling(opts)`.)

- [ ] **Step 3: Build** — `bash scripts/build.sh Release 2>&1 | tail -10` (clean + mexmaca64). Fix compile errors (namespace paths, `vertexGrid` include, signature update at the call site).

- [ ] **Step 4: Extend `bindings/mex/test/test_operators.m`** — add, after the existing gauge-operators block (where `Gl`, `K`/`Kl` are available):

```matlab
% covariant (3-frame) Laplacian
nVc = nV;
Lc3 = Gl.operators.covariantLaplacian;            % default 'ambient', levi-civita
assert(issparse(Lc3) && isequal(size(Lc3),[3*nVc 3*nVc]), 'covariantLaplacian 3N×3N sparse');
assert(nnz(Lc3 - Lc3') == 0 || norm(Lc3 - Lc3','fro') < 1e-9, 'covariantLaplacian symmetric');

% product == blkdiag(real-expand(K), cotanL), using already-exposed operators
Gp = nxr_compute('gauge', h, 'levi-civita', struct('operators',true,'coupling','product'));
Lp3 = Gp.operators.covariantLaplacian;
Kc  = Gl.operators.laplacian;                      % V×V complex (connection L)
Lcot= Gg.operators.laplacian;                      % V×V cotan (from geometry ops; Gg from earlier in test)
ReK = real(Kc); ImK = imag(Kc);
D = [ ReK, -ImK, sparse(nVc,nVc);
      ImK,  ReK, sparse(nVc,nVc);
      sparse(nVc,nVc), sparse(nVc,nVc), Lcot ];
assert(norm(Lp3 - D, 'fro') < 1e-9, 'product == blkdiag(real-expand(K), cotanL)');

% ambient world-form == kron(I3, cotanL) via the realized frame
c = Gl.vertex.rotation .* G.vertex.grid;           % realized LC frame (rotation==1); G from earlier
e1 = real(c); e2 = imag(c); nrm = cross(e1,e2,2);
% block-diag frame Fbd (3N×3N), component-major [a;b;c]
Z = sparse(nVc,nVc);
Fbd = [ spdiags(e1(:,1),0,nVc,nVc), spdiags(e2(:,1),0,nVc,nVc), spdiags(nrm(:,1),0,nVc,nVc);
        spdiags(e1(:,2),0,nVc,nVc), spdiags(e2(:,2),0,nVc,nVc), spdiags(nrm(:,2),0,nVc,nVc);
        spdiags(e1(:,3),0,nVc,nVc), spdiags(e2(:,3),0,nVc,nVc), spdiags(nrm(:,3),0,nVc,nVc) ];
kronI3L = blkdiag(Lcot, Lcot, Lcot);
assert(norm(Fbd*Lc3*Fbd' - kronI3L, 'fro') < 1e-9, 'ambient world-form == kron(I3, cotanL)');
assert(norm(Lc3 - Lp3,'fro') > 1e-6, 'ambient differs from product');
```
IMPORTANT: ensure the variables `Gg` (geometry-with-ops), `G` (plain geometry), `Gl` (levi-civita gauge with ops) exist in scope where you insert this — read the current test_operators.m and reuse/fetch as needed (fetch `G = nxr_compute('geometry', h)` and `Gg = nxr_compute('geometry', h, struct('operators',true))` if not already present). The Fbd block layout MUST match the C++ component-major `[a;b;c]` layout.

- [ ] **Step 5: Build + run the FULL MATLAB + native suite**
- `./build/test_geometry_bundle` (native) → ALL PASSED.
- Via MATLAB MCP run test_operators.m, test_gauge.m, test_bundle.m → ALL TESTS PASSED. If the world-form assertion fails, the MATLAB `Fbd` layout doesn't match the C++ `[a;b;c]` component-major layout — align them. If MATLAB MCP unavailable, report DONE_WITH_CONCERNS (build + native only).

- [ ] **Step 6: Commit**
```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_operators.m
git commit -m "feat(mex): expose Gauge.operators.covariantLaplacian + coupling option

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Before you begin (Task 2)
Read the current `buildGaugeOperators` and its call site in `buildGaugeStruct`, and confirm `integrateTrivialGaugeRotations`'s signature + that `vertexGrid` is reachable. Default coupling is `ambient`. Caching of the final `covL` is optional (its inputs K/cotan/grid are already cached); skip unless trivial to add. Report Status, build result, the MATLAB test outputs (verbatim or unavailable), files, commit SHA.

---

## Self-Review

| Spec item | Task |
|---|---|
| `assembleCovariantLaplacian`, product + ambient | Task 1 |
| 3N×3N real, frame coords `[a;b;c]` component-major | Task 1 (layout), Task 2 (Fbd matches) |
| ambient world-form == kron(I3,cotanL) (exact) | Task 1 native + Task 2 MATLAB |
| product == blkdiag(real-expand(K), cotanL) (exact) | Task 1 native + Task 2 MATLAB |
| `Gauge.operators.covariantLaplacian` + `coupling` opt (default ambient) | Task 2 |
| tangent connection follows gauge (LC/trivial via K + realized grid) | Task 2 |

**Placeholders:** none. **Consistency:** `CovariantCoupling`/`assembleCovariantLaplacian` declared T1, used T2; the C++ `[a;b;c]` layout and the MATLAB `Fbd` layout are both component-major and must match (called out in both task tests).
