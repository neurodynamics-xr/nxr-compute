# Operators Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an opt-in `.operators` sub-struct to `Topology`/`Geometry`/`Gauge` holding live native MATLAB sparse operators (graph/cotan/connection Laplacians, DEC incidence, mass, Hodge stars), assembled on demand and cached on the handle.

**Architecture:** New library `graphLaplacian` (`d0ᵀd0`); a native complex-sparse marshaler; three `buildXxxOperators` MEX helpers that pull from the existing `ensureOps`/`ensureDec`/`clCache` caches (plus new graph/Galerkin slots); an `operators` flag threaded through `topology`/`geometry`/`gauge`/`bundle`. Light by default — byte-identical to today when the flag is absent.

**Tech Stack:** C++17, Eigen, geometry-central, MATLAB MEX. Build `bash scripts/build.sh Release`. Native binaries in `build/`. MATLAB tests via MATLAB MCP `run_matlab_file`.

**Spec:** `docs/superpowers/specs/2026-06-08-operators-surface-design.md`

---

## File Map

| File | Change |
|---|---|
| `include/nxr/compute.h` | declare `ops::graphLaplacian` |
| `src/graph_laplacian.cpp` | **create** — `graphLaplacian = d0ᵀd0` |
| `test/test_geometry_bundle.cpp` | add `testGraphLaplacian` |
| `CMakeLists.txt` | add `src/graph_laplacian.cpp` to library |
| `bindings/mex/src/marshal.h` | add `eigenComplexSparseToMx` |
| `bindings/mex/src/nxr_compute_mex.cpp` | ContextHolder slots; `buildTopology/Geometry/GaugeOperators`; `operators` flag in build/cmd functions |
| `bindings/mex/test/test_operators.m` | **create** |

Context to read first: `src/topology.cpp` (library pattern), the existing `buildTopologyStruct`/`buildGeometryStruct`/`buildGaugeStruct` and `cmdBundle` in `nxr_compute_mex.cpp`, the `ContextHolder` struct + `ensureOps`/`ensureDec`/`clCache`, and `marshal.h`'s `eigenSparseToMx`.

---

## Task 1: `graphLaplacian` library function

**Files:** `include/nxr/compute.h`, `src/graph_laplacian.cpp` (create), `test/test_geometry_bundle.cpp`, `CMakeLists.txt`

- [ ] **Step 1: Declare in `compute.h`** — in `namespace nxr::manifold::ops`, near the operator-assembly declarations:

```cpp
// Graph (combinatorial) Laplacian L = D − A, built as d0ᵀ d0 from the
// metric-free exterior derivative. Pure topology — independent of vertex
// positions. Symmetric PSD; diagonal = vertex degree, off-diagonal = −(#edges).
Eigen::SparseMatrix<double> graphLaplacian(Manifold& m);
```

- [ ] **Step 2: Failing native test** — append to `test/test_geometry_bundle.cpp` and call from `main()`:

```cpp
static void testGraphLaplacian() {
    std::cout << "\n=== graphLaplacian ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    Eigen::SparseMatrix<double> L = nxr::manifold::ops::graphLaplacian(m);
    EXPECT(L.rows() == 12 && L.cols() == 12, "graphLaplacian is V×V");

    // symmetric
    Eigen::SparseMatrix<double> asym = L - Eigen::SparseMatrix<double>(L.transpose());
    EXPECT(asym.norm() < 1e-12, "graphLaplacian symmetric");

    // zero row sums (L * 1 = 0)
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(12);
    EXPECT((L * ones).cwiseAbs().maxCoeff() < 1e-12, "zero row sums");

    // diagonal = degree (icosahedron: every vertex has degree 5), off-diag ∈ {0,-1}
    bool degOk = true, offOk = true;
    for (int i = 0; i < 12; ++i) if (std::abs(L.coeff(i,i) - 5.0) > 1e-12) degOk = false;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) if (i!=j) {
        double v = L.coeff(i,j); if (v != 0.0 && std::abs(v + 1.0) > 1e-12) offOk = false;
    }
    EXPECT(degOk, "diagonal == degree (5 on icosahedron)");
    EXPECT(offOk, "off-diagonal ∈ {0, −1}");
}
```

- [ ] **Step 3: Register test exe is already done** (test_geometry_bundle exists). Build to confirm link failure:
Run: `bash scripts/build.sh Release 2>&1 | grep -E "error:|graphLaplacian|undefined"` → undefined-reference for `graphLaplacian`.

- [ ] **Step 4: Implement `src/graph_laplacian.cpp`**:

```cpp
#include "nxr/compute.h"

namespace nxr::manifold::ops {

// L = d0ᵀ d0. d0 is the signed vertex→edge incidence (exterior derivative on
// 0-forms): each edge row has +1 at head, −1 at tail. d0ᵀd0 then gives
// degree on the diagonal and −1 between adjacent vertices = the graph Laplacian.
Eigen::SparseMatrix<double> graphLaplacian(Manifold& m) {
    const Eigen::SparseMatrix<double>& D0 = d0(m);   // passthrough accessor (compute.h)
    Eigen::SparseMatrix<double> L = D0.transpose() * D0;
    L.makeCompressed();
    return L;
}

} // namespace nxr::manifold::ops
```
Verify `d0(Manifold&)` passthrough exists in compute.h (`const Eigen::SparseMatrix<double>& d0(Manifold& m);`). If not, use `assembleDECOperators(m).d0`. Add `src/graph_laplacian.cpp` to the `add_library(nxr_compute ...)` source list in CMakeLists.txt.

- [ ] **Step 5: Build + run** — `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_geometry_bundle` → ALL PASSED incl. the 5 graphLaplacian assertions. (Native test binaries are in `build/`, not `build/Release/`.)

- [ ] **Step 6: Commit**
```bash
git add include/nxr/compute.h src/graph_laplacian.cpp test/test_geometry_bundle.cpp CMakeLists.txt
git commit -m "feat(ops): add graphLaplacian (combinatorial L = d0ᵀd0)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `eigenComplexSparseToMx` marshaler

**Files:** `bindings/mex/src/marshal.h`

- [ ] **Step 1: Add the helper** after `eigenSparseToMx` (read that function first to mirror its CSC layout):

```cpp
// Eigen complex sparse → native MATLAB complex sparse (interleaved, R2018a+).
inline mxArray* eigenComplexSparseToMx(const Eigen::SparseMatrix<std::complex<double>>& src) {
    Eigen::SparseMatrix<std::complex<double>> m = src;  // ensure CSC + compressed
    m.makeCompressed();
    const mwSize rows = static_cast<mwSize>(m.rows());
    const mwSize cols = static_cast<mwSize>(m.cols());
    const mwSize nnz  = static_cast<mwSize>(m.nonZeros());
    mxArray* arr = mxCreateSparse(rows, cols, nnz, mxCOMPLEX);
    mwIndex* ir = mxGetIr(arr);
    mwIndex* jc = mxGetJc(arr);
    mxComplexDouble* pr = mxGetComplexDoubles(arr);
    const int* outer = m.outerIndexPtr();
    const int* inner = m.innerIndexPtr();
    const std::complex<double>* vals = m.valuePtr();
    for (mwSize c = 0; c <= cols; ++c) jc[c] = static_cast<mwIndex>(outer[c]);
    for (mwSize k = 0; k < nnz; ++k) {
        ir[k] = static_cast<mwIndex>(inner[k]);
        pr[k].real = vals[k].real();
        pr[k].imag = vals[k].imag();
    }
    return arr;
}
```

- [ ] **Step 2: Build** (header-only; unused until Task 3) — `bash scripts/build.sh Release 2>&1 | grep -E "error:|eigenComplexSparseToMx"` → no errors. Confirm `nxr_compute.mexmaca64` builds.

- [ ] **Step 3: Commit**
```bash
git add bindings/mex/src/marshal.h
git commit -m "feat(mex): add eigenComplexSparseToMx (native complex sparse marshaler)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: operator builders + `operators` flag wiring + MATLAB test

**Files:** `bindings/mex/src/nxr_compute_mex.cpp`, `bindings/mex/test/test_operators.m` (create)

- [ ] **Step 1: Add ContextHolder cache slots** — in the `ContextHolder` struct add:
```cpp
    std::unique_ptr<Eigen::SparseMatrix<double>> graphLap;       // lazy: graphLaplacian
    std::unique_ptr<Eigen::SparseMatrix<double>> massGalerkin;   // lazy: Galerkin mass
```

- [ ] **Step 2: Add an `operators`-flag reader** (near the other small helpers, before the build functions):
```cpp
static bool readOperatorsFlag(const mxArray* opts) {
    if (!opts || !mxIsStruct(opts)) return false;
    const mxArray* f = mxGetField(opts, 0, "operators");
    return f && !mxIsEmpty(f) && (mxIsLogicalScalarTrue(f) ||
           (mxIsNumeric(f) && mxGetScalar(f) != 0.0));
}
```

- [ ] **Step 3: Add the three operator builders** (place above the `buildXxxStruct` functions). Read `ensureOps`/`ensureDec` return types and `ManifoldOperators`/`DECOperators` field names to confirm (`cotanLaplacian`, `hodge0/1/2`, `hodge1Inverse`, `d0`, `d1`). `mass.lumped` uses `dec.hodge0` (= lumped barycentric mass, per CLAUDE.md).

```cpp
mxArray* buildTopologyOperators(ContextHolder& h) {
    if (!h.graphLap)
        h.graphLap = std::make_unique<Eigen::SparseMatrix<double>>(
            nxr::manifold::ops::graphLaplacian(*h.ctx));
    auto& dec = ensureDec(h);
    const char* f[] = {"laplacian","dec"};
    mxArray* s = mxCreateStructMatrix(1,1,2,f);
    mxSetField(s,0,"laplacian", eigenSparseToMx(*h.graphLap));
    const char* df[] = {"d0","d1"};
    mxArray* d = mxCreateStructMatrix(1,1,2,df);
    mxSetField(d,0,"d0", eigenSparseToMx(dec.d0));
    mxSetField(d,0,"d1", eigenSparseToMx(dec.d1));
    mxSetField(s,0,"dec", d);
    return s;
}

mxArray* buildGeometryOperators(ContextHolder& h) {
    auto& ops = ensureOps(h);
    auto& dec = ensureDec(h);
    if (!h.massGalerkin) {
        auto g = nxr::manifold::ops::assembleManifoldOperators(
                     *h.ctx, nxr::manifold::ops::MassMatrixVariant::Galerkin);
        h.massGalerkin = std::make_unique<Eigen::SparseMatrix<double>>(g.mass);
    }
    const char* f[] = {"laplacian","mass","hodge"};
    mxArray* s = mxCreateStructMatrix(1,1,3,f);
    mxSetField(s,0,"laplacian", eigenSparseToMx(ops.cotanLaplacian));
    const char* mf[] = {"lumped","galerkin"};
    mxArray* mm = mxCreateStructMatrix(1,1,2,mf);
    mxSetField(mm,0,"lumped",   eigenSparseToMx(dec.hodge0));
    mxSetField(mm,0,"galerkin", eigenSparseToMx(*h.massGalerkin));
    mxSetField(s,0,"mass", mm);
    const char* hf[] = {"h0","h1","h2","h1inv"};
    mxArray* hh = mxCreateStructMatrix(1,1,4,hf);
    mxSetField(hh,0,"h0",    eigenSparseToMx(dec.hodge0));
    mxSetField(hh,0,"h1",    eigenSparseToMx(dec.hodge1));
    mxSetField(hh,0,"h2",    eigenSparseToMx(dec.hodge2));
    mxSetField(hh,0,"h1inv", eigenSparseToMx(dec.hodge1Inverse));
    mxSetField(s,0,"hodge", hh);
    return s;
}

// type/opts mirror buildGaugeStruct; opts may carry singVerts/singValues for trivial.
mxArray* buildGaugeOperators(ContextHolder& h, const std::string& type, const mxArray* opts) {
    namespace cl = nxr::manifold::ops::laplacian::connection;
    cl::ConnectionLaplacianOptions o;
    o.domain = cl::ConnectionDomain::Vertex;
    o.nSym   = 1;
    o.format = cl::ConnectionLaplacianFormat::Complex;
    Eigen::SparseMatrix<std::complex<double>> K;
    if (type == "trivial") {
        if (!opts || !mxIsStruct(opts))
            throw std::invalid_argument("gauge('trivial') operators require singVerts/singValues");
        const mxArray* fv = mxGetField(opts, 0, "singVerts");
        const mxArray* fi = mxGetField(opts, 0, "singValues");
        if (!fv || !fi) throw std::invalid_argument("opts needs singVerts and singValues");
        std::vector<int> verts = mxToVertexIndices(fv);
        Eigen::VectorXd  vals  = mxToEigenVector(fi);
        std::map<int,double> sing;
        for (size_t i = 0; i < verts.size(); ++i) sing[verts[i]] = vals[i];
        auto clr = cl::assembleTrivialConnectionLaplacian(*h.ctx, sing, ensureDec(h), *h.cache, o);
        K = clr.K_complex;
    } else {  // euclidean / levi-civita → Levi-Civita connection Laplacian (clCache)
        ContextHolder::CLKey key{o.domain, o.nSym, o.regularization, o.format};
        auto it = h.clCache.find(key);
        if (it == h.clCache.end()) {
            auto clp = std::make_shared<cl::ConnectionLaplacian>(cl::assembleConnectionLaplacian(*h.ctx, o));
            it = h.clCache.emplace(key, std::move(clp)).first;
        }
        K = it->second->K_complex;
    }
    const char* f[] = {"laplacian"};
    mxArray* s = mxCreateStructMatrix(1,1,1,f);
    mxSetField(s,0,"laplacian", eigenComplexSparseToMx(K));
    return s;
}
```

- [ ] **Step 2-note:** `assembleConnectionLaplacian` with `format=Complex` must populate `K_complex`. Confirm by reading `connection_laplacian.cpp`; if Complex format leaves `K_complex` empty, set `o.format` so the complex matrix is produced (the struct has both `K_real`/`K_complex`; Complex format fills `K_complex`).

- [ ] **Step 4: Thread an `operators` bool through the build functions.** Change the three `buildXxxStruct` signatures to take a trailing `bool withOps` and, at the end (before `return s;`/`plhs[0]=s`), attach the operators sub-struct when set, using `mxAddField`:

```cpp
// at the end of buildTopologyStruct(ContextHolder& h, bool withOps):
if (withOps) {
    int fn = mxAddField(s, "operators");
    mxSetFieldByNumber(s, 0, fn, buildTopologyOperators(h));
}
// buildGeometryStruct(h, withOps): ... buildGeometryOperators(h)
// buildGaugeStruct(h, type, opts, withOps): ... buildGaugeOperators(h, type, opts)
```

- [ ] **Step 5: Update the cmd wrappers + cmdBundle to parse the flag and pass it through.**

```cpp
void cmdTopology(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 2 || nrhs > 3) throw std::invalid_argument("nxr_compute('topology', handle[, opts])");
    ContextHolder& h = getHolder(prhs[1]);
    bool withOps = (nrhs >= 3) && readOperatorsFlag(prhs[2]);
    plhs[0] = buildTopologyStruct(h, withOps);
}
void cmdGeometry(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 2 || nrhs > 3) throw std::invalid_argument("nxr_compute('geometry', handle[, opts])");
    ContextHolder& h = getHolder(prhs[1]);
    bool withOps = (nrhs >= 3) && readOperatorsFlag(prhs[2]);
    plhs[0] = buildGeometryStruct(h, withOps);
}
void cmdGauge(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw std::invalid_argument("nxr_compute('gauge', handle, type[, opts])");
    ContextHolder& h = getHolder(prhs[1]);
    std::string type = getStringArg(prhs[2]);
    const mxArray* opts = (nrhs >= 4) ? prhs[3] : nullptr;
    plhs[0] = buildGaugeStruct(h, type, opts, readOperatorsFlag(opts));
}
void cmdBundle(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 3) throw std::invalid_argument("nxr_compute('bundle', handle, gaugeType[, opts])");
    ContextHolder& h = getHolder(prhs[1]);
    std::string type = getStringArg(prhs[2]);
    const mxArray* opts = (nrhs >= 4) ? prhs[3] : nullptr;
    bool withOps = readOperatorsFlag(opts);
    const char* f[] = {"Topology","Geometry","Gauge"};
    mxArray* s = mxCreateStructMatrix(1,1,3,f);
    mxSetField(s,0,"Topology", buildTopologyStruct(h, withOps));
    mxSetField(s,0,"Geometry", buildGeometryStruct(h, withOps));
    mxSetField(s,0,"Gauge",    buildGaugeStruct(h, type, opts, withOps));
    plhs[0] = s;
}
```
Update the forward-declared/earlier signatures of `buildTopologyStruct`/`buildGeometryStruct`/`buildGaugeStruct` accordingly (existing callers pass the new bool).

- [ ] **Step 6: Build** — `bash scripts/build.sh Release 2>&1 | tail -10` (clean + mexmaca64). Fix compile errors (field-name mismatches, `mxIsLogicalScalarTrue` availability — if missing, use `mxIsLogical(f) && mxGetLogicals(f)[0]`).

- [ ] **Step 7: Create `bindings/mex/test/test_operators.m`** (copy `local_icosahedron()` from test_topology.m):

```matlab
function test_operators
fprintf('[test_operators] starting\n');
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'mex not found'); addpath(hits(1).folder); clear nxr_compute

[V,F] = local_icosahedron(); nV=size(V,1); nF=size(F,1); nE=nV+nF-2;
h = nxr_compute('create', V, F);

% light by default: no operators field
G0 = nxr_compute('geometry', h);
assert(~isfield(G0,'operators'), 'geometry light by default');

% topology operators
T = nxr_compute('topology', h, struct('operators',true));
assert(isfield(T,'operators'), 'topology.operators present');
L = T.operators.laplacian;
assert(issparse(L) && isequal(size(L),[nV nV]), 'graph L V×V sparse');
assert(max(abs(L*ones(nV,1))) < 1e-12, 'graph L zero row sums');
assert(nnz(L-L') == 0, 'graph L symmetric');
assert(isequal(size(T.operators.dec.d0),[nE nV]), 'd0 E×V');
assert(isequal(size(T.operators.dec.d1),[nF nE]), 'd1 F×E');
assert(nnz(T.operators.dec.d1 * T.operators.dec.d0) == 0, 'd1*d0 == 0');

% geometry operators
Gg = nxr_compute('geometry', h, struct('operators',true));
Lc = Gg.operators.laplacian;
assert(issparse(Lc) && isequal(size(Lc),[nV nV]), 'cotan V×V');
assert(max(abs(Lc*ones(nV,1))) < 1e-9, 'cotan zero row sums');
assert(isequal(size(Gg.operators.hodge.h1),[nE nE]), 'h1 E×E');
% cross-surface identity: cotan == d0' * h1 * d0
d0 = T.operators.dec.d0; h1 = Gg.operators.hodge.h1;
assert(max(max(abs(Lc - d0'*h1*d0))) < 1e-9, 'cotan == d0''*h1*d0');

% gauge operators (levi-civita vs trivial differ; complex Hermitian)
Gl = nxr_compute('gauge', h, 'levi-civita', struct('operators',true));
Kl = Gl.operators.laplacian;
assert(~isreal(Kl) && isequal(size(Kl),[nV nV]), 'connection L complex V×V');
assert(norm(Kl - Kl','fro') < 1e-9, 'connection L Hermitian');
opts = struct('singVerts',uint32([1;2]),'singValues',[1;1],'operators',true);
Gt = nxr_compute('gauge', h, 'trivial', opts);
Kt = Gt.operators.laplacian;
assert(norm(Kt - Kt','fro') < 1e-9, 'trivial connection L Hermitian');
assert(norm(Kt - Kl,'fro') > 1e-6, 'trivial differs from levi-civita');

% bundle with operators == standalone with operators
B = nxr_compute('bundle', h, 'levi-civita', struct('operators',true));
assert(isequal(B.Topology.operators.laplacian, T.operators.laplacian), 'bundle topo ops match');
assert(isequal(B.Geometry.operators.laplacian, Gg.operators.laplacian), 'bundle geo ops match');

nxr_compute('destroy', h);
fprintf('ALL TESTS PASSED: test_operators\n');
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

- [ ] **Step 8: Run** — build, then via MATLAB MCP run `bindings/mex/test/test_operators.m` AND the existing `test_topology.m`, `test_geometry_bundle.m`, `test_gauge.m`, `test_bundle.m` (the build-function signature change must not break them). All print ALL TESTS PASSED. Also native `./build/test_geometry_bundle`. If a MATLAB assertion fails, investigate (field names, `K_complex` empty under Complex format, the `d1*d0==0` orientation, or the cotan identity sign).

- [ ] **Step 9: Commit**
```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_operators.m
git commit -m "feat(mex): add opt-in .operators surface (graph/cotan/connection L, DEC, mass, hodge)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

| Spec item | Task |
|---|---|
| graph Laplacian = d0ᵀd0 | Task 1 |
| native complex sparse marshaler | Task 2 |
| Topology/Geometry/Gauge operator builders | Task 3 Step 3 |
| operators flag, light by default | Task 3 Steps 2,4,5 |
| gauge connection follows gauge type | Task 3 buildGaugeOperators |
| caching (graph/galerkin slots + existing ops/dec/clCache) | Task 3 Step 1,3 |
| cross-surface identity cotan == d0ᵀ h1 d0 | Task 3 test |
| light-by-default preserved (no operators field) | Task 3 test |

**Placeholders:** none. **Signature consistency:** `buildXxxStruct` gains `bool withOps`; all call sites (cmd wrappers + cmdBundle) updated in Step 5. `graphLaplacian` decl (T1) matches use (T3). `eigenComplexSparseToMx` (T2) used in `buildGaugeOperators` (T3).
