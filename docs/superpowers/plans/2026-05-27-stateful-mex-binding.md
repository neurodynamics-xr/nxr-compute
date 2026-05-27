# Stateful MEX binding at WASM parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `bindings/mex/src/nxr_compute_mex.cpp` stateful — a C++ handle map (the MEX analogue of WASM's `ContextWrapper`) exposed through `create`/`destroy` + handle-based op commands — and bring it to full parity with the WASM `ContextWrapper` surface.

**Architecture:** A file-scope `std::unordered_map<uint64_t, ContextHolder>` keeps the geometry-central `Manifold`, assembled operators, `CholeskyCache`, eigen result, and the stateful geometry-central solvers alive across calls. Each op command detects a `uint64` handle in `prhs[1]` and runs against cached state; absent a handle it falls back to the existing stateless V,F path, unchanged. New parity ops are handle-only.

**Tech Stack:** C++17, Eigen, geometry-central, MATLAB MEX C-API (`matlab_add_mex`, R2018a interleaved-complex API), libut Ctrl-C bridge. Tests run through the MATLAB MCP server.

> **Status: IMPLEMENTED** on branch `feat/stateful-mex-binding` (2026-05-27). All phases complete; `nxr_compute_mex_smoke.m`, `test_mex_context.m`, and `test_mex_parity.m` green on MATLAB R2023b / Apple Silicon (`.mexmaca64`).

**Reference docs:** `docs/superpowers/specs/2026-05-27-stateful-mex-binding-design.md` (design), `include/nxr/compute.h` (verbatim API), `bindings/wasm/src/nxr_compute_wasm.cpp` (the stateful pattern to mirror), `scripts/_smoke-wasm.mjs` (the parity test to mirror).

**Verbatim entry points used below** (from `include/nxr/compute.h`):
- `nxr::manifold::Manifold(const double* v, int nV, const int32_t* f, int nF)`
- `nxr::manifold::ops::assembleManifoldOperators(Manifold&, MassMatrixVariant=Voronoi)`
- `nxr::manifold::ops::DECOperators assembleDECOperators(Manifold&)`
- `nxr::manifold::ops::CholeskyCache` (default ctor; `laplacian/hodgeExact/hodgeCoExact`)
- `nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(Manifold&, const ConnectionLaplacianOptions&={})`
- `nxr::manifold::solve::eigen(K, M, k, sigma, normalize, removeDC, cancel, progress)`
- `nxr::manifold::solve::poisson(ManifoldOperators&, CholeskyCache&, const std::map<int,double>&)`
- `nxr::manifold::solve::HeatGeodesicSolver(Manifold&, double tCoef=1.0)` + `heat(solver, const std::vector<int>&)`
- `nxr::manifold::solve::SignedHeatSolver(Manifold&, double tCoef=1.0)` + `signedHeat(solver, curve, isLoop, levelSet)`
- `nxr::manifold::solve::hodge(Manifold&, DECOperators&, CholeskyCache&, const Eigen::VectorXd& omega)` → `HodgeResult`
- `nxr::manifold::query::tracePath(Manifold&, int, int)` → `[N,3]`
- `nxr::manifold::transport::VectorHeatSolver(Manifold&, double=1.0)` + `parallel/extendScalar/logMap/findCenter`
- `nxr::manifold::connection::trivial(Manifold&, DECOperators&, CholeskyCache&, const std::map<int,double>&)` → `DirectionFieldResult`
- `nxr::manifold::connection::smoothFace(Manifold&, int=4, bool=false)` / `smoothVertex(Manifold&, int=2, bool=false)`
- `nxr::manifold::parametrization::bff(Manifold&)` → `[V,2]`
- `nxr::manifold::geometry::curvatures(Manifold&)` → `CurvatureResult` / `normals(Manifold&, NormalType)` / `frames(Manifold&)` → `FaceFrames`
- `nxr::field::generate::heatDiffusion(ManifoldOperators&, EigenResult&, u0, timesteps, alpha=1.0)` → `MatrixXf[T,n]`
- `nxr::field::generate::dampedWave(EigenResult&, modeIndices, amplitudes, dampings, phases, timesteps)` → `MatrixXf[T,nV]`
- `nxr::field::generate::randomDecomposed1Form(DECOperators&, nV, nE, nF, αStr, βStr, γStr, seed=42)`
- `nxr::field::interp::whitney(Manifold&, DECOperators&, oneForm)` → `[nF,3]`
- `nxr::field::op::gradient(Manifold&, scalarField)` → `[nF,3]`
- `nxr::field::extract::isoline(Manifold&, scalarField, numLevels=20, min=0, max=0)` → `IsolineResult`
- `nxr::field::extract::streamline(Manifold&, faceField, numSeeds=15, stepCoef=0.15, maxSteps=1000)` → `StreamlineResult`

---

## File structure

| File | Responsibility |
|---|---|
| `bindings/mex/src/nxr_compute_mex.cpp` | Dispatcher. Gains the handle map, `ContextHolder`, `create`/`destroy`, `mexAtExit`, lazy `ensure*`, handle branches for existing ops, and all new parity `cmdXxx`. |
| `bindings/mex/src/marshal.h` | Stateless mxArray↔Eigen converters + `*ToStruct` builders. Gains builders for the new parity return types. |
| `include/nxr/errors.h` | Add `InvalidHandle` to `ErrorCode` + `errorCodeName`. |
| `bindings/mex/test/nxr_compute_mex_smoke.m` | Existing stateless smoke. Fix hardcoded `.mexw64` → `mexext`; fix artifact path. |
| `bindings/mex/test/test_mex_context.m` | **New.** Stateful lifecycle / caching / coexistence / error tests. |
| `bindings/mex/test/test_mex_parity.m` | **New.** Full WASM-parity numerical harness on the icosahedron, handle mode. |

---

## Phase 0 — Build & baseline (GATE)

The whole plan depends on building + running the MEX on this Apple-Silicon Mac. Do not proceed past this phase until its exit criterion is met.

### Task 0.1: Confirm the MATLAB toolchain

- [ ] **Step 1: Confirm MATLAB + MEX compiler are reachable.**
  Run (MATLAB MCP `evaluate_matlab_code`): `mexext, cc = mex.getCompilerConfigurations('C++','Selected'); disp(cc.Name)`
  Expected: prints `mexmaca64` and a C++ compiler name (e.g. `Xcode Clang++`). If no compiler is configured, run `mex -setup C++` and report to the user before continuing.

### Task 0.2: Build the static lib + MEX

- [ ] **Step 1: Build.**
  Run: `bash scripts/build.sh Release`
  Expected: CMake prints `nxr-compute-mex: building against MATLAB at <path>` and `libut at <path>`; build completes without error.

- [ ] **Step 2: Locate the built MEX artifact.**
  Run: `find build -name 'nxr_compute.mex*' -print`
  Expected: one path, e.g. `build/bindings/mex/nxr_compute.mexmaca64` (record the exact directory — the smoke test's `addpath` must point here).

### Task 0.3: Make the stateless smoke test portable, then run it

**Files:** Modify `bindings/mex/test/nxr_compute_mex_smoke.m`

- [ ] **Step 1: Replace the hardcoded Windows artifact name + path with portable equivalents.**
  Change the artifact locator block to resolve the build dir and the platform extension dynamically:

```matlab
thisDir = fileparts(mfilename('fullpath'));
% Build output: <repo>/build/.../nxr_compute.<mexext>. Search for it so
% the test is independent of the per-platform CMake output subdir.
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found under %s/build', mexext, repoRoot);
mexDir = hits(1).folder;
addpath(mexDir);
assert(exist(fullfile(mexDir, ['nxr_compute.' mexext]), 'file') == 3, ...
    'nxr_compute.%s not found in %s', mexext, mexDir);
```

- [ ] **Step 2: Run the stateless smoke test.**
  Run (MATLAB MCP `run_matlab_file`): `bindings/mex/test/nxr_compute_mex_smoke.m`
  Expected: ends with `[nxr_compute_mex] all assertions passed ✓`.

- [ ] **Step 3: Commit.**

```bash
git add bindings/mex/test/nxr_compute_mex_smoke.m
git commit -m "test(mex): make smoke test platform-portable (mexext + glob build dir)"
```

**Exit criterion:** stateless MEX builds and `nxr_compute_mex_smoke.m` is green on this machine. If the toolchain is unavailable, STOP and report — do not build the rest of the plan on an unbuildable target.

---

## Phase 1 — Stateful core

Add the handle map and wire it into the existing ops. After this phase the existing stateless commands AND a parallel handle-based path both work.

### Task 1.1: Add the `InvalidHandle` error code

**Files:** Modify `include/nxr/errors.h`

- [ ] **Step 1: Add the enumerator.** In `enum class ErrorCode { … }` add `InvalidHandle,` after `InvalidInput,`.

- [ ] **Step 2: Add its name.** In `errorCodeName`, add the case mapping `InvalidHandle` → `"INVALID_HANDLE"` (follow the exact `case ErrorCode::X: return "...";` form used for the neighbours).

- [ ] **Step 3: Build to confirm no break.**
  Run: `bash scripts/build.sh Release`
  Expected: builds clean (the new enumerator is additive; `toMatlabIdentifier` in the MEX will turn it into `invalidHandle`).

- [ ] **Step 4: Commit.**

```bash
git add include/nxr/errors.h
git commit -m "feat(core): add InvalidHandle error code for stateful bindings"
```

### Task 1.2: Add the handle map, `create`, `destroy`, `mexAtExit`

**Files:** Modify `bindings/mex/src/nxr_compute_mex.cpp`

- [ ] **Step 1: Write the failing test** (`bindings/mex/test/test_mex_context.m`, new file — start it here; later tasks append).

```matlab
function test_mex_context()
% Stateful MEX context: lifecycle, caching, coexistence, errors.
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'MEX not built'); addpath(hits(1).folder);

[V, F] = local_icosahedron();

% ── create / destroy ──────────────────────────────────────
h1 = nxr_compute('create', V, F);
assert(isa(h1, 'uint64') && isscalar(h1), 'create must return a uint64 scalar handle');
h2 = nxr_compute('create', V, F);
assert(h1 ~= h2, 'distinct contexts must get distinct handles');
nxr_compute('destroy', h1);
nxr_compute('destroy', h2);
fprintf('  create/destroy ✓\n');
end

function [V, F] = local_icosahedron()
t = (1 + sqrt(5)) / 2;
raw = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; ...
        0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
V = raw ./ vecnorm(raw, 2, 2);
F = [ 1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; 2 6 10; 6 12 5; ...
     12 11 3; 11 8 7; 8 2 9; 4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; ...
      5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
end
```

- [ ] **Step 2: Run it to verify it fails.**
  Run (MATLAB MCP `run_matlab_test_file`): `bindings/mex/test/test_mex_context.m`
  Expected: FAIL — `Unknown command: "create"`.

- [ ] **Step 3: Implement.** In the anonymous namespace at the top of `nxr_compute_mex.cpp`, add the include and the storage:

```cpp
#include <unordered_map>
#include <cstdint>
#include <utility>   // std::pair
#include <tuple>     // CLKey
```

```cpp
// ── Stateful context handle map ──────────────────────────────
// MEX analogue of the WASM ContextWrapper. Each ContextHolder keeps
// the geometry-central mesh + caches + stateful solvers alive across
// calls; the uint64 handle is the MATLAB-facing proxy.
struct ContextHolder {
    std::unique_ptr<nxr::manifold::Manifold>                      ctx;
    std::unique_ptr<nxr::manifold::ops::ManifoldOperators>        ops;
    std::unique_ptr<nxr::manifold::ops::DECOperators>             dec;
    std::unique_ptr<nxr::manifold::ops::CholeskyCache>            cache;
    std::unique_ptr<nxr::manifold::solve::EigenResult>            eigCache;
    std::unique_ptr<nxr::manifold::transport::VectorHeatSolver>   vhm;
    std::unique_ptr<nxr::manifold::solve::SignedHeatSolver>       shs;
    std::unique_ptr<nxr::manifold::solve::HeatGeodesicSolver>     heatGeo;
    std::map<std::pair<int,bool>, Eigen::MatrixXd>                          smoothFaceFieldCache;
    std::map<std::pair<int,bool>, nxr::manifold::connection::SmoothVertexFieldResult> smoothVertexFieldCache;
    using CLKey = std::tuple<
        nxr::manifold::ops::laplacian::connection::ConnectionDomain,
        int, double,
        nxr::manifold::ops::laplacian::connection::ConnectionLaplacianFormat>;
    std::map<CLKey, std::shared_ptr<
        nxr::manifold::ops::laplacian::connection::ConnectionLaplacian>>     clCache;
};

static uint64_t sNextHandle = 1;
static std::unordered_map<uint64_t, ContextHolder> sContexts;

// Look up the holder for a uint64 handle in prhs[idx]; throw on miss.
ContextHolder& getHolder(const mxArray* arr) {
    if (!mxIsUint64(arr) || mxGetNumberOfElements(arr) != 1) {
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidHandle,
            "expected a scalar uint64 context handle");
    }
    uint64_t h = *static_cast<const uint64_t*>(mxGetData(arr));
    auto it = sContexts.find(h);
    if (it == sContexts.end()) {
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidHandle,
            "invalid or destroyed context handle");
    }
    return it->second;
}

// True iff prhs[1] looks like a handle (drives additive dispatch).
bool isHandleArg(int nrhs, const mxArray** prhs) {
    return nrhs >= 2 && mxIsUint64(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1;
}
```

Add the `create` / `destroy` command functions:

```cpp
void cmdCreate(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 3) {
        throw std::invalid_argument("nxr_compute('create', V, F) takes exactly 2 arguments");
    }
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);

    ContextHolder holder;
    holder.ctx   = std::make_unique<nxr::manifold::Manifold>(verts.data(), nV, faces.data(), nF);
    holder.cache = std::make_unique<nxr::manifold::ops::CholeskyCache>();

    uint64_t h = sNextHandle++;
    sContexts.emplace(h, std::move(holder));

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(plhs[0])) = h;
}

void cmdDestroy(int /*nlhs*/, mxArray** /*plhs*/, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument("nxr_compute('destroy', handle) takes exactly 1 argument");
    }
    if (mxIsUint64(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1) {
        sContexts.erase(*static_cast<const uint64_t*>(mxGetData(prhs[1])));
    }
}
```

Register the `mexAtExit` cleanup once, at the top of `mexFunction` (before the dispatch):

```cpp
static bool sAtExitRegistered = false;
if (!sAtExitRegistered) {
    mexAtExit([]() { sContexts.clear(); });
    sAtExitRegistered = true;
}
```

Add to the dispatch chain in `mexFunction`:

```cpp
else if (cmd == "create")   cmdCreate(nlhs, plhs, nrhs, prhs);
else if (cmd == "destroy")  cmdDestroy(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 4: Build and run the test.**
  Run: `bash scripts/build.sh Release` then `run_matlab_test_file` on `test_mex_context.m`.
  Expected: PASS — `create/destroy ✓`.

- [ ] **Step 5: Commit.**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_mex_context.m
git commit -m "feat(mex): add stateful context handle map + create/destroy"
```

### Task 1.3: Lazy `ensure*` helpers + handle path for `assembleManifoldOperators`

**Files:** Modify `bindings/mex/src/nxr_compute_mex.cpp`

- [ ] **Step 1: Append failing test** to `test_mex_context.m` (call from `test_mex_context`):

```matlab
% ── assembleManifoldOperators: handle path == stateless path ──
h = nxr_compute('create', V, F);
opsH = nxr_compute('assembleManifoldOperators', h);
opsS = nxr_compute('assembleManifoldOperators', V, F);
assert(opsH.nV == size(V,1) && opsH.nF == size(F,1), 'handle ops sizes');
assert(norm(opsH.cotanLaplacian - opsS.cotanLaplacian, 'fro') < 1e-12, 'K handle==stateless');
assert(norm(opsH.mass - opsS.mass, 'fro') < 1e-12, 'M handle==stateless');

% Cache hit: second assemble is the cached struct (numerically identical).
t1 = tic; nxr_compute('assembleManifoldOperators', h); first = toc(t1);
t2 = tic; opsH2 = nxr_compute('assembleManifoldOperators', h); second = toc(t2);
assert(norm(opsH2.cotanLaplacian - opsH.cotanLaplacian,'fro') < 1e-15, 'cached K identical');
nxr_compute('destroy', h);
fprintf('  assembleManifoldOperators handle path ✓ (cold %.2g s, warm %.2g s)\n', first, second);
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — handle path returns error or wrong result (`assembleManifoldOperators` doesn't yet branch on a handle).

- [ ] **Step 3: Implement.** Add lazy helpers near the handle map:

```cpp
nxr::manifold::ops::ManifoldOperators& ensureOps(ContextHolder& h) {
    if (!h.ops) h.ops = std::make_unique<nxr::manifold::ops::ManifoldOperators>(
        nxr::manifold::ops::assembleManifoldOperators(*h.ctx));
    return *h.ops;
}
nxr::manifold::ops::DECOperators& ensureDec(ContextHolder& h) {
    if (!h.dec) h.dec = std::make_unique<nxr::manifold::ops::DECOperators>(
        nxr::manifold::ops::assembleDECOperators(*h.ctx));
    return *h.dec;
}
nxr::manifold::transport::VectorHeatSolver& ensureVHM(ContextHolder& h) {
    if (!h.vhm) h.vhm = std::make_unique<nxr::manifold::transport::VectorHeatSolver>(*h.ctx);
    return *h.vhm;
}
nxr::manifold::solve::SignedHeatSolver& ensureSHS(ContextHolder& h) {
    if (!h.shs) h.shs = std::make_unique<nxr::manifold::solve::SignedHeatSolver>(*h.ctx);
    return *h.shs;
}
nxr::manifold::solve::HeatGeodesicSolver& ensureHeatGeo(ContextHolder& h) {
    if (!h.heatGeo) h.heatGeo = std::make_unique<nxr::manifold::solve::HeatGeodesicSolver>(*h.ctx);
    return *h.heatGeo;
}
```

Add a handle branch at the top of `cmdAssembleMeshOperators` (keep the existing stateless body as the `else`):

```cpp
void cmdAssembleMeshOperators(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (isHandleArg(nrhs, prhs)) {
        ContextHolder& h = getHolder(prhs[1]);
        auto& ops = ensureOps(h);
        plhs[0] = meshOperatorsToStruct(ops, h.ctx->nV(), h.ctx->nE(), h.ctx->nF());
        return;
    }
    // ── existing stateless path unchanged below ──
    if (nrhs != 3) { /* …existing… */ }
    /* …existing body… */
}
```

- [ ] **Step 4: Build + run test.** Expected: PASS, with warm time ≤ cold time.

- [ ] **Step 5: Commit.**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_mex_context.m
git commit -m "feat(mex): lazy ensure* helpers + handle path for assembleManifoldOperators"
```

### Task 1.4: Handle path for `solve` (pull K,M from cached ops)

**Files:** Modify `bindings/mex/src/nxr_compute_mex.cpp`

- [ ] **Step 1: Append failing test** to `test_mex_context.m`:

```matlab
% ── solve in handle mode: K/M come from cached ops ──
h = nxr_compute('create', V, F);
eig = nxr_compute('solve', h, 6);
assert(eig.k == 6, 'handle solve returns k modes');
assert(isequal(size(eig.eigenvectors), [size(V,1) 6]), 'eigvec V×k');
assert(issorted(eig.eigenvalues), 'eigvals ascending');
assert(abs(eig.eigenvalues(2) - 2.0) < 1e-6, 'icosa λ₁≈2');
nxr_compute('destroy', h);
fprintf('  solve handle path ✓\n');
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — handle `solve` mis-parses (`prhs[1]` is a handle, not sparse K).

- [ ] **Step 3: Implement.** Add a handle branch at the top of `cmdSolveEigenmodes`:

```cpp
if (isHandleArg(nrhs, prhs)) {
    if (nrhs != 3) {
        throw std::invalid_argument("nxr_compute('solve', handle, k) takes exactly 2 arguments");
    }
    ContextHolder& h = getHolder(prhs[1]);
    auto& ops = ensureOps(h);
    int k = getIntArg(prhs[2]);
    auto result = nxr::manifold::solve::eigen(
        ops.cotanLaplacian, ops.mass, k, -1e-8,
        /*normalize=*/false, /*removeDC=*/false, makeCtrlCToken());
    h.eigCache = std::make_unique<nxr::manifold::solve::EigenResult>(result);
    plhs[0] = eigenResultToStruct(result);
    return;
}
// ── existing stateless (K, M, k) path unchanged below ──
```

- [ ] **Step 4: Build + run test.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_mex_context.m
git commit -m "feat(mex): handle-mode solve pulls K,M from cached ops; caches eigenresult"
```

### Task 1.5: Handle paths for `precompute` and the remaining wired ops

Each existing op gains a handle branch that routes through cached state / `ensure*`. The stateless bodies stay unchanged. Apply the same TDD micro-loop (append test → fail → implement branch → pass → commit) for each.

- [ ] **`precompute`** — `nxr_compute('precompute', h, k)`: `ensureOps`, run `eigen(..., normalize=true, removeDC=true, makeCtrlCToken())`, store in `eigCache`, return struct.
- [ ] **`parallel`** — `nxr_compute('parallel', h, srcVerts, srcVecs)`: `ensureVHM`, call `transport::parallel`.
- [ ] **`extendScalar`** — `nxr_compute('extendScalar', h, srcVerts, srcVals)`: `ensureVHM`, call `transport::extendScalar`.
- [ ] **`logMap`** — `nxr_compute('logMap', h, srcVertex, [strategy])`: `ensureVHM`, call `transport::logMap`.
- [ ] **`findCenter`** — `nxr_compute('findCenter', h, srcVerts, [p])`: `ensureVHM`, call `transport::findCenter`.
- [ ] **`signedHeat`** — `nxr_compute('signedHeat', h, curveVerts, isLoop, [levelSet])`: `ensureSHS`, call `solve::signedHeat`.
- [ ] **`smoothFace`** — `nxr_compute('smoothFace', h, [nSym], [align])`: cache lookup in `smoothFaceFieldCache` keyed by `{nSym, align}`; miss → `connection::smoothFace`, store, return.
- [ ] **`smoothVertex`** — `nxr_compute('smoothVertex', h, [nSym], [align])`: cache lookup in `smoothVertexFieldCache`; miss → `connection::smoothVertex`, store, return struct.
- [ ] **`compute`** — `nxr_compute('compute', h, vertexFieldRaw, freq, [connect])`: call `stripes::compute`.
- [ ] **`computeFreq`** — `nxr_compute('computeFreq', h, vertexFieldRaw, freqs, [connect])`: call `stripes::computeFreq`.
- [ ] **`normalize` / `removeDC`** — leave stateless only (they operate on already-extracted matrices/structs; no mesh state needed). Document that they have no handle form.

For each: the test asserts the handle-mode output shape matches the existing stateless smoke's assertion for that op (e.g. `parallel` → V×3, `signedHeat` → straddles zero, `smoothVertex.vertexFieldRaw` length `2*nV`). Commit per op or per small group.

### Task 1.6: Coexistence + invalid-handle tests

- [ ] **Step 1: Append tests** to `test_mex_context.m`:

```matlab
% ── two coexisting contexts on different meshes ──
[V1, F1] = local_icosahedron();
V2 = V1 * 2;                         % scaled copy (different geometry)
a = nxr_compute('create', V1, F1);
b = nxr_compute('create', V2, F1);
opsA = nxr_compute('assembleManifoldOperators', a);
opsB = nxr_compute('assembleManifoldOperators', b);
assert(abs(opsB.totalArea - 4*opsA.totalArea) < 1e-6, 'scaled mesh has 4× area');
nxr_compute('destroy', a);
% b still works after a is destroyed
opsB2 = nxr_compute('assembleManifoldOperators', b);
assert(abs(opsB2.totalArea - opsB.totalArea) < 1e-12, 'b survives a destroy');
% using a destroyed handle errors with nxr:invalidHandle
caught = false;
try, nxr_compute('assembleManifoldOperators', a); catch e, caught = strcmp(e.identifier,'nxr:invalidHandle'); end
assert(caught, 'destroyed handle must raise nxr:invalidHandle');
nxr_compute('destroy', b);
fprintf('  coexistence + invalid-handle ✓\n');
```

- [ ] **Step 2: Run.** Expected: PASS (implementation already supports this from 1.2–1.3).
- [ ] **Step 3: Re-run the stateless smoke** (`nxr_compute_mex_smoke.m`) to confirm no regression. Expected: still green.
- [ ] **Step 4: Commit.**

```bash
git add bindings/mex/test/test_mex_context.m
git commit -m "test(mex): context coexistence + invalid-handle error contract"
```

---

## Phase 2 — Wire to parity

Each new op is **handle-only**. Pattern for every op (Task 2.x sub-bullets follow it exactly):

1. Add any `*ToStruct` builder to `marshal.h` (reusing `eigenMatrixToMx` / `eigenVectorToMx` / `eigenSparseToMx`).
2. Add `cmdXxx(...)` in `nxr_compute_mex.cpp`: `getHolder(prhs[1])`, parse remaining args, route through `ensure*` / cache, build the return struct.
3. Add the `else if (cmd == "xxx") cmdXxx(...)` dispatch line and update the unknown-command help string.
4. Append a test to `test_mex_parity.m` asserting shape + a numerical invariant.
5. Build, run, commit.

### Task 2.1: Worked exemplar — `assembleDECOperators` (Group B)

**Files:** Modify `bindings/mex/src/marshal.h`, `bindings/mex/src/nxr_compute_mex.cpp`; create `bindings/mex/test/test_mex_parity.m`.

- [ ] **Step 1: Write the failing test** (`test_mex_parity.m`, new — establishes the harness; later ops append to it):

```matlab
function test_mex_parity()
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'MEX not built'); addpath(hits(1).folder);
[V, F] = local_icosahedron();   % same helper as test_mex_context
h = nxr_compute('create', V, F);

% ── assembleDECOperators ──
dec = nxr_compute('assembleDECOperators', h);
assert(isequal(size(dec.d0), [30 12]), 'd0 nE×nV');
assert(isequal(size(dec.d1), [20 30]), 'd1 nF×nE');
assert(issparse(dec.hodge1) && isequal(size(dec.hodge1), [30 30]), 'hodge1 nE×nE');
fprintf('  assembleDECOperators ✓\n');

nxr_compute('destroy', h);
end
% (copy local_icosahedron() helper from test_mex_context.m)
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — `Unknown command: "assembleDECOperators"`.

- [ ] **Step 3: Implement the marshaller** in `marshal.h`:

```cpp
inline mxArray* decOperatorsToStruct(const nxr::manifold::ops::DECOperators& dec) {
    const char* fields[] = {"d0","d1","hodge0","hodge1","hodge2","hodge1Inverse"};
    mxArray* s = mxCreateStructMatrix(1, 1, 6, fields);
    mxSetField(s, 0, "d0",            eigenSparseToMx(dec.d0));
    mxSetField(s, 0, "d1",            eigenSparseToMx(dec.d1));
    mxSetField(s, 0, "hodge0",        eigenSparseToMx(dec.hodge0));
    mxSetField(s, 0, "hodge1",        eigenSparseToMx(dec.hodge1));
    mxSetField(s, 0, "hodge2",        eigenSparseToMx(dec.hodge2));
    mxSetField(s, 0, "hodge1Inverse", eigenSparseToMx(dec.hodge1Inverse));
    return s;
}
```

Implement the command in `nxr_compute_mex.cpp`:

```cpp
void cmdAssembleDECOperators(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs != 2) {
        throw std::invalid_argument("nxr_compute('assembleDECOperators', handle) takes 1 argument");
    }
    ContextHolder& h = getHolder(prhs[1]);
    plhs[0] = decOperatorsToStruct(ensureDec(h));
}
```

Register: `else if (cmd == "assembleDECOperators") cmdAssembleDECOperators(nlhs, plhs, nrhs, prhs);` and add it to the help string.

- [ ] **Step 4: Build + run.** Expected: PASS — `assembleDECOperators ✓`.
- [ ] **Step 5: Commit.** `git commit -m "feat(mex): assembleDECOperators (handle)"`

### Task 2.2: Remaining Group B — operators

Follow the 2.1 pattern for each. Signatures + struct mapping + test invariant given.

- [ ] **`assembleConnectionLaplacian`** — `nxr_compute('assembleConnectionLaplacian', h, opts)` where `opts` is a struct `{domain, nSym, regularization, format}` (strings via `parseConnectionDomain`/`parseConnectionLaplacianFormat`). Build `CLKey`, look up `clCache`, miss → `assembleConnectionLaplacian(*h.ctx, o)`. Return struct `{K_real (sparse, iff Real2N), baseDim, outputDim, nSym, regularization}` (Complex format → return `K_real`+`K_imag` from `K_complex`'s real/imag triplets). Test: default opts → `K_real` is `2nV×2nV`, symmetric.
- [ ] **`frames`** — `nxr_compute('frames', h)` → struct `{e1,e2,normals}` each `[nF,3]` via `eigenMatrixToMx`. Test: per-face `‖e1‖≈1`, `e1·e2≈0`.
- [ ] **`normals`** — `nxr_compute('normals', h, [typeName])` where typeName→`NormalType` (add a `parseNormalType` helper accepting "angle"/"area"/"equal"/"sphere"/"mean"/"gauss"; default AngleWeighted). Return `[nV,3]`. Test: rows are unit length on the sphere fixture.

### Task 2.3: Group C — solvers

- [ ] **`poisson`** — `nxr_compute('poisson', h, sourceVerts, sourceValues)`: build `std::map<int,double>` density (1-based→0-based verts), `ensureOps`, `solve::poisson(ops, *h.cache, densityMap)` → `[nV]` vector. Test: result length nV, finite, M-weighted mean ≈ 0.
- [ ] **`heat`** — `nxr_compute('heat', h, sourceVerts)`: `ensureHeatGeo`, `solve::heat(solver, srcIdx)` → `[nV]`. Test: distance at source ≈ 0, all ≥ 0.
- [ ] **`tracePath`** — `nxr_compute('tracePath', h, vStart, vEnd)` (1-based→0-based): `query::tracePath(*h.ctx, a, b)` → `[N,3]` via `eigenMatrixToMx`. Test: ≥2 rows; polyline length in `(2.0, π)` for antipodes.
- [ ] **`hodge`** — `nxr_compute('hodge', h, omega)`: `ensureDec`, `solve::hodge(*h.ctx, dec, *h.cache, omega)` → struct with all `HodgeResult` fields (`exactPotential`, `coExactPotentialF`, `coExactPotentialV`, `combinedPotential`, `omega`, `dAlpha`, `deltaBeta`, `gamma` as vectors; `omegaVectors`, `dAlphaVectors`, `deltaBetaVectors`, `gammaVectors` as `[nF,3]`). Test: `dAlpha + deltaBeta + gamma ≈ omega` (max abs err < 1e-8).

### Task 2.4: Group D — geometric

- [ ] **`curvatures`** — `nxr_compute('curvatures', h)` → struct `{gaussian,mean,kMin,kMax}` vectors + `principalDirMax` `[nV,3]`. Test: `sum(gaussian) ≈ 4π` (Gauss-Bonnet on the sphere).
- [ ] **`bff`** — `nxr_compute('bff', h)` → `[V,2]`. Note: the closed icosahedron has no boundary → `bff` throws. Test asserts the throw maps to a structured `nxr:*` identifier; add a positive shape test only if a boundary fixture is available (defer otherwise).
- [ ] **`isoline`** — `nxr_compute('isoline', h, scalarField, [numLevels], [minVal], [maxVal])` → struct `{positions [2*segs,3], segmentCount}`. Test: `segmentCount ≥ 0`; `size(positions,1) == 2*segmentCount`.
- [ ] **`trivial`** — `nxr_compute('trivial', h, singVerts, singValues)`: build `std::map<int,double>`, `ensureDec`, `connection::trivial(*h.ctx, dec, *h.cache, map)` → struct `{connections, directionVectors[nF,3], orthogonalVectors[nF,3], eulerCharacteristic, gaussBonnetSatisfied}`. Test: on the icosa, singularities summing to χ=2 → `gaussBonnetSatisfied == true`.
- [ ] **`streamline`** — `nxr_compute('streamline', h, faceField, [numSeeds], [stepCoef], [maxSteps])` → struct `{positions, segmentCount}`. Test: `segmentCount ≥ 0`; positions `2*segmentCount × 3`.

### Task 2.5: Group E — vector field

- [ ] **`whitney`** — `nxr_compute('whitney', h, oneForm)`: `ensureDec`, `field::interp::whitney(*h.ctx, dec, oneForm)` → `[nF,3]`. Test: length nF; finite.
- [ ] **`gradient`** — `nxr_compute('gradient', h, scalarField)`: `field::op::gradient(*h.ctx, scalar)` → `[nF,3]`. Test: delta at vertex 1 → max gradient magnitude > 0.

### Task 2.6: Group F — time-varying generators

- [ ] **`heatDiffusion`** — `nxr_compute('heatDiffusion', h, sourceVerts, sourceValues, timesteps, [alpha])`: requires `eigCache` (throw `nxr::core::Error(NotPrecomputed, "call solve/precompute first")` if null), build `u0` via `field::generate::delta(nV, sources)`, `ensureOps`, `field::generate::heatDiffusion(ops, *h.eigCache, u0, timesteps, alpha)` → `[T,n]` matrix (convert `MatrixXf`→double `mxArray`). Test: after `precompute`, `size == [T n]`; row-sum (mass-weighted mean) ~constant across frames.
- [ ] **`dampedWave`** — `nxr_compute('dampedWave', h, modeIndices, amplitudes, dampings, phases, timesteps)`: requires `eigCache`; `field::generate::dampedWave(*h.eigCache, ...)` → `[T,nV]`. Test: `size == [T nV]`, finite.
- [ ] **`randomDecomposed1Form`** — `nxr_compute('randomDecomposed1Form', h, alphaStr, betaStr, gammaStr, [seed])`: `ensureDec`, `field::generate::randomDecomposed1Form(dec, nV, nE, nF, …)` → `[nE]`. Test: length nE; round-trips through `hodge` (dα+δβ recovers ω within tol).
- [ ] **(optional) `solveEigenmodesFromTriplets`** — low priority; MATLAB passes native sparse to `solve`. Implement only if a consumer needs the COO entry point; otherwise document as intentionally omitted (MEX-native sparse supersedes it).

---

## Phase 3 — Parity test harness

### Task 3.1: Complete the parity harness mirroring `_smoke-wasm.mjs`

**Files:** `bindings/mex/test/test_mex_parity.m`

- [ ] **Step 1:** Ensure every op from Phases 1–2 is exercised once on the icosahedron handle in `test_mex_parity.m`, each with the same numerical invariant the WASM smoke asserts (areas, eigenvalue spectrum, M-orthonormality, geodesic-at-source = 0, Gauss-Bonnet, hodge recomposition). Cross-check the eigenvalue triplet/doublet (`λ₁≈2`, `λ₃≈4.34164`) against the values baked into `scripts/_smoke-wasm.mjs` and `test_eigen.cpp` for a cross-binding numerical-parity assertion.
- [ ] **Step 2: Run both MATLAB test files.** Expected: `test_mex_context.m` and `test_mex_parity.m` both green; stateless `nxr_compute_mex_smoke.m` still green.
- [ ] **Step 3: Commit.** `git commit -m "test(mex): full WASM-parity numerical harness on icosahedron"`

### Task 3.2: Docs + finish

- [ ] **Step 1:** Update `bindings/mex/matlab/README.md` and CLAUDE.md (MEX section) to note the MEX is now stateful (handle map mirroring WASM `ContextWrapper`) and list the handle commands. Mark the design/plan docs as implemented.
- [ ] **Step 2: Commit.** `git commit -m "docs(mex): document stateful handle API"`

**Exit criterion:** every WASM `ContextWrapper` method has a green MEX counterpart on the fixture; stateful lifecycle tests pass; the stateless path is unregressed.

---

## Notes / decisions baked in

- **`InvalidHandle`** is added to the core enum (Task 1.1) → MATLAB `nxr:invalidHandle` via the existing `toMatlabIdentifier`.
- **`normalize` / `removeDC`** stay stateless-only (they transform already-materialized data; no mesh state).
- **MEX storage is host-native** (column-major dense, CSC sparse, real `V×K` matrices) — exempt from the §11 row-major flatten rule. Parity = identical numbers, not identical byte layout.
- **`heatDiffusion` / `dampedWave`** require a prior `solve`/`precompute` (read `eigCache`); throw `NotPrecomputed` otherwise (mirrors WASM).
- **`bff`** throws on the closed icosahedron (no boundary); the parity test asserts the structured throw rather than a UV shape unless a boundary fixture is added.
