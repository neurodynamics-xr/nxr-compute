# Stateful MEX binding at WASM parity — design

**Status:** implemented 2026-05-27 (branch `feat/stateful-mex-binding`)
**Author:** Diellor Basha (with Claude as scribe)
**Repository:** `nxr-compute`
**Supersedes:** the rough draft formerly at `docs/matlab-mex.md`

---

## 1. Goal

Make the MEX binding shell (`bindings/mex/src/nxr_compute_mex.cpp`)
**stateful**, mirroring the WASM binding's `ContextWrapper`
(`bindings/wasm/src/nxr_compute_wasm.cpp`). A mesh is bound once via a
`create` command; the resulting opaque handle keeps the assembled
geometry-central mesh, operator structs, Cholesky/LU factor cache, and
the stateful geometry-central solvers (`VectorHeatSolver`,
`SignedHeatSolver`, `HeatGeodesicSolver`) alive for the rest of the
handle's lifetime. Subsequent calls reuse that cached state instead of
rebuilding the halfedge mesh and re-factorizing on every invocation.

The end state is **full parity with the WASM `ContextWrapper` surface**:
every method the Embind class exposes has a handle-based MEX command
counterpart.

## 2. Key framing — where statefulness lives

The statefulness is a property of the **compiled binding shell**, not of
the host-language wrappers. In WASM, the state lives inside the C++
`ContextWrapper` object in the WASM heap; JavaScript holds only a thin
Embind proxy (a pointer). The "upstream JS classes" are application-side
conveniences over that proxy.

The MEX is the structural equivalent of the WASM module — both compile
the same `nxr_compute` engine and expose it to a host language. But the
two shells were written differently: **`nxr_compute_wasm.cpp` is
stateful** (defines `ContextWrapper`), while **`nxr_compute_mex.cpp` is
stateless** — every command constructs a fresh `nxr::manifold::Manifold`
on the stack, uses it, and destroys it on return. Nothing persists
between calls.

MEX has no Embind, so the proxy-pointer analogue is an **opaque
`uint64` handle**: `h = nxr_compute('create', V, F)` →
`nxr_compute('eigen', h, k)` → `nxr_compute('destroy', h)`. The handle
map inside the MEX *is* the `ContextWrapper` equivalent; the integer
handle is the proxy-pointer equivalent.

## 3. Non-goals (explicitly out of scope)

- **No `nxr.Manifold` MATLAB class.** A `handle`-class MATLAB wrapper is
  shelved. Any MATLAB `.m` wrapper over the handle is an application-side
  concern, exactly as the JS wrappers over the Embind `Manifold` class
  are application-side. This spec stops at the MEX (C++) layer.
- **No change to the existing `nxr.manifold.*` functional API.** Those
  `.m` leaves stay as-is (stateless, struct-based). They keep working
  unchanged because the stateless commands they call are left untouched
  (see §5).
- **No toolbox packaging** (the distributable `.zip` with per-platform
  `.mex*` artifacts). Packaging is a downstream/CI concern, not a code
  change in this work.
- **No MATLAB `+bct` numerical oracle harness.** Parity here is validated
  against the WASM / native counterparts on the shared icosahedron
  fixture. The `+bct`-oracle gap (CLAUDE.md §11) remains a separate,
  later effort.

## 4. Decisions captured during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Target surface | **Full WASM parity, phased** | The whole point is to mirror the stateful WASM build; partial parity leaves `notWired`-style gaps. |
| MATLAB-side wrapper | **Shelved** | Statefulness lives in the shell; the MATLAB handle holder is app-side, like the JS wrappers. |
| Dispatch convention | **Additive handle layer** | `create`/`destroy` + a `uint64`-handle branch in each op. Existing stateless commands stay untouched → strict superset, zero regressions to the current functional API + its smoke test. |
| Bad-handle error | **New `InvalidHandle` error code** (fallback: reuse `InvalidInput`) | Lets MATLAB pattern-match `ME.identifier == "nxr:invalidHandle"`. Confirm during Phase 1 whether touching the core enum is acceptable. |

## 5. Architecture

### 5.1 The handle map — `ContextHolder`

A file-scope table in the anonymous namespace of
`nxr_compute_mex.cpp`. Each entry mirrors the WASM `ContextWrapper`
field-for-field:

```cpp
struct ContextHolder {
    std::unique_ptr<nxr::manifold::Manifold>                      ctx;       // built at create
    std::unique_ptr<nxr::manifold::ops::ManifoldOperators>        ops;       // lazy: ensureOps()
    std::unique_ptr<nxr::manifold::ops::DECOperators>             dec;       // lazy: ensureDec()
    std::unique_ptr<nxr::manifold::ops::CholeskyCache>            cache;     // built at create
    std::unique_ptr<nxr::manifold::solve::EigenResult>            eigCache;  // set by solve/precompute
    std::unique_ptr<nxr::manifold::transport::VectorHeatSolver>   vhm;       // lazy: ensureVHM()
    std::unique_ptr<nxr::manifold::solve::SignedHeatSolver>       shs;       // lazy: ensureSHS()
    std::unique_ptr<nxr::manifold::solve::HeatGeodesicSolver>     heatGeo;   // lazy: ensureHeatGeo()

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
```

Lifecycle:

- **`create(V,F)`** — parse via existing `mxToVertexBuffer` /
  `mxToFaceBuffer`; construct `ctx` and `cache`; insert at
  `sNextHandle`; return `sNextHandle++` as an `mxUINT64_CLASS` scalar.
- **`destroy(h)`** — read `uint64` handle from `prhs[1]`; erase the entry
  (frees all owned state). Idempotent / tolerant of unknown handles.
- **`mexAtExit`** — registered once behind a static bool guard;
  `sContexts.clear()` on MEX unload.
- Lazy `ensureOps / ensureDec / ensureVHM / ensureSHS / ensureHeatGeo`
  with the WASM guard pattern `if (!ptr) ptr = std::make_unique<…>(…)`.

The handle table holder keeps `ctx` and the operator/solver state
together; no code path calls `unrequire*` on the geometry while the
view-typed `ManifoldOperators` fields are alive (CLAUDE.md lifetime
contract).

### 5.2 Dispatch routing (additive)

Each op command tests
`mxIsUint64(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1`:

- **handle present** → look up `ContextHolder`, run against cached state.
- **otherwise** → existing stateless V,F path, **unchanged**.

The 14 currently-wired commands gain a handle branch. The new parity ops
(§6, group B onward) are **handle-only** — they have no stateless
analogue, matching how WASM exposes them only as `ContextWrapper`
methods. A missing/stale handle throws `nxr::core::Error(InvalidHandle)`
→ `MException 'nxr:invalidHandle'`.

### 5.3 Caching contract (mirror WASM)

- `ops`, `dec` — assembled once on first demand, reused.
- `cache` (`CholeskyCache`) — shared across Poisson / Hodge /
  trivial-connection solves (CLAUDE.md rule 2).
- `vhm`, `shs`, `heatGeo` — stateful geometry-central solvers built once;
  the constructor pre-factors, method calls back-substitute only
  (CLAUDE.md rule 8). **Never construct inline per call.**
- `eigCache` — set by `solve` / `precompute`; read by `heatDiffusion` /
  `dampedWave`, which throw a clear error if it is unset (mirror WASM).
- `smoothFaceFieldCache`, `smoothVertexFieldCache`, `clCache` —
  result-level caches for stateless free functions, keyed by all input
  parameters (CLAUDE.md rules 9 / Pattern-C).

### 5.4 Marshalling (`marshal.h` additions)

New `*ToStruct` builders for parity return types that do not exist yet:
DEC operators, `assembleConnectionLaplacian`, `frames`, `normals`,
`tracePath`, `hodge` (α/β/γ), `curvatures` (gaussian / mean / k₁ / k₂ /
principal dirs), `bff` (uv), `isoline`, `streamline`, `heatDiffusion` /
`dampedWave` (T×V), `randomDecomposed1Form`, `whitney`, `gradient`.

**Storage convention — MEX is exempt from §11 row-major flattening.**
Where WASM returns flattened typed arrays + COO objects, the MEX returns
**MATLAB-native** values: dense → column-major `mxArray`; sparse → CSC;
`[V×K]` → a real MATLAB matrix so `U(:,k)` is mode `k` contiguously. The
existing `marshal.h` already does this (`eigenMatrixToMx` is a
column-major memcpy); new builders follow the same idiom.

> **Parity means identical numbers and identical operations, NOT
> identical byte layout.** The MEX returns the host-native representation
> for every value; only the underlying computation must match WASM /
> native bit-for-bit on the fixture.

### 5.5 Cancellation / progress

Handle-mode `solve` / `precompute` / `hodge` keep the existing
`makeCtrlCToken()` bridge (libut `utIsInterruptPending`). Progress stays
deferred — MEX has no progress surface (CLAUDE.md §12). `heatDiffusion` /
`dampedWave` require a prior `solve` / `precompute` (read `eigCache`).

## 6. Parity gap

The MEX wires 14 ops + `version` today (all stateless). The WASM
`ContextWrapper` exposes ~35 methods + 2 free functions
(`version`, `solveEigenmodesFromTriplets`).

**Group A — already wired (add handle branch + caching):**
`assembleManifoldOperators`, `solve`, `normalize`, `removeDC`,
`precompute`, `parallel`, `extendScalar`, `logMap`, `findCenter`,
`signedHeat`, `smoothFace`, `smoothVertex`, `compute`, `computeFreq`.

**Missing (handle-only, to wire for parity):**

| Group | Commands |
|---|---|
| B — operators | `assembleDECOperators`, `assembleConnectionLaplacian`, `frames`, `normals` |
| C — solvers | `poisson`, `heat`, `tracePath`, `hodge` |
| D — geometric | `curvatures`, `bff`, `isoline`, `trivial`, `streamline` |
| E — vector field | `whitney`, `gradient` |
| F — generators | `heatDiffusion`, `dampedWave`, `randomDecomposed1Form` |
| (free fn) | `solveEigenmodesFromTriplets` |

## 7. Phasing, steps, and tests

Each phase ends green before the next begins. Tests run through the
MATLAB MCP server (`run_matlab_test_file` / `run_matlab_file`).

### Phase 0 — Build & baseline (gate)

The plan depends on building + running the MEX on this Apple-Silicon Mac.

- **Steps:** confirm MATLAB + a MEX-capable compiler are reachable
  (`mex -setup C++`); build via `bash scripts/build.sh Release` →
  `nxr_compute.mexmaca64`; fix the smoke test's hardcoded `.mexw64` →
  `mexext`; resolve the build-output path the smoke test uses
  (`build/Release` vs CMake's actual MEX output dir).
- **Test:** the existing `bindings/mex/test/nxr_compute_mex_smoke.m`
  passes unchanged (other than the `mexext` fix) via the MATLAB MCP.
- **Exit criterion:** stateless MEX builds and its smoke test is green
  here. If the toolchain is unavailable, stop and report — do not build
  the rest of the plan on an unbuildable target.

### Phase 1 — Stateful core

- **Steps:** add `<unordered_map>`; add `ContextHolder`, `sContexts`,
  `sNextHandle`; `mexAtExit` guard; `create` / `destroy` commands; lazy
  `ensure*` helpers; `InvalidHandle` error code (or `InvalidInput`
  fallback); handle-detection branch wired into the 14 Group-A commands
  with their caches. `solve` in handle mode pulls `K`/`M` from
  `ensureOps()` (no need to pass them back in).
- **Tests** (`bindings/mex/test/test_mex_context.m`, new):
  - `create` returns a `uint64` scalar; two `create`s yield distinct
    handles.
  - `assembleManifoldOperators(h)` and the stateless `(V,F)` form return
    identical `K`/`M`/areas (max abs err < 1e-12).
  - Cache hit: second `assembleManifoldOperators(h)` is materially faster
    (wall-clock) than the first; numerically identical.
  - Two coexisting handles on different meshes return mesh-appropriate
    sizes; destroying one leaves the other fully functional.
  - `destroy(h)` then any op on `h` throws `nxr:invalidHandle`.
  - Stateless smoke (Phase 0) still passes — no regression.

### Phase 2 — Wire to parity

Sub-phases B–F, each = marshalling builder + handle command + test. Order
by dependency and value: **2B operators → 2C solvers → 2D geometric →
2E vector field → 2F generators**, then the `solveEigenmodesFromTriplets`
free function.

- **Per-op steps:** add the `*ToStruct` (and any `mxTo*`) marshaller; add
  `cmdXxx` reading the handle + args from `prhs`; register the
  `else if (cmd == "...")` line; route through the appropriate cached
  state / `ensure*` / result-cache.
- **Per-op test:** call on the icosahedron handle; assert the same shapes
  and numerical invariants the WASM smoke asserts for that op (e.g. DEC
  `d0` is nE×nV and `d1` is nF×nE; `heat` distance at source ≈ 0;
  `curvatures` integrate to ≈ 4π on the sphere; `hodge` recomposes ω).

### Phase 3 — Parity test harness

- **Steps:** author `bindings/mex/test/test_mex_parity.m` mirroring
  `scripts/_smoke-wasm.mjs` — every op exercised on the shared
  icosahedron fixture with the same numerical assertions, but in handle
  mode and asserting MATLAB-native shapes. Add the stateful-specific
  tests from Phase 1. Optionally cross-check eigenvalues / areas against
  the values baked into `test_eigen.cpp` and the WASM smoke for a
  cross-binding numerical-parity assertion.
- **Exit criterion:** every WASM `ContextWrapper` method has a green MEX
  counterpart on the fixture; stateful lifecycle tests pass; no
  regression to the stateless path.

## 8. Files to create / modify

| File | Action |
|---|---|
| `bindings/mex/src/nxr_compute_mex.cpp` | Add handle map, `ContextHolder`, `create`/`destroy`, `mexAtExit`, lazy `ensure*`, handle branches for Group A, and all Group B–F + free-function commands. |
| `bindings/mex/src/marshal.h` | Add `*ToStruct` / `mxTo*` builders for the parity return types. |
| `include/nxr/compute.h` (maybe) | Add `InvalidHandle` to the `ErrorCode` enum + `errorCodeName`, if we don't reuse `InvalidInput`. |
| `bindings/mex/test/nxr_compute_mex_smoke.m` | Fix hardcoded `.mexw64` → `mexext`; fix artifact path if needed. |
| `bindings/mex/test/test_mex_context.m` | **New** — stateful lifecycle + caching + coexistence + error tests. |
| `bindings/mex/test/test_mex_parity.m` | **New** — full WASM-parity numerical harness on the icosahedron. |
| `docs/matlab-mex.md` | Replace rough draft with a pointer to this spec. |

## 9. Risks

- **Build feasibility** (Phase 0 gate) — MATLAB + compiler must be usable
  here; otherwise the plan can't be executed/verified on this machine.
- **Core enum change** — `InvalidHandle` touches `nxr::core`; if that's
  undesirable, fall back to `InvalidInput` with a descriptive message.
- **`connectionLaplacian.m` ambiguity** — the existing leaf references
  both `nxr_compute(` and `notWired`; confirm its true state when wiring
  `assembleConnectionLaplacian`.
- **Solver-pattern regressions** — caching mistakes (inline solver
  construction, Pattern-C misses) are exactly what CLAUDE.md rules 8–10
  guard against; the Phase 1/3 cache-hit tests are the canary.
