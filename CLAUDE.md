# CLAUDE.md — nxr-compute

C++ compute engine for cortical surface analysis, shipped with four
bindings (N-API addon, WASM/Embind, MATLAB MEX, CLI). All four wrap
the same `nxr_compute` static library — same source, same numerical
contract.

---

## Overview

The library lives in `src/` and `include/nxr/compute.h`. The four
binding shells live under `bindings/`:

| Path | Artifact |
|---|---|
| `bindings/node/src/addon.cpp` | `nxr_compute_addon.node` (N-API) |
| `bindings/wasm/src/nxr_compute_wasm.cpp` | `nxr_compute.wasm` (Embind) |
| `bindings/mex/src/nxr_compute_mex.cpp` | `nxr_compute.mexw64` (MATLAB MEX) |
| `bindings/cli/src/main.cpp` | `nxr_compute.exe` (CLI smoke harness) |

All four are **context-based**: each mesh is bound once via
`createContext(vertices, faces)` and the resulting opaque handle holds
the assembled operator structs and a `CholeskyCache` for the rest of
its lifetime. There are no JS-side handles for individual factors —
the cache is internal to the C++ side.

`MeshOperators` and `DECOperators` mix view and value semantics. Field
names match geometry-central's canonical cache names where the concept
maps directly (`cotanLaplacian`, `vertexDualAreas`, `vertexNormals`,
all `DECOperators` fields):

- **View fields** — const-references into geometry-central's cached
  matrices: `MeshOperators::cotanLaplacian` and every field on
  `DECOperators`. `assembleMeshOperators` / `assembleDECOperators`
  pin the caches via `require*` and bind the references.
- **Value-owned fields** — `MeshOperators::mass` is variant-aware
  (`Lumped` / `Galerkin`, matching geometry-central's
  `vertexLumpedMassMatrix` and `vertexGalerkinMassMatrix`). Held by
  value so the struct shape is uniform across variants — the alternative
  would require a view variant tied to two different GC fields.
  `vertexDualAreas` and `vertexNormals` are also value-owned (geometry-
  central stores normals as a `VertexData<Vector3>`, not a matrix;
  `vertexNormals` uses the row-major `VertexNormalsMatrix` alias so
  the §11 row-major binding output is a direct memcpy).

  Note: nxr-compute's `Lumped` variant is the lumped barycentric mass
  (M_ii = sum of A_T/3 over incident triangles, sourced from
  `vertexLumpedMassMatrix`). It is **not** Meyer's mixed-Voronoi mass
  or any circumcentric construction — geometry-central does not
  expose one. Anything labelled "Voronoi mass" in older nxr-compute
  call sites or external tooling that wraps this library is the same
  barycentric quantity under a misleading name; the hard break to
  `lumped` / `galerkin` strings in `parseMassMatrixVariant()` is
  intentional and version-bumps any caller using the old names.

Lifetime contract: the binding holders (`ContextHolder` /
`ContextWrapper`) keep the operator structs and the owning
`ComputeContext` together, and no code path should call `unrequire*`
on the geometry while the views are alive. See the comment block
above `MeshOperators` in `include/nxr/compute.h`.

---

## MATLAB Reference Functions

Every C++ function here has a MATLAB reference in `../bioctree/toolbox/+bct/`.
When implementing or debugging, consult these files:

| C++ function | MATLAB reference |
|---|---|
| `assembleMeshOperators` | `+bct.+manifold.+operator.laplacebeltrami`, `+bct.+manifold.+operator.mass` |
| `assembleDECOperators` | `+bct.+manifold.+operator.dec` |
| `solveEigenmodes` | `+bct.+manifold.+eigen.solve` |
| `normalizeEigenmodes` | `+bct.+manifold.+eigen.normalize` |
| `removeDC` | `+bct.+manifold.+eigen.removeDC` |
| `solvePoisson` | `+bct.+manifold.+solve.poisson` |
| `hodgeDecompose` | Helmholtz/Hodge — α (gradient), β (curl), γ (harmonic) |
| `computeGeodesicDistance` | `+bct.+manifold.+solve.heatdistance` |
| `computeCurvatures` | `+bct.+manifold.+geometry.curvatures` |
| `computeDirectionField` | trivial connections (Crane / de Goes / Desbrun) |
| `whitneyInterpolate` / `scalarGradient` | `+bct.+field.+generate.gradient` |
| mesh health checks | `+bct.+manifold.+health.*` |

---

## C++ API Surface

The compute library API (callable from every binding) is declared in
`include/nxr/compute.h`. Highlights:

```cpp
// Per-mesh state
class ComputeContext { /* geometry-central mesh + geometry */ };
struct MeshOperators { cotanLaplacian, mass, vertexDualAreas, vertexNormals, totalArea, … };
struct DECOperators  { d0, d1, hodge0/1/2, hodge1Inverse };

MeshOperators assembleMeshOperators(ComputeContext&);
DECOperators  assembleDECOperators(ComputeContext&);

// Pre-factored Cholesky / LU cache (lazy)
class CholeskyCache {
public:
    static constexpr double kRegularization = 1e-8;
    // Primary signatures take raw sparse matrices (mesh- and graph-friendly).
    const Eigen::SimplicialLLT<…>& laplacian   (const Eigen::SparseMatrix<double>& K);
    const Eigen::SimplicialLLT<…>& hodgeExact  (const DECOperators&);  // mesh-only (DEC)
    const Eigen::SparseLU       <…>& hodgeCoExact(const DECOperators&);  // mesh-only (DEC)
    // Convenience overload: laplacian(MeshOperators&) forwards to laplacian(ops.cotanLaplacian).
};

// Spectral kernel — already agnostic in K, M.
EigenResult         solveEigenmodes(K, M, k, sigma=-1e-8, cancel={}, progress={});
Eigen::MatrixXd     normalizeEigenmodes(U, M);
EigenResult         removeDC(EigenResult);

// Convenience layer — agnostic primary signatures + MeshOperators overloads.
Eigen::VectorXd     solvePoisson(K, M, CholeskyCache&, densityMap);
Eigen::MatrixXf     generateHeatDiffusion(M, eig, u0, timesteps, alpha);
//   …and inline:    solvePoisson(MeshOperators&, …)             → forwards
//                   generateHeatDiffusion(MeshOperators&, …)     → forwards

// Mesh-only (need a halfedge surface — no graph analogue).
HodgeResult         hodgeDecompose(ComputeContext&, DECOperators&, CholeskyCache&, omega);
DirectionFieldResult computeDirectionField(ComputeContext&, DECOperators&, CholeskyCache&, singMap);
Eigen::VectorXd     computeGeodesicDistance(ComputeContext&, sourceVerts);
CurvatureResult     computeCurvatures(ComputeContext&);
…
```

The N-API bindings in `bindings/node/src/addon.cpp` hold a
`ContextHolder` per JS handle and plumb each call through. The holder
fields:

```cpp
struct ContextHolder {
    std::shared_ptr<ComputeContext>     ctx;
    std::shared_ptr<MeshOperators>      ops;        // view, lazy via ensureOps()
    std::shared_ptr<DECOperators>       dec;        // view, lazy via ensureDec()
    std::shared_ptr<CholeskyCache>      factors;    // pre-factored Cholesky/LU
    std::shared_ptr<EigenResult>        eigenmodes; // populated by EigenSolveWorker
    std::shared_ptr<VectorHeatSolver>   vhm;        // lazy via ensureVHM()
    std::shared_ptr<SignedHeatSolver>   shs;        // lazy via ensureSHS()
    std::shared_ptr<HeatGeodesicSolver> heatGeo;    // lazy via ensureHeatGeo()
    std::map<CLCacheKey,
        std::shared_ptr<ConnectionLaplacian>> connectionLaplacian;     // Pattern-C
    std::map<SmoothFieldKey, Eigen::MatrixXd>             smoothFaceFieldCache;   // Pattern-C
    std::map<SmoothFieldKey, SmoothVertexFieldResult>     smoothVertexFieldCache; // Pattern-C
};
```

The WASM `ContextWrapper` (`bindings/wasm/src/nxr_compute_wasm.cpp`)
holds an equivalent set. Both lazy-initialise solver PIMPLs on first
use; both cache stateless free-function results (connection-Laplacian,
smooth-direction-field) keyed by all input parameters.

The MEX binding (`bindings/mex/src/nxr_compute_mex.cpp`) is also
stateful: a file-scope `std::unordered_map<uint64_t, ContextHolder>`
is the MEX analogue of `ContextWrapper`, and the `uint64` handle
returned by `nxr_compute('create', V, F)` is the proxy-pointer
analogue (MEX has no Embind, so it exposes the context through the
single `nxr_compute('cmd', handle, …)` entry point). Dispatch is
**additive**: a `uint64` scalar in `prhs[1]` selects the handle path,
`V, F` arrays select the legacy stateless path (unchanged). The MEX
holder mirrors the same fields + lazy `ensure*` + result caches and
reaches full parity with the `ContextWrapper` surface. A bad/destroyed
handle throws `Error(InvalidHandle)` → `nxr:invalidHandle`. A MATLAB
`handle`-class wrapper is intentionally out of scope (application-side,
like the JS wrappers over WASM's `Manifold`); see
`docs/superpowers/specs/2026-05-27-stateful-mex-binding-design.md`.

---

## CholeskyCache — pre-factored Cholesky / LU

`CholeskyCache` exists because mesh-derived sparse factorizations cost
O(n^1.5) to build but only O(nnz) per back-substitution. Without
caching we re-paid that cost on every Hodge / Poisson / direction-field
call (and would re-pay it 60× per second on any future per-frame
pipeline). Both geometry-processing-js and geometry-central pre-factor
and reuse for the same reason.

**Contract.**

- One regularization constant: `CholeskyCache::kRegularization = 1e-8`,
  identical to the value the previous inline solvers used. There is no
  per-call `eps` argument — that would silently return a stale factor
  on a different `eps`.
- Each cached factor is built by the same matrix-assembly +
  factorization sequence the previous inline solvers used. Solves on
  the cached factor are bit-for-bit identical to fresh inline factor +
  solve. Verified by `test/test_cholesky_cache.cpp`.
- Lifetime: owned by `ContextHolder` in `addon.cpp`; the cache is
  destroyed when the JS handle releases (i.e. when the mesh changes).
- Single-threaded. The addon is invoked from Node's main thread; no
  mutex inside the cache.

**Not cached.** The eigensolver's `(K − σM)` factor lives inside
Spectra's `SymShiftInvert` and is discarded per call. Caching it would
require replacing Spectra's `ShiftInvertOp` with a custom one — out of
scope for v1.

**Perf options compatible with the modularity rule** (root CLAUDE.md
§10): AVX2 / AVX-512 compile flags (the build enables these on
x86_64), Eigen's separate `analyzePattern()` + `factorize()` for
matrices that share sparsity pattern, multi-RHS batched solves, and
GPU triangular back-substitution via TSL for hot per-frame paths.

**Not on the table:** CHOLMOD / SuiteSparse / MKL / PARDISO. They
would give a 3–10× win on x86 native but break the WASM and MEX
targets (`nxr_compute` must compile from the same source for all
three). The build saying "Building without SuiteSparse" is the desired
state, not a missed optimization.

---

## Implementation Rules

1. **Use geometry-central** for all operator assembly. Do NOT reimplement
   cotangent weights, DEC operators, or vertex areas.

2. **Use the CholeskyCache** for any solve against `L`, `d0ᵀ ★₁ d0`, or
   `d1 ★₁⁻¹ d1ᵀ`. Never inline a `SimplicialLLT::compute(...)` of those
   matrices in a new solver. If you need a new factor, add a slot to
   `CholeskyCache`.

   The `laplacian(K)` slot now takes a raw `const SparseMatrix<double>&`
   so it serves graph Laplacians as well as mesh stiffness. There is an
   inline `laplacian(const MeshOperators&)` overload that forwards to
   `laplacian(ops.cotanLaplacian)` — this preserves source compatibility
   for every existing call site.

3. **Spectra configuration**: Use `SymGEigsShiftSolver<…, ShiftInvert>`
   for the smallest eigenmodes. Always enforce real + symmetric on the
   stiffness matrix before passing it in (see `mesh_operators.cpp`).

4. **M-orthonormalisation** (`normalizeEigenmodes`): Implements the
   Gram-matrix whitening approach (Cholesky path + eigen fallback) from
   the MATLAB `normalize.m`. Critical for clustered eigenvalues.

5. **Remove DC mode** after solving — see `removeDC.m` reference.

6. **Error handling**: nxr-compute throws `nxr::compute::Error` (carrying an
   `ErrorCode` enum + message + optional hint) on every failure path.
   `Error` derives from `std::runtime_error` so existing
   `catch (const std::exception&)` keeps working, but new code should
   catch `Error` and switch on `code()`. Bindings translate
   `code()` per their idiom — see §11 below.  Don't catch and silently
   zero out — failing loud is the contract.

7. **Thread safety / async**: Three N-API addon methods are wrapped
   in `Napi::AsyncWorker` so they don't block the event loop:
   - `EigenSolveWorker` for `solveEigenmodes`
   - `HodgeDecomposeWorker` for `hodgeDecompose`
   - `DirectionFieldWorker` for `computeDirectionField`

   All three return Promises (the JS shim's `solve.eigen`,
   `solve.hodge`, `interpolate.directionField` are `async`). Inputs
   are deep-copied into the worker so the caller can release the
   originating TypedArrays immediately. Smooth-field methods are
   sync — the Pattern-C cache (rule 9) makes warm calls instant.
   Other heavy ops (Poisson, etc.) run synchronously; promote to
   AsyncWorker only if profiling shows they freeze the UI on real
   meshes. The WASM binding has no AsyncWorker analogue — consumers
   that need off-thread WASM dispatch must host the module in a
   Web Worker (see `docs/wasm-web-worker.md` for the recipe +
   `bindings/wasm/js/nxr_compute_wasm.worker.template.mjs`).

8. **Solver-instance caching for stateful geometry-central solvers.**
   Whenever you bind one of geometry-central's `*Solver` classes
   (`HeatMethodDistanceSolver`, `VectorHeatMethodSolver`,
   `SignedHeatSolver`), wrap it in a PIMPL class on the nxr-compute
   side and hold the wrapper alive on `ContextWrapper` (or the addon's
   `ContextHolder`) via an `ensureXxx()` helper. **Never construct the
   solver inline per call** — that's a real bug we caught and fixed
   (`computeGeodesicDistance`, commit `01a212a`). The constructor
   pre-factors the Cholesky systems; subsequent method calls
   back-substitute only. Throwing the solver away every call is
   the perf gap. See `docs/integration-lessons.md` §A.

9. **Result-level cache for stateless geometry-central free functions.**
   Some geometry-central algorithms (e.g.
   `computeSmoothest{,Boundary}AlignedFaceDirectionField`) are pure
   free functions with no Solver class — each call rebuilds and
   factorizes internally. Adding a Solver wrapper here would require
   reimplementing the algorithm. Instead, cache the **output** at the
   binding level, keyed by all input parameters. Both the N-API
   addon and the WASM `ContextWrapper` carry parallel slots:
   `smoothFaceFieldCache` / `smoothVertexFieldCache` (keyed by
   `(nSym, alignToCurvature)`) and `connectionLaplacian` (keyed by
   `(domain, nSym, regularization, format)`). The trade-off:
   parameter changes pay the full cold cost once per new key,
   identical-parameter repeats are cache-hit fast. See
   `docs/integration-lessons.md` §C.

10. **Eigensolve K ceiling on browser/WASM.** `solveEigenmodes` throws
    `EigensolveInvalidK` for `k > 1000`. Spectra's Krylov basis size
    `ncv = 2k+1` (capped at n) approaches an n×n dense matrix at
    large k; on a 10K-vertex mesh, k=5000 OOMs against the WASM 2 GB
    cap. The cap applies regardless of binding for predictability —
    native consumers (addon, MEX) inherit the same ceiling even
    though they have memory headroom. See `docs/eigensolve-cap.md`
    for the path to lift the cap.

11. **Bench-validate every solver-pattern change.** The downstream
    consumer
    `nxr-design-system/apps/galleries/gallery-mesh-tests/public/probes/bench-*.html`
    is the safety net for caching-pattern regressions. Run
    `npm run bench:all && npm run bench:diff` against
    `bench/baselines.json` after any change touching a method's
    solver pattern, factor caching, or memory layout. The verdict
    column in `bench/REPORT.md` flags methods where warm calls
    don't amortise (`❌ no cache`) — any new `❌` is a signal that
    rules 8-10 weren't followed.

---

## Common pitfalls — quick reference

Read `docs/integration-lessons.md` for the full retrospective from
the May-2026 bench round. Quick callouts that bit us already:

- **Inline solver construction.** `HeatMethodDistanceSolver(geom)` /
  `VectorHeatMethodSolver(geom)` / `SignedHeatSolver(geom)` MUST be
  built once and held alive. Constructing per call discards both
  Cholesky factors every time. Pattern A in
  `docs/integration-lessons.md`.
- **Stateless free-function results.** geometry-central's smooth-
  direction-field functions don't have a Solver class. Cache the
  OUTPUT in the binding, keyed by parameters. Don't try to extract
  the internal factor. Pattern C in `docs/integration-lessons.md`.
- **Spectra shift-invert factor is per-call by design.** Methods
  that bundle an eigensolve (e.g. `precompute({k})`) will look like
  they're "no caching" in bench output even when everything else IS
  cached. Document this on the new method, don't try to fix it.
- **Embind exception `.message`** is wired correctly as of commit
  `a0cbabe`: `getExceptionMessage` is in
  `bindings/wasm/CMakeLists.txt`'s `EXPORTED_RUNTIME_METHODS`, so
  JS consumers catching nxr-compute exceptions see
  `[CODE_NAME] msg | hint: …` instead of `[object Object]`.
- **emscripten 5.x needs Python 3.10+.** macOS default
  `/Library/Developer/CommandLineTools/usr/bin/python3` is 3.9 and
  fails the assertion. Prefix `PATH="/opt/homebrew/bin:$PATH"` (or
  set `EMSDK_PYTHON`) before invoking the WASM build.

---

## Build & Test

```sh
bash scripts/build.sh Release
# Outputs (build/ at repo root):
#   build/Release/nxr_compute_addon.node          (N-API; also copied to repo root)
#   build/Release/nxr_compute.exe                 (CLI smoke harness)
#   build/Release/nxr_compute.mexw64              (MATLAB MEX)
#   build/Release/test_eigen.exe                  (end-to-end smoke, 14 tests)
#   build/Release/test_cholesky_cache.exe         (cache contract canary)
#   build/Release/test_mass_variants.exe          (Lumped / Galerkin)
#   build/Release/test_connection_laplacian.exe   (vertex/face/edge CL)
#   build/Release/test_field_generators.exe       (eigenmode / heat / wave)
#   build/Release/test_graph_agnostic.exe         (K/M-agnostic solvers on graphs)
#   build/Release/test_geometry_central_extras.exe(GC solver wrappers)
#   build/Release/test_visualization_primitives.exe(isolines, streamlines, BFF)
#   build/Release/test_cancellation.exe           (CancellationToken)
#   build/Release/test_progress.exe               (ProgressObserver)
```

WASM build (separate toolchain — emscripten):

```sh
bash scripts/build-wasm.sh
node scripts/_smoke-wasm.mjs   # Embind round-trip, ~10 ms on icosahedron
```

Run the standalone native tests directly:

```sh
./build/Release/test_eigen.exe
./build/Release/test_cholesky_cache.exe
./build/Release/test_cancellation.exe
./build/Release/test_progress.exe
```

`test_cholesky_cache.cpp` is the canary for the bit-for-bit cache
contract. Any change to `CholeskyCache` or the solvers that consume
it must keep this test passing. `test_cancellation.cpp` and
`test_progress.cpp` guard the §12 cancel/progress contract.
`test_mass_variants.cpp` validates that both mass variants return the
same GC-sourced `vertexDualAreas` and produce distinct λ₁ spectra
(a guard against accidental aliasing — the historical Voronoi/
Barycentric pair were bit-identical for the same reason).

`deps/` carries `geometry-central` (operator assembly) and
`polyscope` (debug-only viewer used during native development; not
shipped in the addon).

---

## §11 Storage convention (hard rule)

Every JS-facing binding shell (N-API addon, WASM/Embind) must
honour this layout when flattening Eigen matrices to typed arrays:

| Eigen shape | Flat layout | Why |
|---|---|---|
| `[V × 3]` / `[F × 3]` / `[N × 3]` | row-major: `xyz xyz xyz …` | three.js BufferAttribute, NumPy `.reshape(-1, 3)`. |
| `[V × K]` eigenvectors | row-major (vMajor): `U[v*K + k]` | Cortical-flow's Zarr schema (`manifold/eigenmodes/eigenvectors` shape `[V, K]`) and the GPU spectral-synthesis access pattern. |
| `[T × V]` activity time-series | row-major: frame at `data[t*V .. (t+1)*V]` | Zarr `recordings/.../activity` schema. |
| Sparse | COO `{ row, col, data, rows, cols }` | Existing convention; both bindings honour it. |

**MEX is exempt**: a `V × K` Eigen matrix becomes a real MATLAB matrix
(column-major in MATLAB's native storage), so MATLAB users get
`U(:,k)` for mode k contiguously. The flatten rule applies only when
crossing into a JS typed array.

**Internal C++ storage is mostly column-major** — `Eigen::MatrixXd`
keeps its default column-major layout for most outputs (eigenvectors,
principal directions, isolines), and the binding helpers transpose
at the flatten step via an `Eigen::Map<RowMajor>` assignment that
Eigen vectorises into a SIMD memcpy + transpose. The one exception
is `MeshOperators::vertexNormals`, declared row-major via the
`VertexNormalsMatrix` alias (`Matrix<double, Dynamic, 3, RowMajor>`)
because it's consumed row-wise only and the binding-side memcpy
becomes a direct copy. Helper signatures (`matrixToFloat64Array`,
`eigenMatrixToVal`) are templated on `Eigen::MatrixBase<Derived>`
so both layouts share one code path.

## §12 Cancellation and progress contract

Long-running nxr-compute operations accept two optional parameters:

```cpp
EigenResult solveEigenmodes(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int k,
    double sigma                          = -1e-8,
    const nxr::compute::CancellationToken& cancel  = {},   // never cancelled
    const nxr::compute::ProgressObserver& progress = {});  // discard updates
```

nxr-compute is binding-agnostic: every shell builds `CancellationToken` from
its own native cancel mechanism, but the solver sees one type.

| Binding | CancellationToken construction | ProgressObserver wiring |
|---|---|---|
| **N-API addon** | `CancellationToken(reinterpret_cast<atomic<int32_t>*>(int32Arr.Data()))` where `int32Arr` is a SAB-backed `Int32Array`. | Same array kind; layout `[iter, totalIter, residual×1e6]`. |
| **WASM/Embind** | `CancellationToken(reinterpret_cast<atomic<int32_t>*>(cancelAddr))` where `cancelAddr` is a wasm-heap pointer (`Module._malloc`). | Same; pointer to a 3×int32 region in the heap. |
| **MEX** | `CancellationToken([](){ return utIsInterruptPending(); })` — bridges MATLAB Ctrl-C. | Deferred to Phase B (synchronous MATLAB has no natural progress UI surface). |
| **CLI** | `CancellationToken(&g_sigintFlag)` — flipped by a `SIGINT` handler. | Not wired (no UI surface). |

Bindings translate `Error` per their idiom:

| Binding | Error surface |
|---|---|
| N-API addon | JS `Error` with `.code` (string-named enumerator, e.g. `"CANCELLED"`) and `.hint`. |
| WASM | JS `Error` whose `.message` is `"[CODE_NAME] message [\| hint: ...]"`. `getExceptionMessage` is in `EXPORTED_RUNTIME_METHODS` as of commit `a0cbabe`, so the `.message` is actually populated. Phase B may add a richer mapping via Embind exception registration. |
| MEX | `MException` with identifier `nxr-compute:cancelled`, `nxr-compute:nonManifold`, etc. (`toMatlabIdentifier` turns the enumerator name into camelCase). |
| CLI | Exit code: `130` for `Cancelled` (POSIX 128+SIGINT), `1` otherwise. |

The cancellation poll point inside nxr-compute is once per Spectra
`perform_op` call, giving sub-second cancel latency on
cortical-sized meshes. The wrapper that drives this is
`CancelProgressOp` in `src/eigensolver.cpp`. Only `solveEigenmodes`
plumbs cancel + progress through today; `hodgeDecompose` and
`computeDirectionField` inherit the C++ contract but the addon's
worker classes (`HodgeDecomposeWorker`, `DirectionFieldWorker`)
don't expose cancel/progress slots yet — those calls complete in
1–3 s on cortical meshes and the async wrapping alone keeps the
UI thread responsive.

---

## Testing Against MATLAB

For any numerical function, create a test that:

1. Loads a known mesh (icosphere or cortical mesh from test fixtures).
2. Runs the C++ function.
3. Compares output to pre-saved MATLAB results (stored as .zarr or .mat).
4. Asserts max absolute error < 1e-10 for float64 operations.

The MATLAB MCP server (`reference_matlab_mcp` in personal memory) can
be used to generate fixtures from the `+bct` toolbox. Test fixtures
should live in `test/fixtures/` once the harness exists. As of
2026-04-27 there is **no** MATLAB-oracle test harness; this is the
single biggest gap in numerical confidence and the next step in the
correctness program.
