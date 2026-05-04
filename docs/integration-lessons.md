# Integration lessons — caching patterns + perf gotchas

A retrospective written after the May-2026 bench round on the
`@nxr/charts-mesh` consumer (the WASM binding's primary downstream).
Identifies the architectural shapes geometry-central exposes, the
caching pattern each one demands from us, and the symptoms / detection
methods for getting it wrong.

> **Audience:** anyone adding a new method to the C++ surface or a
> new method binding to `bindings/wasm/src/nxr_compute_wasm.cpp`.
> Read this BEFORE writing the code, not after.

> **Empirical baseline:** the bench harness in
> `nxr-design-system/apps/galleries/gallery-mesh-tests/public/probes/bench-*.html`
> exercised every public method × {small_bunny 1.4K V, cowhead 4.5K V,
> fsaverage5 10K V, bunny 14K V} × cold/warm. Numbers cited here are
> warm p95 unless noted. Pre-fix and post-fix snapshots live in
> `nxr-design-system/bench/REPORT-DIFF.md`.

## The three caching patterns geometry-central forces on us

geometry-central exposes mesh-processing algorithms in three distinct
architectural shapes. Each demands a different caching pattern from
the WASM binding. Mismatch is the #1 source of the
`computeGeodesicDistance`-shaped perf bug we caught.

### Pattern A — solver class with stateful factor

Examples in geometry-central:
- `surface::HeatMethodDistanceSolver` (geodesic distance via heat method)
- `surface::VectorHeatMethodSolver` (parallel transport, scalar
   extension, log map, Karcher mean)
- `surface::SignedHeatSolver` (signed distance from curve)

**Shape**: the constructor pre-factors one or more Cholesky systems
(typically `M + tL` and `L`); subsequent method calls back-substitute
only.

**Rule**: wrap in a PIMPL class on the nxr-compute side
(`nxr::compute::HeatGeodesicSolver`, `VectorHeatSolver`, `SignedHeatSolver`),
and hold the wrapper alive on `ContextWrapper` via `ensureXxx()`.
**Never construct inline per call.**

This is the fix that landed in commit `01a212a`
(`HeatGeodesicSolver`). Before the fix, `geodesic.cpp` had:

```cpp
// WRONG — throws away the factor every call
HeatMethodDistanceSolver solver(geometry);
auto distances = solver.computeDistance(sources);
```

After:

```cpp
// solver kept alive on ContextWrapper via ensureHeatGeo()
auto distances = s.solver.computeDistance(sources);
```

Bench impact on bunny (14K V): `computeGeodesicDistance` warm dropped
from 39 ms → 1.9 ms (23× speedup). Cold call still pays the factor
once; that's correct.

### Pattern B — free function taking `CholeskyCache`

Examples in nxr-compute:
- `solvePoisson(K, M, CholeskyCache&, ...)`
- `hodgeDecompose(ctx, dec, CholeskyCache&, omega)`
- `computeDirectionField(ctx, dec, CholeskyCache&, singularities)`

**Shape**: we built our own factor cache because geometry-central
doesn't expose one for these. Each function takes a
`CholeskyCache&` and pulls the relevant factor (`laplacian`,
`hodgeExact`, `hodgeCoExact`) on demand.

**Rule**: use the existing slots in `CholeskyCache`. If you need a
new factor for a new method, **add a slot** to `CholeskyCache` —
don't inline `SimplicialLLT::compute(...)` of a cacheable matrix.

**Sharing**: `cache.hodgeExact` is shared between `hodgeDecompose`
and `computeDirectionField`. Calling either method warms the factor
for the other — the bench cross-warmup probe confirms this works.

### Pattern C — stateless free function (no cache parameter)

Examples in geometry-central:
- `computeSmoothestFaceDirectionField(geom, nSym)`
- `computeSmoothestBoundaryAlignedFaceDirectionField(geom, nSym)`
- `computeCurvatureAlignedFaceDirectionField(geom, nSym)`
- BFF parametrization (`computeBoundaryFirstFlattening`)

**Shape**: pure functions taking `geometry` + parameters, returning
a result. No solver state, no cache parameter. **Each call rebuilds
the connection-Laplacian and factorizes it.**

**Rule**: cache the **output** at the binding level, keyed by all
input parameters. **Don't** try to add a Solver class without
reimplementing the algorithm — that's a deep undertaking and the
geometry-central API doesn't lend itself to factor extraction.

This is the fix that landed in commit `6313bc7`. Result-level cache
keyed by `(nSym, alignToCurvature)` in `ContextWrapper`:

```cpp
std::map<std::pair<int, bool>, Eigen::MatrixXd> smoothFaceFieldCache_;
```

Bench impact on bunny: `computeSmoothFaceField` warm dropped from
277 ms → 0.2 ms on cache hits (1385× speedup). The trade-off is that
parameter changes (e.g. `nSym=4 → nSym=2`) still pay the full cold
cost once per new key, which is acceptable for realistic gallery
usage where users hold parameters constant during a session.

**Caveat — non-determinism across solver instances**: `computeSmoothFaceField`
results have a representative-rotation ambiguity for nRosy fields (the
underlying field is identical, but the picked representative direction
can differ by a full symmetry-class rotation between independent solver
runs). When validating the cache, compare WITHIN a single ctx (cache
hit must return byte-identical output); don't assert equality across
independent ctx instances.

### How to tell which pattern an algorithm is

Quick test: read the geometry-central declaration.

| Looks like | Pattern |
|---|---|
| `class XxxSolver { public: XxxSolver(geom, …); result method(…); }` | A — instance-cache via `ensureXxx()` |
| `result xxx(geom, …)` (free fn, no solver class) | C — result-level cache |
| nxr-compute already has it as `xxx(K, M, CholeskyCache&, …)` | B — extend `CholeskyCache` |

If you're unsure, check the geometry-central header in
`deps/geometrycentral/include/geometrycentral/surface/`.

## Common pitfalls

### Constructing a solver inline per call

Symptom: cold/warm timing essentially identical for a method that
SHOULD amortise.

Detection: bench harness prints `[some_module] …` C++ stdout that
includes timing on first call; if the message repeats every call, the
factor is being rebuilt.

Fix: see Pattern A above.

### Inline Cholesky factor in a new solver

Symptom: same as above, but for matrices we already cache.

Example anti-pattern:
```cpp
// WRONG — should use cache.laplacian(K) instead
SimplicialLLT<SparseMatrix<double>> llt(K + epsI);
```

Fix: add a slot to `CholeskyCache` if needed; otherwise use
`cache.laplacian(K)`.

### Returning sparse matrices via `convertJSArrayToNumberVector` round-trip

The embind path `convertJSArrayToNumberVector<double>(jsArr)` does a
full copy out of the wasm heap. For dense matrices this is fine.
For sparse matrices, **flatten to COO** in the binding side and
return the `{row, col, data, rows, cols, nnz}` object. Don't try to
return raw `Eigen::SparseMatrix` — embind doesn't know how.

### Eigensolve K ceiling on browser WASM

Spectra's `SymGEigsShiftSolver` allocates a Krylov basis of size
`ncv = max(2k+1, 20)` clamped at `n`. Each Krylov vector is `n`
doubles. On a 10K-vertex cortical mesh:

| k    | ncv  | basis size | total memory inc. eigenvectors |
|-----:|-----:|-----------:|-------------------------------:|
|   10 |   21 | 1.7 MB     | ~5 MB                           |
|  100 |  201 | 16 MB      | ~30 MB                          |
| 1000 | 2001 | 164 MB     | ~250 MB                         |
| 5000 | 10001 ≈ n | 820 MB | ~1.3 GB (hits 2 GB WASM cap)   |

**Cap**: `solveEigenmodes` throws `EigensolveInvalidK` when k > 1000.
See `docs/eigensolve-cap.md` for the path to lift the cap (memory64
WASM, server-side fallback).

### Spectra shift-invert factor is per-call

`Spectra::SymShiftInvert` builds the `(K - σM)` factor inside
`solveEigenmodes()`. **Not cacheable** across calls without replacing
Spectra's `ShiftInvertOp` with a custom one — out of scope.

This means `precompute({k})` and similar bundle paths look like
they're "no caching" in the bench output even though everything
else IS cached. That's by design, not a bug. Document this
behaviour in any new bundled method that includes an eigensolve.

### Embind exception messages don't reach JS

`nxr::compute::Error(SomeCode, "msg", "hint")` is rethrown as
`std::runtime_error` in the WASM binding (see the catch block in
`solveEigenmodes`). When the runtime_error crosses to JS, it becomes
an embind `CppException` object whose `.message` is empty. JS sees
`[object Object]` instead of `[CODE_NAME] msg | hint: …`.

The downstream `parseNxrComputeError` in nxr-design-system attempts a
heap-walk via `e.excPtr` to recover the message, but it depends on
`getExceptionMessage` and `UTF8ToString` being in the WASM build's
`EXPORTED_RUNTIME_METHODS`. As of this writing, those exports are
missing from `bindings/wasm/CMakeLists.txt` — fix is a one-line
change to the linker flags. Until then, `errorCode` decodes as
`UNKNOWN` for all WASM consumers; the throw-vs-no-throw behaviour is
intact, only the structured code is missing.

Tracked as cleanup work in
`nxr-design-system/docs/dev/bench-roadmap.md`.

### Build environment — Python version mismatch

emscripten 5.x requires Python 3.10+. macOS ships `python3 = 3.9` by
default at `/Library/Developer/CommandLineTools/usr/bin/python3`,
which appears first in `PATH` for non-shell-snapshot Bash invocations.
Symptom:

```
emscripten requires python 3.10 or above
(/Library/Developer/CommandLineTools/usr/bin/python3 3.9.6 ...)
```

Fix: prefix the build command with `PATH="/opt/homebrew/bin:$PATH"`
(or wherever `python3.x` ≥ 3.10 lives):

```sh
PATH="/opt/homebrew/bin:$PATH" cmake --build build_wasm --target nxr_compute_wasm
```

## Detection: the bench harness as safety net

Every change to nxr-compute that touches a method's solver pattern,
factor caching, or memory layout should be validated against the
bench harness in nxr-design-system:

```sh
# In nxr-design-system:
npm run bench:all       # runs every probe
npm run bench:report    # renders bench/REPORT.md
npm run bench:diff      # before/after vs the committed baselines.json
npm run bench:promote   # update baselines.json after a clean run
```

The verdict column in `bench/REPORT.md` flags methods where warm
calls don't amortise (`❌ no cache`). Any new `❌` is a signal that
the patterns above weren't followed correctly.

The cross-warmup probe (`bench-cross-warmup.html`) drives a
realistic 7-step interactive session sequence per mesh and records
each step as one cell — useful for detecting whether cross-method
factor sharing (e.g. hodge → directionField via shared
`cache.hodgeExact`) actually amortises in real flow.

## Phase 3.5 record (May 2026)

Three fixes landed in nxr-compute against the patterns above:

| Commit | Fix | Pattern | Impact (bunny warm p95) |
|---|---|---|---|
| `01a212a` | Add `class HeatGeodesicSolver`, refactor `computeGeodesicDistance` to take it | A → was inline per call | 39 ms → 1.9 ms (23×) |
| `6313bc7` | Result-level cache in `ContextWrapper` for `computeSmoothFaceField` / `computeSmoothVertexField` | C → cache output | 277 ms → 0.2 ms (1385×) |
| `add9475` | Cap k ≤ 1000 in `solveEigenmodes` with `EigensolveInvalidK` | n/a — guardrail | k=5000 OOM (384 s) → instant error |

Each fix shipped with a corresponding correctness probe in
nxr-design-system that asserts the cached path returns identical
results to the uncached path:

- `bench-test-geodesic-correctness.html` — bit-identical (max diff = 0)
- `bench-test-smooth-field-correctness.html` — byte-identical within ctx, per-key isolation verified
- `bench-test-eigensolve-cap.html` — k=999/1000 succeed, k=1001 throws

The full diff of pre-fix vs post-fix bench numbers is in
`nxr-design-system/bench/REPORT-DIFF.md`.
