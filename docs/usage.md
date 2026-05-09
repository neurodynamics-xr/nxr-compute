# nxr-compute — usage guide for downstream consumers

`nxr-compute` is a portable C++ math library for halfedge-mesh-based scientific
computing. It exposes a single C++ API to four bindings — N-API addon
(Node.js / Electron), WebAssembly (browsers, Web Workers), MATLAB MEX,
and a CLI — with no preferred consumer. This guide covers usage from
each binding, focusing on the Phase A additions: structured errors,
cancellation, and progress observation.

For the C++-side design rationale see `docs/nxr-compute-extensions.md` (the
original Phase A proposal) and `native/CLAUDE.md` (binding contract).

---

## 1. Storage convention

When nxr-compute returns a matrix, every JS-facing binding flattens it
identically:

| Eigen shape | Flat layout in JS | Indexing |
|---|---|---|
| `[V × 3]`, `[F × 3]`, `[N × 3]` | row-major: `xyz xyz xyz …` | `data[i*3 + axis]` |
| `[V × K]` eigenvectors | row-major (vMajor) | `data[v*K + k]` |
| `[T × V]` activity | row-major (frame-major) | `data[t*V + v]` |
| Sparse matrix | COO triplets | `{ row, col, data, rows, cols, nnz }` |

**Eigenvector layout is vMajor** (vertex-major, `U[v*K + k]`). This
matches the cortical-flow Zarr schema (`manifold/eigenmodes/eigenvectors`
shape `[V, K]`), and the GPU spectral-synthesis access pattern. To
extract mode `k` as a flat array of length `V`, do a strided read with
stride `K`:

```js
function modeK(eigenvectors, V, K, k) {
  const out = new Float64Array(V)
  for (let v = 0; v < V; v++) out[v] = eigenvectors[v * K + k]
  return out
}
```

**MEX is exempt**: a `V × K` Eigen matrix becomes a real MATLAB matrix
(column-major, MATLAB's native storage), so MATLAB users get
`U(:, k)` for mode `k` contiguously without any extra work.

---

## 2. Error handling

Every nxr-compute failure throws `nxr::compute::Error` carrying a stable
`ErrorCode` enumerator:

| Code | When |
|---|---|
| `INVALID_INPUT` | Bad argument (size mismatch, out-of-range index, etc.) |
| `NON_MANIFOLD` | Mesh has non-manifold edges/vertices |
| `OPEN_MESH_REQUIRED` | Closed mesh passed to e.g. BFF parametrization |
| `CLOSED_MESH_REQUIRED` | Open mesh passed where a closed one is needed |
| `GEOMETRY_DEGENERATE` | Cholesky / LU factorization failed |
| `EIGENSOLVE_INVALID_K` | `k < 1` or `k > N - 1` |
| `EIGENSOLVE_NOT_CONVERGED` | Spectra reported numerical issue |
| `NOT_PRECOMPUTED` | Lazy resource (e.g. eigenmodes) requested before compute |
| `CANCELLED` | User requested cancellation mid-solve |
| `OUT_OF_MEMORY` | Allocation failure (rare; usually fatal anyway) |
| `INTERNAL_ERROR` | Anything else |

Each binding surfaces the code in its native idiom — see §4 below.

---

## 3. Cancellation

Long-running nxr-compute operations (today: `solveEigenmodes`; Phase B will
extend the list) accept an optional cancellation token. The solver
polls it once per Spectra `perform_op` call (~ a few ms on
cortical-sized meshes), so cancel latency is sub-second.

The C++ token has two construction paths so every binding can plug
in its native cancel mechanism — atomic flag, polling function, etc.
JS uses the atomic-flag path; MATLAB uses Ctrl-C via
`utIsInterruptPending`; the CLI uses SIGINT.

When cancellation fires, nxr-compute throws `Error(Cancelled)`, which
each binding translates per §4.

---

## 4. Per-binding usage

### 4.1 JavaScript via WASM (`@neurodynamics-xr/nxr-compute`)

The package ships a prebuilt WASM in `dist/wasm/` and a shim at
`bindings/wasm/js/index.mjs`. Recommended path is the shim:

```js
import { initNxrCompute } from '@neurodynamics-xr/nxr-compute'
const nxrCompute = await initNxrCompute()
```

Or, if you need the raw Emscripten factory (e.g. to override `locateFile`
in an unusual asset-serving setup), import the artifact directly:

```js
import createNxrComputeModule from '@neurodynamics-xr/nxr-compute/wasm/nxr_compute.js'
const nxrCompute = await createNxrComputeModule()

const ctx = new nxrCompute.ComputeContext(verticesFloat64, facesInt32)

// ── Cancellation ────────────────────────────────────────────
//
// Allocate a 4-byte cancel slot in the wasm heap. Pass 0 to opt out.
// With SHARED_MEMORY enabled (and COOP/COEP headers on the page),
// you can flip this from another thread via Atomics.store.
const cancelPtr = nxrCompute.Module._malloc(4)
nxr-compute.Module.HEAP32[cancelPtr >> 2] = 0  // not cancelled

// ── Progress (optional) ─────────────────────────────────────
//
// 3 × int32: [iteration, totalIterations, residual × 1e6].
// Read from any thread that has access to the wasm heap.
const progressPtr = nxrCompute.Module._malloc(12)
for (let i = 0; i < 3; i++) nxr-compute.Module.HEAP32[progressPtr/4 + i] = 0

// Kick the solve (resolves with the eigenmodes, or throws on cancel).
let modes
try {
  modes = ctx.solveEigenmodes(
    /* k        */ 500,
    /* sigma    */ -1e-8,
    /* cancel   */ cancelPtr,
    /* progress */ progressPtr,
    /* progressLen */ 3,
  )
} catch (e) {
  // Phase A error format: "[CODE_NAME] message [| hint: …]"
  const m = e.message.match(/^\[([A-Z_]+)\]\s*(.*?)(?:\s*\|\s*hint:\s*(.*))?$/)
  const code    = m?.[1] ?? 'INTERNAL_ERROR'
  const message = m?.[2] ?? e.message
  const hint    = m?.[3]
  if (code === 'CANCELLED') {
    console.log('User cancelled.')
  } else {
    console.error(`nxr-compute [${code}] ${message}`, hint ? `(hint: ${hint})` : '')
  }
} finally {
  nxr-compute.Module._free(cancelPtr)
  nxr-compute.Module._free(progressPtr)
  ctx.delete()
}

// ── Cancel from elsewhere ──────────────────────────────────
// Single-threaded:
nxr-compute.Module.HEAP32[cancelPtr >> 2] = 1

// Cross-thread (requires SHARED_MEMORY=1 + COOP/COEP):
const view = new Int32Array(nxrCompute.Module.HEAPU8.buffer)
Atomics.store(view, cancelPtr >> 2, 1)
```

**Eigenvector access** (vMajor):

```js
const { eigenvectors, eigenvalues, k, nV } = modes
// eigenvectors is a Float64Array of length nV * k
// vertex v's value in mode i: eigenvectors[v * k + i]
```

### 4.2 MATLAB via MEX (`nxr_compute.mexw64`)

```matlab
% Build (one-time, from project root):
%   bash scripts/build.sh Release
% then add build/Release to your MATLAB path.

V = readSurface('lh.pial').vertices;   % V × 3 double
F = readSurface('lh.pial').faces;      % F × 3 (1-based)

% ── One-shot pipeline ──────────────────────────────────────
% Ctrl-C interrupts cleanly: nxrCompute throws MException 'nxr:cancelled'
% which you catch like any MATLAB error.
try
  result = nxr_compute('precompute', V, F, 1000);
catch ME
  switch ME.identifier
    case 'nxr:cancelled'
      fprintf('User cancelled (Ctrl-C).\n');
    case 'nxr:nonManifold'
      fprintf('Mesh has non-manifold elements: %s\n', ME.message);
    otherwise
      rethrow(ME);
  end
end

% ── Step-by-step ───────────────────────────────────────────
% Same Ctrl-C contract: each long-running call is interruptible.
ops = nxr_compute('assembleMeshOperators', V, F);
eig = nxr_compute('solveEigenmodes', ops.stiffness, ops.mass, 500);
eig.eigenvectors = nxr_compute('normalizeEigenmodes', eig.eigenvectors, ops.mass);
eig = nxr_compute('removeDC', eig);

% Mode k as a column vector (MATLAB-native column-major storage):
mode_k = eig.eigenvectors(:, k);
```

`MException` identifiers follow `nxr-compute:<camelCaseCode>`:
`nxr-compute:cancelled`, `nxr-compute:invalidInput`, `nxr-compute:nonManifold`,
`nxr-compute:openMeshRequired`, `nxr-compute:eigensolveInvalidK`,
`nxr-compute:geometryDegenerate`, etc.

Progress feedback during MATLAB MEX calls is deferred to Phase B —
synchronous MEX has no clean UI surface for live progress. If your
workflow needs it, run the call in `parfeval` and poll a shared
variable, or break the work into chunks (e.g. solve `k=200` modes
five times rather than `k=1000` once).

### 4.3 Electron / Node.js via N-API (`nxr_compute_addon.node`)

```ts
// Used internally by cortical-flow's renderer; the same contract
// works for any Node.js consumer that requires the addon.
const nxrCompute = require('./nxr_compute_addon.node')
const ctx = nxrCompute.createContext(verticesF64, facesI32)

// Cancel + progress slots, both SAB-backed so a UI thread can write
// to them while a worker holds the C++ solve.
const cancelArr = new Int32Array(new SharedArrayBuffer(4))
const progressArr = new Int32Array(new SharedArrayBuffer(12))   // 3 × int32

try {
  const modes = await nxr-compute.solveEigenmodes(ctx, 1000, cancelArr, progressArr)
  // modes.eigenvectors: Float64Array, vMajor [nV * K]
  // modes.eigenvalues:  Float64Array [K]
} catch (e) {
  // Addon attaches structured fields to the JS Error.
  if (e.code === 'CANCELLED') {
    console.log('Cancelled.')
  } else {
    console.error(`nxr-compute [${e.code}] ${e.message}`, e.hint && `(hint: ${e.hint})`)
  }
}

// Cancel from anywhere with access to cancelArr:
Atomics.store(cancelArr, 0, 1)

// UI thread can poll progress at any cadence:
const iter = Atomics.load(progressArr, 0)
const tot  = Atomics.load(progressArr, 1)
const res  = Atomics.load(progressArr, 2) / 1e6
console.log(`solved ${iter}/${tot} units, residual ≈ ${res.toExponential(2)}`)
```

`e.code` is the string-named enumerator (`'CANCELLED'`,
`'NON_MANIFOLD'`, …). `e.hint` carries an optional human-readable
remediation string when present.

### 4.4 CLI (`nxr-compute precompute`)

```sh
nxr-compute precompute --input lh.pial --output session.zarr --k 1000
# Ctrl-C cancels cleanly:
#   exit code 130 (= 128 + SIGINT)
#   message starts with [CANCELLED]
# Other failures: exit code 1 with [CODE] prefix.
```

---

## 5. Build & install

```sh
# All native bindings (addon + cli + mex) — needs Node.js + MATLAB:
bash scripts/build.sh Release
# Outputs at build/Release/:
#   nxr_compute_addon.node   (also copied to project root)
#   nxr_compute.exe          (CLI)
#   nxr_compute.mexw64       (MEX, if MATLAB detected at configure time)

# WASM (separate toolchain — needs emsdk):
bash scripts/build-wasm.sh Release
# Build outputs at build_wasm/, then refreshed into the committed
# prebuilt at dist/wasm/:
#   dist/wasm/nxr_compute.js     (factory function — loaded by the shim)
#   dist/wasm/nxr_compute.wasm   (binary)

# Smoke tests:
./build/Release/test_cancellation       # cancellation contract
./build/Release/test_progress           # progress observer
./build/Release/test_eigen              # end-to-end
node scripts/_smoke-wasm.mjs            # WASM round-trip
```

---

## 6. Graph signal processing

nxr-compute's spectral kernel is **agnostic in K and M** — `solveEigenmodes`,
`normalizeEigenmodes`, `removeDC`, `CholeskyCache::laplacian`,
`solvePoisson`, and `generateHeatDiffusion` all accept any sparse
SPD pair, not just mesh operators. A user with a graph adjacency `W`
and node weights can run the full spectral pipeline by assembling
the Laplacian themselves and feeding it through the existing API:

```cpp
// 1. User assembles graph operators (nxr-compute does NOT do this — keep it
//    a pure linear-algebra engine, no graph theory inside).
Eigen::SparseMatrix<double> W = loadAdjacency();          // [n, n]
Eigen::VectorXd d = W * Eigen::VectorXd::Ones(W.rows());  // degrees
Eigen::SparseMatrix<double> L = buildDegreeMinus(W, d);   // L = D - W
Eigen::SparseMatrix<double> M = identity(W.rows());       // node weights

// 2. Hand off to nxr-compute — the same API the mesh path uses.
auto eig = nxr::compute::solveEigenmodes(L, M, k);
eig.eigenvectors = nxr::compute::normalizeEigenmodes(eig.eigenvectors, M);
eig = nxr::compute::removeDC(eig);

nxr::compute::CholeskyCache cache;
auto phi  = nxr::compute::solvePoisson(L, M, cache, sources);   // same factor reused
auto heat = nxr::compute::generateHeatDiffusion(M, eig, u0, ts, alpha);
```

Spectral filtering of a graph signal `f` is three Eigen lines, no
nxr-compute API needed:

```cpp
Eigen::VectorXd c        = eig.eigenvectors.transpose() * (M * f);
Eigen::VectorXd cFilt    = c.cwiseProduct(weights);   // weights[k] = w(λ_k)
Eigen::VectorXd fFilt    = eig.eigenvectors * cFilt;
```

The same recipe works in JavaScript (via WASM) and MATLAB (via MEX) —
the eigenvector layout (`vMajor` in JS, native column-major in MATLAB)
follows §1.

**What nxr-compute does NOT do for graphs:** community detection, centrality,
modularity, BFS / Dijkstra, motif counting, magnetic / signed Laplacian
variants, polynomial filter banks (Chebyshev, spectral graph wavelets).
Those are downstream concerns — nxr-compute stays a linear-algebra engine. A
user wanting Chebyshev approximation of a spectral filter writes it
on top of the nxr-compute primitives in a few lines; nxr-compute doesn't ship it.

**What nxr-compute does NOT do that's surface-only:** Hodge decomposition,
Whitney interpolation, geodesics, streamlines, BFF, curvatures, face
frames. These need a 2-simplicial structure (faces + halfedge mesh)
that plain graphs don't have. nxr-compute's `ComputeContext` and DEC operators
exist precisely to provide that — they're scoped to mesh inputs.

A native test (`test_graph_agnostic.cpp`) exercises this pipeline
end-to-end on a synthetic 50-node path graph and verifies the
eigenvalues match the analytic formula `λₖ = 2(1 − cos(πk/n))`.

---

## 7. Phase A scope (current state)

Wired end-to-end:

- `solveEigenmodes` accepts cancellation + progress in every
  binding. Sub-second cancel latency on cortical-sized meshes.
- Structured errors via `nxr::compute::Error` + `ErrorCode` everywhere.
  Each binding surfaces `.code` natively.
- vMajor eigenvector layout is the canonical contract. WASM was
  flipped from column-major to match.

Deferred to later phases:

- Cancellation/progress on operations other than `solveEigenmodes`
  (Hodge, Poisson, geodesic distance, BFF). Same `CancellationToken`
  / `ProgressObserver` types — just add the parameter to the
  C++ signature and propagate.
- C++20 baseline (`std::span` parameters, designated initializers
  in option structs). Currently blocked by an Eigen 3.4 + Emscripten
  Clang template-deduction bug; needs an Eigen patch / upgrade.
- Richer JS error subclass for WASM (Embind exception registration)
  so `e.code` is a real property rather than a `[CODE]` message
  prefix. The current `[CODE]`-prefix format is documented and stable
  for the duration of Phase A.
- MATLAB live-progress surface (deferred — synchronous MEX has no
  natural progress UI without parfeval).
