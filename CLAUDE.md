# CLAUDE.md — Native C++ Addon (nxr_compute_addon.node)

Read the root `CLAUDE.md` first. This file contains C++-specific
instructions for the native addon.

---

## Overview

The native addon (`nxr_compute_addon.node`) is a thin N-API wrapper around
the `nxr_compute` static library. The library lives in
`native/src/` and `native/include/nxr/compute.h`; the N-API
bindings live in `native/src/addon.cpp`. Build with CMake via cmake-js.

The addon is **context-based**: each mesh is bound once via
`createContext(vertices, faces)` and the resulting opaque handle owns
its `MeshOperators`, `DECOperators`, and `CholeskyCache` for the rest of
its lifetime. There are no JS-side handles for individual factors —
the cache is internal to the C++ side.

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

The compute library API (callable from `addon.cpp` and from CLI) is
declared in `native/include/nxr/compute.h`. Highlights:

```cpp
// Per-mesh state
class ComputeContext { /* geometry-central mesh + geometry */ };
struct MeshOperators { stiffness, mass, vertexAreas, normals, totalArea, … };
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
    // Convenience overload: laplacian(MeshOperators&) forwards to laplacian(ops.stiffness).
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

The N-API bindings in `addon.cpp` expose these as `native:*` IPC
methods (see `electron/CLAUDE.md` for the IPC list). The bindings
hold the `ContextHolder { ctx, ops, dec, factors }` per JS handle and
plumb each call through.

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
  solve. Verified by `native/test/test_factor_cache.cpp`.
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
   `laplacian(ops.stiffness)` — this preserves source compatibility for
   every existing call site.

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

7. **Thread safety / async**: The eigensolver is wrapped in
   `Napi::AsyncWorker` (see `EigenSolveWorker` in `addon.cpp`) so it
   doesn't block the event loop. Other heavy ops (Hodge, Poisson) run
   synchronously today; promote to AsyncWorker only if profiling shows
   they freeze the UI on real meshes.

---

## Build & Test

```sh
bash scripts/build-native.sh Release
# Outputs (cmake-js flat layout):
#   native/build_node/Release/nxr_compute_addon.node          (copied to project root)
#   native/build_node/Release/nxr_compute.exe                 (CLI)
#   native/build_node/Release/nxr_compute.mexw64              (MATLAB MEX)
#   native/build_node/Release/test_eigen.exe          (end-to-end smoke)
#   native/build_node/Release/test_cholesky_cache.exe (cache contract)
#   native/build_node/Release/test_field_generators.exe
#   native/build_node/Release/test_visualization_primitives.exe
#   native/build_node/Release/test_cancellation.exe   (Phase A)
#   native/build_node/Release/test_progress.exe       (Phase A)
```

WASM build (separate toolchain):

```sh
emcmake cmake -B native/build_wasm -S native -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build_wasm --target nxr_compute_wasm
node scripts/_smoke-wasm.mjs   # Embind round-trip, ~10 ms on icosahedron
```

Run the standalone native tests directly:

```sh
./native/build_node/Release/test_eigen.exe          # 14 tests, end-to-end
./native/build_node/Release/test_cholesky_cache.exe # cache contract
./native/build_node/Release/test_cancellation.exe   # CancellationToken
./native/build_node/Release/test_progress.exe       # ProgressObserver
```

`test_cholesky_cache.cpp` is the canary for the bit-for-bit cache contract.
Any change to `CholeskyCache` or the solvers that consume it must keep
this test passing. `test_cancellation.cpp` and `test_progress.cpp`
guard the §12 cancel/progress contract.

`native/deps/` carries `geometry-central` (operator assembly) and
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

**Internal C++ storage is unchanged** — `Eigen::MatrixXd` keeps its
default column-major layout. The bindings transpose at the flatten
step (e.g. via `Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>
rowMajor = m;` in `eigenMatrixToVal`).

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
| WASM | JS `Error` whose `.message` is `"[CODE_NAME] message [\| hint: ...]"`. Phase B will add a richer mapping via Embind exception registration. |
| MEX | `MException` with identifier `nxr-compute:cancelled`, `nxr-compute:nonManifold`, etc. (`toMatlabIdentifier` turns the enumerator name into camelCase). |
| CLI | Exit code: `130` for `Cancelled` (POSIX 128+SIGINT), `1` otherwise. |

The cancellation poll point inside nxr-compute is once per Spectra
`perform_op` call, giving sub-second cancel latency on
cortical-sized meshes. The wrapper that drives this is
`CancelProgressOp` in `nxr-compute/src/eigensolver.cpp`.

---

## Testing Against MATLAB

For any numerical function, create a test that:

1. Loads a known mesh (icosphere or cortical mesh from test fixtures).
2. Runs the C++ function.
3. Compares output to pre-saved MATLAB results (stored as .zarr or .mat).
4. Asserts max absolute error < 1e-10 for float64 operations.

The MATLAB MCP server (`reference_matlab_mcp` in personal memory) can
be used to generate fixtures from the `+bct` toolbox. Test fixtures
should live in `native/test/fixtures/` once the harness exists. As of
2026-04-27 there is **no** MATLAB-oracle test harness; this is the
single biggest gap in numerical confidence and the next step in the
correctness program.
