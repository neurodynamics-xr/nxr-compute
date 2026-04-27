# CLAUDE.md — Native C++ Addon (cxf_addon.node)

Read the root `CLAUDE.md` first. This file contains C++-specific
instructions for the native addon.

---

## Overview

The native addon (`cxf_addon.node`) is a thin N-API wrapper around
the `cortical_compute` static library. The library lives in
`native/src/` and `native/include/cortical_compute.h`; the N-API
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
declared in `native/include/cortical_compute.h`. Highlights:

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
    const Eigen::SimplicialLLT<…>& laplacian   (const MeshOperators&);
    const Eigen::SimplicialLLT<…>& hodgeExact  (const DECOperators&);
    const Eigen::SparseLU       <…>& hodgeCoExact(const DECOperators&);
};

// Solvers (each takes the cache and reuses factors across calls)
EigenResult         solveEigenmodes(K, M, k, sigma=-1e-8);
Eigen::MatrixXd     normalizeEigenmodes(U, M);
EigenResult         removeDC(EigenResult);
Eigen::VectorXd     solvePoisson(MeshOperators&, CholeskyCache&, densityMap);
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
targets (`cortical_compute` must compile from the same source for all
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

3. **Spectra configuration**: Use `SymGEigsShiftSolver<…, ShiftInvert>`
   for the smallest eigenmodes. Always enforce real + symmetric on the
   stiffness matrix before passing it in (see `mesh_operators.cpp`).

4. **M-orthonormalisation** (`normalizeEigenmodes`): Implements the
   Gram-matrix whitening approach (Cholesky path + eigen fallback) from
   the MATLAB `normalize.m`. Critical for clustered eigenvalues.

5. **Remove DC mode** after solving — see `removeDC.m` reference.

6. **Error handling**: `cortical_compute` throws `std::runtime_error` on
   factorization failure. The addon is built with `NAPI_CPP_EXCEPTIONS`,
   so node-addon-api converts these into JS errors. Don't catch and
   silently zero out — failing loud is the contract.

7. **Thread safety / async**: The eigensolver is wrapped in
   `Napi::AsyncWorker` (see `EigenSolveWorker` in `addon.cpp`) so it
   doesn't block the event loop. Other heavy ops (Hodge, Poisson) run
   synchronously today; promote to AsyncWorker only if profiling shows
   they freeze the UI on real meshes.

---

## Build & Test

```sh
bash scripts/build-native.sh Release
# Outputs:
#   native/build_node/Release/cxf_addon.node  (copied to project root)
#   native/build_node/Release/test_eigen.exe
#   native/build_node/Release/test_factor_cache.exe
```

Run the standalone tests directly:

```sh
./native/build_node/Release/test_eigen.exe          # 14 tests, end-to-end
./native/build_node/Release/test_factor_cache.exe   # cache contract
```

`test_factor_cache.cpp` is the canary for the bit-for-bit cache contract.
Any change to `CholeskyCache` or the solvers that consume it must keep
this test passing.

`native/deps/` carries `geometry-central` (operator assembly) and
`polyscope` (debug-only viewer used during native development; not
shipped in the addon).

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
