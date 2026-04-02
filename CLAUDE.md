# CLAUDE.md — Native C++ Addon (bioctree.node)

Read the root `CLAUDE.md` first. This file contains C++-specific
instructions for the native addon.

---

## Overview

The native addon (`bioctree.node`) is a thin N-API wrapper around the
C++ mathematical core. It lives in `native/` and builds with CMake.

---

## MATLAB Reference Functions

Every C++ function here has a MATLAB reference in `../bioctree/toolbox/+bct/`.
When implementing or debugging, consult these files:

| C++ function | MATLAB reference |
|---|---|
| `assembleMeshOperators` | `+bct.+manifold.+operator.dec`, `+bct.+manifold.+operator.laplacebeltrami`, `+bct.+manifold.+operator.mass` |
| `solveEigenmodes` | `+bct.+manifold.+eigen.solve` |
| `normaliseEigenmodes` | `+bct.+manifold.+eigen.normalize` |
| `removeDC` | `+bct.+manifold.+eigen.removeDC` |
| `helmholtzDecompose` | `+bct.+manifold.+operator.divergence`, `+bct.+manifold.+operator.curl` |
| `applyOperator` | `+bct.+operators.apply` |
| `solveCholesky` (flow) | `+bct.+manifold.+solve.poisson` |
| `synthetic: heat` | `+bct.+manifold.+solve.heat`, `+bct.+field.+generate.heat` |
| `synthetic: wave` | `+bct.+field.+generate.heatadvection` |
| mesh health checks | `+bct.+manifold.+health.*` |

---

## N-API Function Signatures

```typescript
// TypeScript signatures — implemented in native/src/addon.cpp

assembleMeshOperators(
  vertices: Float64Array,   // [V×3] vertex positions
  faces: Int32Array         // [F×3] face indices
): {
  laplacian:  { indptr: Int32Array, indices: Int32Array, data: Float64Array },
  mass:       { indptr: Int32Array, indices: Int32Array, data: Float64Array },
  d0:         { indptr: Int32Array, indices: Int32Array, data: Float64Array },
  d1:         { indptr: Int32Array, indices: Int32Array, data: Float64Array },
  hodge0:     Float64Array,
  hodge1:     Float64Array,
  vertexAreas: Float64Array,
  normals:    Float64Array
}

solveEigenmodes(
  L_indptr: Int32Array, L_indices: Int32Array, L_data: Float64Array,
  M_indptr: Int32Array, M_indices: Int32Array, M_data: Float64Array,
  k: number, sigma: number
): { eigenvectors: Float64Array, eigenvalues: Float64Array }

normaliseEigenmodes(
  U: Float64Array, M_indptr: Int32Array, M_indices: Int32Array,
  M_data: Float64Array, V: number, k: number
): Float64Array

factoriseCholesky(
  L_indptr: Int32Array, L_indices: Int32Array, L_data: Float64Array
): number  // handle

solveCholesky(handle: number, rhs: Float64Array): Float64Array

applyOperator(
  op_indptr: Int32Array, op_indices: Int32Array, op_data: Float64Array,
  field: Float64Array
): Float64Array

helmholtzDecompose(
  field: Float64Array, d0_csr: CSR, d1_csr: CSR, L_chol_handle: number
): { irrotational: Float64Array, solenoidal: Float64Array }

projectEigenmodes(
  U: Float64Array, M_diag: Float64Array, u: Float64Array
): Float64Array
```

---

## Implementation Rules

1. **Use geometry-central** for all operator assembly. Do NOT reimplement
   cotangent weights, DEC operators, or vertex areas.

2. **Cholesky factor cache**: Use `std::map<int, Eigen::SimplicialLLT<...>>`
   in module scope. Free when mesh changes. Never expose raw pointers to JS.

3. **Spectra configuration**: Use `SymGEigsShiftSolver` with
   `GEigsMode::Cholesky`. CHOLMOD is used automatically when SuiteSparse
   is available. Always enforce real + symmetric before passing to Spectra.

4. **M-orthonormalisation** (`normaliseEigenmodes`): Must implement the
   Gram-matrix whitening approach (Cholesky path + eigen fallback) from
   the MATLAB `normalize.m`. This is critical for clustered eigenvalues.

5. **Remove DC mode** after solving — see `removeDC.m` reference.

6. **Error handling**: All N-API functions must catch C++ exceptions and
   convert to `Napi::Error::New(env, msg)`. Never let exceptions propagate.

7. **Thread safety**: The addon runs in the main Node.js thread. Long
   operations (eigensolver) should use `Napi::AsyncWorker` to avoid
   blocking the event loop.

---

## Build System

Primary: **CMake 3.20+**. Fallback: `binding.gyp` for `node-gyp`.

```cmake
cmake_minimum_required(VERSION 3.20)
project(bioctree_addon)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(deps/geometry-central)
find_package(Eigen3 3.4 REQUIRED)
find_package(SuiteSparse REQUIRED)
# Spectra headers in deps/spectra/include
# N-API via cmake-js or manual Node.js include paths
```

Build: `bash scripts/build-native.sh`
Output: `bioctree_addon.node` → copied to project root.

---

## Testing Against MATLAB

For any numerical function, create a test that:
1. Loads a known mesh (icosphere or cortical mesh from test fixtures)
2. Runs the C++ function
3. Compares output to pre-saved MATLAB results (stored as .zarr or .mat)
4. Asserts max absolute error < 1e-10 for float64 operations

Test fixtures live in `tests/fixtures/`.
