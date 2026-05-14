# nxr API refactor proposal: flat → namespace-modular

> **Status (2026-05-13): abandoned.** The full namespace-tree refactor
> proposed here — renaming `nxr::compute::*` to a domain-aligned tree
> with `nxr::manifold::ops::`, `nxr::manifold::eigen::`, `nxr::field::*`
> etc. — was not pursued. Two lighter alignment passes landed instead
> and addressed the most important motivating concerns:
>
> 1. **`MeshOperators` / `DECOperators` field names aligned to
>    geometry-central** (commit `2dbb95b`): `stiffness` →
>    `cotanLaplacian`, `vertexAreas` → `vertexDualAreas`, `normals` →
>    `vertexNormals`. DEC fields already matched GC. This addresses
>    the cross-binding consumer's mental-translation cost without
>    moving every C++ symbol.
> 2. **View-struct dedup** (commit `b59779e`): the operator structs
>    are now views over geometry-central's cached matrices instead
>    of value-owning copies. Addresses the structural concern that
>    the old `MeshOperators` was an opaque value blob.
>
> The full +bct-mirroring rename (`assembleMeshOperators` →
> `manifold::ops::assembleMesh`, etc.) was judged not worth the
> migration cost for downstream consumers (nxr-design-system,
> cortical-flow, MATLAB +bct users) given that the naming-and-shape
> wins above ship 80% of the perceived benefit. The MATLAB side
> continues to live at `+bct.+manifold.+eigen.solve` etc. while
> the C++ side stays flat at `nxr::compute::*`.
>
> Preserved as historical design rationale. Do not implement.

**Scope:** rename and reorganize the public API of `nxr-compute` (and its
four bindings) from a single flat namespace to a domain-aligned namespace
tree that mirrors the MATLAB `+bct` toolbox.
**Non-goals:** repository split, per-component CMake graph, multiple
versioned releases. This is a structural refactor inside one bundle, not
a modularization of the build.

---

## 1. Context

`nxr-compute` is currently a flat C++ API under one namespace (`nxr::compute`)
that exposes ~40 free functions, ~15 result structs, three stateful solver
classes, and a handful of cross-cutting utilities (`Error`,
`CancellationToken`, `ProgressObserver`). All four binding shells (Node addon,
WASM/Embind, MEX, CLI) flatten the same surface into a single dispatch layer.

This shape worked while the library was scoped to differential geometry on
triangle meshes. It starts to break down for two reasons:

1. **Domain expansion.** Adding `pocketfft`-backed time-series tools (FFT,
   Morlet CWT, filtering — the engine for the GBFs eigenmode-time-frequency
   pipeline in `nxr-neuro`) puts unrelated names in the same namespace. A
   downstream consumer reading `nxr.solveEigenmodes` next to `nxr.cwtMorlet`
   has no signal that one is mesh DG and the other is signal processing.
2. **MATLAB-side reference drift.** The `bioctree/toolbox/+bct` toolbox
   already organizes the same conceptual API into a clean namespace tree
   (`+bct.+manifold.+eigen.solve`, `+bct.+field.+generate.delta`, etc.).
   `nxr-compute`'s `CLAUDE.md` "MATLAB Reference Functions" table maps each
   C++ function to its `+bct` cousin, but the C++ side doesn't mirror the
   structure — only the function bodies. The asymmetry forces every
   cross-binding consumer (notably MATLAB callers via MEX) to mentally
   translate between two layouts.

The +bct survey (recorded in this document under §3) shows nxr-compute's
current 12 ported functions are a strict subset of a much larger
~100-function reference system. Aligning structurally now is cheap; doing
it after pocketfft and the GBFs gaps land is more work.

## 2. Goals & non-goals

### Goals

- A single namespace tree, mirroring `+bct`, that downstream consumers can
  navigate intuitively.
- Per-domain headers and source folders, so file-level organization matches
  API-level organization.
- One binding shell per target (no fragmentation), with nested-object
  registration so the JS / MEX / CLI surface preserves the namespace tree.
- Backward-compat shims that let downstream apps (cortical-flow,
  three.js gallery apps, MATLAB consumers, the bench harness) migrate
  lazily, file by file, with `@deprecated` IDE strikethrough as the TODO
  list.
- Hard rule against cross-namespace internal includes, so the tree stays
  meaningful over time.

### Non-goals

- **No CMake component graph.** All sources compile into one static
  library. Optional per-namespace `NXR_BUILD_*` flags can be added later
  if a downstream consumer ever needs to slim a WASM bundle, but the
  default and only required mode is "everything on."
- **No new repos.** No `nxr-signal` sibling library; `nxr::time::*` lives
  inside `nxr-compute`.
- **No semantic changes to existing functions.** Rename, regroup, re-expose.
  Every existing test must pass bit-for-bit on the renamed surface,
  including the bench infrastructure (`bench:all` + `bench:diff` against
  `bench/baselines.json`).
- **No upstream-MATLAB changes** as part of this PR. The `+bct` rename is
  one-way: nxr mirrors +bct, not the other way around.

## 3. Target namespace tree

```
nxr::
├── core::                    [cross-cutting infrastructure]
│   ├── Error, ErrorCode
│   ├── CancellationToken
│   ├── ProgressObserver
│   └── storage helpers (§11 row-major / col-major flatten utilities)
│
├── manifold::                [mesh / differential geometry — was nxr::compute::*]
│   ├── ComputeContext        [per-mesh state container, top-level]
│   ├── ops::                 [+bct.+manifold.+operator]
│   │   ├── MeshOperators
│   │   ├── DECOperators
│   │   ├── CholeskyCache
│   │   ├── assembleMesh
│   │   ├── assembleDEC
│   │   └── (gap) gradient, curl, divergence    [matrix exports]
│   ├── eigen::               [+bct.+manifold.+eigen]
│   │   ├── EigenResult
│   │   ├── solve
│   │   ├── normalize
│   │   └── removeDC
│   ├── solve::               [+bct.+manifold.+solve]
│   │   ├── poisson
│   │   └── (gap) heat        [equation, not distance]
│   ├── geometry::            [+bct.+manifold.+geometry]
│   │   ├── CurvatureResult, NormalType, FaceFrames
│   │   ├── curvatures
│   │   ├── vertexNormals
│   │   ├── faceFrames
│   │   └── parametrization::compute            [BFF]
│   ├── distance::            [+bct.+manifold.+solve.{heatdistance, signedDistance}]
│   │   ├── HeatGeodesicSolver
│   │   ├── SignedHeatSolver, SignedHeatLevelSet
│   │   ├── compute           [heat-method geodesic, was computeGeodesicDistance]
│   │   ├── signed            [was signedHeatDistance]
│   │   └── tracePath         [flip-out geodesic, was tracePath]
│   ├── transport::           [vector heat — was vectorHeat*]
│   │   ├── VectorHeatSolver
│   │   ├── LogMapResult, LogMapStrategy
│   │   ├── parallel          [was vectorHeatTransport]
│   │   ├── extendScalar      [was vectorHeatExtendScalar]
│   │   ├── logMap            [was vectorHeatLogMap]
│   │   └── findCenter        [was vectorHeatFindCenter]
│   ├── connection::          [direction fields, +bct.+manifold.+connection]
│   │   ├── DirectionFieldResult, SmoothVertexFieldResult
│   │   ├── trivial           [was computeDirectionField]
│   │   ├── smoothFace        [was computeSmoothFaceField]
│   │   └── smoothVertex      [was computeSmoothVertexField]
│   ├── decompose::           [Hodge — co-located here as it operates on DEC]
│   │   ├── HodgeResult
│   │   └── hodge             [was hodgeDecompose]
│   ├── (gap) query::         [+bct.+manifold.+query: BFS, shortest paths]
│   ├── (gap) transform::     [+bct.+manifold.+transform: gradient projection]
│   └── (gap) health::        [+bct.+manifold.+health: validation]
│
├── field::                   [+bct.+field]
│   ├── generate::            [+bct.+field.+generate]
│   │   ├── delta
│   │   ├── randomVertexScalar, randomFaceScalar, randomOmega
│   │   ├── randomDecomposed1Form
│   │   ├── eigenmodeField
│   │   ├── heatDiffusion     [was generateHeatDiffusion]
│   │   └── dampedWave        [was generateDampedWave]
│   ├── interp::
│   │   └── whitney           [was whitneyInterpolate]
│   ├── op::
│   │   └── gradient          [scalar field → face gradients, was scalarGradient]
│   ├── stripes::             [Knöppel-Crane stripe patterns]
│   │   ├── StripePatternResult
│   │   ├── compute           [was computeStripePattern]
│   │   └── computeFreq       [was computeStripePatternFreq]
│   └── viz::
│       ├── IsolineResult, StreamlineResult
│       ├── isolines          [was computeIsolines]
│       └── streamlines       [was traceStreamlines]
│
├── time::                    [NEW — Phase 1]
│   ├── fft::                 [forward, inverse, plan cache; pocketfft-backed]
│   ├── wavelet::             [cwtMorlet; later dwt if needed]
│   ├── filter::              [bandpass, FIR/IIR, Hilbert]
│   └── window::              [hann, hamming, dpss]
│
└── (future) graph::, image::, volume::, spectral::, brush::, kernel::
```

### Notes on naming choices

- **`ops` not `operator`.** `operator` is a reserved word in C++; can't be
  a namespace. `ops` is the closest unambiguous shortening and is used
  in similar libraries (TensorFlow, PyTorch).
- **`distance` covers both unsigned and signed.** `+bct` puts these under
  `+manifold/+solve` (`heatdistance`, `signedDistance`); promoting to a
  dedicated sub-namespace keeps the four related solvers (heat geodesic,
  signed heat, flip-out path, future Dijkstra) co-located.
- **`transport` for vector heat.** Mirrors the conceptual operation
  (parallel transport of tangent vectors) rather than the implementation
  detail (heat method). Keeps room for non-heat-method transport later.
- **`connection` houses direction fields.** `+bct.+manifold.+connection`
  has parallel-transport-related material; smooth-direction-field
  generators land here because they're solving for a connection.
- **`decompose::hodge`** rather than `solve::hodge`. The Hodge
  decomposition is an algebraic decomposition of a 1-form into three
  parts, not a "solve for an unknown given an equation." Naming it
  `decompose` keeps `solve` reserved for PDE solvers.
- **`field::stripes`** is a sub-namespace because both `compute` and
  `computeFreq` exist; placing them as `field::stripes::compute` /
  `::computeFreq` reads better than `field::computeStripePattern*`.

## 4. Mapping table: current → new

Every public symbol in `include/nxr/compute.h` and the cross-cutting headers,
mapped to its new home. Organized by current header.

### From `compute.h` — types

| Current | New | Notes |
|---|---|---|
| `nxr::compute::ComputeContext` | `nxr::manifold::ComputeContext` | Top-level; not in a sub-namespace because it's the root state container. |
| `nxr::compute::MeshOperators` | `nxr::manifold::ops::MeshOperators` | |
| `nxr::compute::DECOperators` | `nxr::manifold::ops::DECOperators` | |
| `nxr::compute::CholeskyCache` | `nxr::manifold::ops::CholeskyCache` | |
| `nxr::compute::EigenResult` | `nxr::manifold::eigen::EigenResult` | |
| `nxr::compute::HeatGeodesicSolver` | `nxr::manifold::distance::HeatGeodesicSolver` | |
| `nxr::compute::HodgeResult` | `nxr::manifold::decompose::HodgeResult` | |
| `nxr::compute::CurvatureResult` | `nxr::manifold::geometry::CurvatureResult` | |
| `nxr::compute::NormalType` | `nxr::manifold::geometry::NormalType` | |
| `nxr::compute::FaceFrames` | `nxr::manifold::geometry::FaceFrames` | |
| `nxr::compute::IsolineResult` | `nxr::field::viz::IsolineResult` | |
| `nxr::compute::VectorHeatSolver` | `nxr::manifold::transport::VectorHeatSolver` | |
| `nxr::compute::LogMapResult` | `nxr::manifold::transport::LogMapResult` | |
| `nxr::compute::LogMapStrategy` | `nxr::manifold::transport::LogMapStrategy` | |
| `nxr::compute::SignedHeatSolver` | `nxr::manifold::distance::SignedHeatSolver` | |
| `nxr::compute::SignedHeatLevelSet` | `nxr::manifold::distance::SignedHeatLevelSet` | |
| `nxr::compute::SmoothVertexFieldResult` | `nxr::manifold::connection::SmoothVertexFieldResult` | |
| `nxr::compute::StripePatternResult` | `nxr::field::stripes::StripePatternResult` | |
| `nxr::compute::DirectionFieldResult` | `nxr::manifold::connection::DirectionFieldResult` | |
| `nxr::compute::StreamlineResult` | `nxr::field::viz::StreamlineResult` | |

### From `compute.h` — functions

| Current | New |
|---|---|
| `assembleMeshOperators(ctx)` (and 2-arg overload) | `manifold::ops::assembleMesh(ctx)` |
| `assembleDECOperators(ctx)` | `manifold::ops::assembleDEC(ctx)` |
| `solveEigenmodes(...)` | `manifold::eigen::solve(...)` |
| `normalizeEigenmodes(U, M)` | `manifold::eigen::normalize(U, M)` |
| `removeDC(result)` | `manifold::eigen::removeDC(result)` |
| `solvePoisson(...)` (both overloads) | `manifold::solve::poisson(...)` |
| `computeGeodesicDistance(...)` | `manifold::distance::compute(...)` |
| `hodgeDecompose(...)` | `manifold::decompose::hodge(...)` |
| `generateRandomOmega(...)` | `field::generate::randomOmega(...)` |
| `generateDelta(...)` | `field::generate::delta(...)` |
| `generateRandomVertexScalar(...)` | `field::generate::randomVertexScalar(...)` |
| `generateRandomFaceScalar(...)` | `field::generate::randomFaceScalar(...)` |
| `generateEigenmodeField(...)` | `field::generate::eigenmodeField(...)` |
| `generateRandomDecomposed1Form(...)` | `field::generate::randomDecomposed1Form(...)` |
| `generateHeatDiffusion(...)` (both overloads) | `field::generate::heatDiffusion(...)` |
| `generateDampedWave(...)` | `field::generate::dampedWave(...)` |
| `computeCurvatures(ctx)` | `manifold::geometry::curvatures(ctx)` |
| `computeVertexNormals(ctx, type)` | `manifold::geometry::vertexNormals(ctx, type)` |
| `computeFaceFrames(ctx)` | `manifold::geometry::faceFrames(ctx)` |
| `tracePath(...)` | `manifold::distance::tracePath(...)` |
| `computeUVCoordinates(ctx)` | `manifold::geometry::parametrization::compute(ctx)` |
| `whitneyInterpolate(...)` | `field::interp::whitney(...)` |
| `scalarGradient(...)` | `field::op::gradient(...)` |
| `computeIsolines(...)` | `field::viz::isolines(...)` |
| `vectorHeatTransport(...)` | `manifold::transport::parallel(...)` |
| `vectorHeatExtendScalar(...)` | `manifold::transport::extendScalar(...)` |
| `vectorHeatLogMap(...)` | `manifold::transport::logMap(...)` |
| `vectorHeatFindCenter(...)` | `manifold::transport::findCenter(...)` |
| `signedHeatDistance(...)` | `manifold::distance::signed(...)` |
| `computeSmoothFaceField(...)` | `manifold::connection::smoothFace(...)` |
| `computeSmoothVertexField(...)` | `manifold::connection::smoothVertex(...)` |
| `computeStripePattern(...)` | `field::stripes::compute(...)` |
| `computeStripePatternFreq(...)` | `field::stripes::computeFreq(...)` |
| `computeDirectionField(...)` | `manifold::connection::trivial(...)` |
| `traceStreamlines(...)` | `field::viz::streamlines(...)` |

### From cross-cutting headers

| Current | New |
|---|---|
| `nxr::compute::Error` (`errors.h`) | `nxr::core::Error` |
| `nxr::compute::ErrorCode` | `nxr::core::ErrorCode` |
| `nxr::compute::CancellationToken` (`cancellation.h`) | `nxr::core::CancellationToken` |
| `nxr::compute::ProgressObserver` (`progress.h`) | `nxr::core::ProgressObserver` |

### What stays where it is

- `geometrycentral::*` forward declarations stay forward-declared in
  `manifold/ops.h` (not exported through any nxr namespace).
- The `inline` overload helpers (e.g. `solvePoisson(MeshOperators&, ...)`)
  stay as inline forwarders — they just live in the new namespace.
- `NormalType` enum values (e.g. `NormalType::AreaWeighted`) keep their
  enum-class names; only the enclosing namespace changes.

## 5. Header & source layout

### Headers

```
include/nxr/
├── nxr.h                                # umbrella; #includes everything below
├── core.h                               # rolls up core/*
├── core/
│   ├── error.h
│   ├── cancellation.h
│   ├── progress.h
│   └── storage.h                        # §11 flatten helpers (lifted from current bindings)
├── manifold.h                           # rolls up manifold/*
├── manifold/
│   ├── context.h                        # ComputeContext
│   ├── ops.h
│   ├── eigen.h
│   ├── solve.h
│   ├── geometry.h
│   ├── distance.h
│   ├── transport.h
│   ├── connection.h
│   └── decompose.h
├── field.h                              # rolls up field/*
└── field/
    ├── generate.h
    ├── interp.h
    ├── op.h
    ├── stripes.h
    └── viz.h
```

`compute.h` becomes a thin compatibility shim during the transition:

```cpp
// include/nxr/compute.h — DEPRECATED
// Kept during the transition window only. New code should include
// <nxr/nxr.h> or the per-namespace headers directly.
#pragma once
#include <nxr/nxr.h>

namespace nxr::compute {
    // Re-export the renamed symbols under their old names via using-aliases.
    using EigenResult = manifold::eigen::EigenResult;
    using ComputeContext = manifold::ComputeContext;
    // … one line per renamed symbol
}
```

The shim allows existing C++ callers (the addon, the WASM/Embind bindings,
the MEX dispatcher, the CLI, and the test harness) to keep building
unchanged on day 1 of the rename. They migrate to new include paths
opportunistically.

### Sources

```
src/
├── core/
│   ├── error.cpp
│   ├── cancellation.cpp
│   └── progress.cpp
├── manifold/
│   ├── context.cpp                      # was implicit in mesh_operators.cpp
│   ├── ops/
│   │   ├── mesh_operators.cpp           # moved verbatim
│   │   └── cholesky_cache.cpp
│   ├── eigen/
│   │   ├── eigensolver.cpp              # moved
│   │   └── normalize.cpp
│   ├── solve/
│   │   └── poisson_solver.cpp
│   ├── geometry/
│   │   ├── curvatures.cpp
│   │   ├── normals.cpp
│   │   ├── face_frames.cpp
│   │   └── parametrization.cpp
│   ├── distance/
│   │   ├── geodesic.cpp
│   │   ├── geodesic_path.cpp
│   │   └── signed_heat.cpp
│   ├── transport/
│   │   └── vector_heat.cpp
│   ├── connection/
│   │   ├── direction_field.cpp
│   │   └── smooth_field.cpp
│   └── decompose/
│       └── hodge.cpp
├── field/
│   ├── generate/
│   │   └── field_generators.cpp         # split if it grows; one file ok for now
│   ├── interp/
│   │   └── whitney.cpp                  # currently inline; extract
│   ├── stripes/
│   │   └── stripe_patterns.cpp
│   ├── op/
│   │   └── gradient.cpp                 # currently inline; extract
│   └── viz/
│       ├── isolines.cpp
│       └── vector_field.cpp             # streamlines
└── (Phase 1) time/
    ├── fft.cpp
    ├── wavelet.cpp
    ├── filter.cpp
    └── window.cpp
```

This is mostly `git mv` with namespace edits at the top of each file.
The split of `field_generators.cpp` and the extraction of inline
implementations from `compute.h` are the only sub-file edits.

## 6. Binding strategy

One binding shell per target. Same patterns as today; only the registered
names change.

### Node addon (N-API)

Nested-object registration — N-API supports this directly:

```cpp
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    auto manifold = Napi::Object::New(env);
    {
        auto eigen = Napi::Object::New(env);
        eigen.Set("solve",     Napi::Function::New(env, ManifoldEigenSolve));
        eigen.Set("normalize", Napi::Function::New(env, ManifoldEigenNormalize));
        eigen.Set("removeDC",  Napi::Function::New(env, ManifoldEigenRemoveDC));
        manifold.Set("eigen", eigen);

        auto ops = Napi::Object::New(env);
        ops.Set("assembleMesh", Napi::Function::New(env, ManifoldOpsAssembleMesh));
        ops.Set("assembleDEC",  Napi::Function::New(env, ManifoldOpsAssembleDEC));
        manifold.Set("ops", ops);
        // … etc
    }
    exports.Set("manifold", manifold);

    auto field = Napi::Object::New(env);
    // … etc
    exports.Set("field", field);

    // Phase 1
    // auto time = Napi::Object::New(env);
    // exports.Set("time", time);

    // Backward-compat shim — flat names alias into the tree
    exports.Set("solveEigenmodes", manifold.Get("eigen").As<Napi::Object>().Get("solve"));
    // … one line per legacy name
    return exports;
}
```

`addon.cpp` stays a single file but is internally organized into
namespace blocks (one block per `nxr::manifold::*`, one per
`nxr::field::*`, etc.) so it's still readable at ~1000 LoC. If it grows
past comfort, split into `addon/manifold.cpp`, `addon/field.cpp`, etc.,
all linked into the same `.node` artifact.

### WASM (Embind)

Embind doesn't have first-class nested namespaces in JS, so use flat
registered names with a JS wrapper that rebuilds the tree:

```cpp
EMSCRIPTEN_BINDINGS(nxr) {
    function("manifold_eigen_solve", &binding_manifold_eigen_solve);
    function("manifold_eigen_normalize", &binding_manifold_eigen_normalize);
    // …
}
```

```js
// bindings/wasm/wrapper.js
const Module = await loadModule();
const nxr = {
  manifold: {
    eigen: {
      solve: Module.manifold_eigen_solve,
      normalize: Module.manifold_eigen_normalize,
      removeDC: Module.manifold_eigen_removeDC,
    },
    // …
  },
  field: { /* … */ },
};

// Backward-compat shim
/** @deprecated Use nxr.manifold.eigen.solve */
nxr.solveEigenmodes = nxr.manifold.eigen.solve;
// …

export default nxr;
```

### MEX

Keep the single `nxr_compute_mex.mexw64` artifact. Dispatch on the first
input argument (a dotted method name string):

```matlab
% +nxr/+manifold/+eigen/solve.m
function result = solve(K, M, k, varargin)
    result = nxr_compute_mex('manifold.eigen.solve', K, M, k, varargin{:});
end
```

This makes the MATLAB-side surface `nxr.manifold.eigen.solve(...)` —
identical structurally to the C++ and JS surfaces. The `+nxr/` package
folder becomes a thin shim layer; the MEX dispatcher does string→handler
lookup. Worst case ~50 .m files, all one-liners.

### CLI

Sub-command structure mirrors the namespace tree:

```sh
nxr_compute manifold.eigen.solve --K=K.mtx --M=M.mtx --k=200 --out=eigs.npz
nxr_compute field.generate.heatDiffusion --eig=eigs.npz --u0=u0.npy --out=diffusion.npz
```

CLI dispatch is a single string → function-pointer table. Today's
single-purpose CLI (canonical spectrum check) becomes one of many
sub-commands.

## 7. Backward compatibility

Three compat layers are needed during the transition window. All three
are cheap and let downstream apps migrate lazily.

### C++ (in nxr-compute itself, the addon, MEX dispatcher, CLI)

- `compute.h` becomes a shim that re-exports the new symbols under their
  old names via `using` aliases (see §5 example).
- The shim is annotated `[[deprecated("Use <nxr/manifold/eigen.h> etc.")]]`
  so downstream C++ callers see compiler warnings (not errors).
- Tests are migrated as part of the rename PR (mechanical), but the shim
  stays so external C++ consumers (none today, but future-proof) can
  upgrade independently.

### JS (Node addon and WASM wrapper)

- The flat names (`nxr.solveEigenmodes`, `nxr.computeGeodesicDistance`,
  etc.) stay registered, aliased to the new tree positions. ~30 lines per
  binding.
- TypeScript `.d.ts` files mark legacy names with `@deprecated` so VSCode
  shows them struck through.

### MATLAB

- The existing flat function names (if any are exposed at the package
  root) get one-line forwarders to the new dotted-name dispatcher.
- MATLAB callers using `+bct.+manifold.+eigen.solve(...)` are
  **already** on the future API — they need no changes.

### IPC layer (cortical-flow Electron)

- The CLAUDE.md mentions `native:*` IPC channels (`native:solveEigenmodes`,
  etc.). These get aliased the same way: register both
  `native:manifold.eigen.solve` (new) and `native:solveEigenmodes`
  (deprecated, forwards to new). Renamer-bot territory if you want to
  scrub the renderer all at once.

## 8. Migration phases

### Phase 0: rename + restructure (this proposal)

**Effort:** 1–2 days of mechanical work + 1 day of bench validation.
**Branch:** `refactor/api-namespace-tree` in a worktree.
**Risk:** low — no algorithmic changes.

Steps:

1. Add new headers (`include/nxr/{core,manifold,field}/*.h`) with namespace
   declarations matching §3.
2. Move source files into the new `src/{core,manifold,field}/*` layout.
   Use `git mv` so history is preserved.
3. Update each `.cpp`'s namespace declaration. No body changes.
4. Update CMake `target_sources` lists.
5. Convert `compute.h` to the shim form (§5).
6. Update bindings: Node addon and Embind wrapper now register nested
   objects; keep flat aliases as the shim.
7. Update tests: switch to new include paths.
8. Update CLAUDE.md:
   - Replace `nxr::compute::` with `nxr::manifold::` etc. in code samples.
   - Add the new "namespace discipline" rule (§9 below).
   - Regenerate the "MATLAB Reference Functions" table from the +bct
     map in `docs/bct-api-map.md` (proposed sibling document — not yet
     written; can be generated from the +bct survey we already have).
9. Run `bench:all && bench:diff` against `bench/baselines.json`. Verdict
   must be all-green. Any change in numerics is a refactor bug.
10. Run `test_eigen.exe`, `test_cholesky_cache.exe`, `test_cancellation.exe`,
    `test_progress.exe` natively. All must pass.
11. WASM smoke (`scripts/_smoke-wasm.mjs`) and MEX smoke must pass.

### Phase 1: nxr::time::* with pocketfft

**Prereq:** Phase 0 complete and on main.
**Effort:** 2–3 days for FFT + Morlet CWT skeleton; more for full filter
suite.
**Risk:** medium — new dependency, new binding surface.

Adds:

- `include/nxr/time/{fft,wavelet,filter,window}.h`
- `src/time/{fft,wavelet,filter,window}.cpp`
- pocketfft as a header-only dep (one new line in `cmake/Dependencies.cmake`,
  vendored under `deps/pocketfft/` or fetched).
- Plan caching for FFT (analogous to `manifold::ops::CholeskyCache`).
- Bindings for `nxr.time.fft.{forward,inverse}`,
  `nxr.time.wavelet.cwtMorlet`.
- New native test `test_time_fft.exe` cross-validating against SciPy
  reference fixtures (numpy.fft / scipy.signal).

### Phase 2: high-value gaps in manifold::

**Prereq:** Phase 0.
**Effort:** depends on which gaps; the GBFs-relevant ones are small.
**Risk:** low — incremental additions.

Priority order (filtered to nxr-neuro / GBFs use cases):

- `manifold::transform::projectToEigenbasis(U, field)` — vertex-domain
  field → mode-domain coefficients. Just a `U^T · M · field` for the
  M-orthonormal case. **This is the function GBFs needs.**
- `manifold::ops::{gradient, curl, divergence}` — exporting d0, d1, d1ᵀ
  as readable matrices. Trivial since geometry-central already builds
  them; just expose.
- `manifold::query::{bfs, shortestPath}` — graph queries on the mesh
  adjacency. Useful for patch / neighborhood analysis.
- `manifold::health::{isManifold, hasBoundary, eulerCharacteristic}` —
  mesh validation. Tiny functions, cheap to add.

### Phase 3+: future namespaces

Add as needed, not preemptively. Candidates from the +bct survey:

- `nxr::filter::*` — spectral filtering (cross-domain: works on both
  mesh-spectral and time-spectral inputs).
- `nxr::spectral::*` — joint mode × frequency spectra. **This is
  natural home for the eigenmode time-frequency analysis** the
  GBFs / nxr-neuro pipeline ultimately produces.
- `nxr::brush::*` — Gaussian / spectral / trajectory selection patches.
  Useful for interactive UIs.
- `nxr::kernel::*` — kernel evaluators with parameter binding.
- `nxr::graph::*` — graph (non-mesh) analysis: connectome ops,
  community detection, centrality.

Each of these is a separate proposal, separate PR, separate decision.
Not blockers for Phase 0/1/2.

## 9. Discipline rules (additions to CLAUDE.md)

Add the following hard rule under a new section "Namespace organization":

> **Top-level namespaces under `nxr::` are domain-isolated.** A `.cpp`
> file under `src/manifold/` may not include any header under
> `nxr/time/`, `nxr/graph/`, `nxr/image/`, etc., and vice versa. The
> only cross-cutting include is `<nxr/core.h>`.
>
> Functions that span domains (e.g. "time-frequency decomposition of
> mode-coefficient time-series") belong in the **application layer** —
> `nxr-neuro` or whichever orchestrator owns the cross-domain workflow —
> not inside any nxr-compute namespace.
>
> Verification: a CI check that `grep -r '#include <nxr/' src/<domain>/`
> returns only `<nxr/core.h>` and `<nxr/<domain>/*.h>`.

This is the rule that makes the namespace tree mean something. Without
it, the umbrella drifts into a kitchen sink within a year.

## 10. Open decisions

These need a user judgment call before Phase 0 can land.

1. **Retain `nxr::compute` as an umbrella alias indefinitely?** Either:
   (a) keep `compute.h` as a permanent shim, or
   (b) remove it at a future major version.
   I'd recommend (b) — keep it for one major version (≈ 6 months of
   downstream migration), then drop. Track with an issue/milestone.
2. **Package rename?** `nxr-compute` is no longer just "compute" — it's
   a multi-domain toolkit. Options:
   (a) keep `nxr-compute` (the npm/Cargo/whatever name) and just
       restructure internally;
   (b) rename to `nxr` or `nxr-toolkit` at the package level too.
   (a) is lower-friction; (b) is more honest. Not blocking; can defer.
3. **Embind nesting strategy.** Either:
   (a) flat names in C++, JS-side wrapper rebuilds the tree (proposed in
       §6 — simple), or
   (b) Embind `value_object` + nested classes — more native-feeling but
       more boilerplate per binding.
   (a) keeps the WASM artifact small and the Embind code idiomatic.
4. **MEX dispatcher: single mexFunction or many?** Either:
   (a) one mexFunction, dispatches on dotted method-name string
       (proposed — fewer artifacts, smaller binary footprint), or
   (b) one mexFunction per leaf method.
   (a) wins for size and MATLAB Compiler / package distribution
   ergonomics. Tiny per-call overhead from the string lookup, but
   measured against the underlying compute it's negligible.
5. **`+bct` mirror for the new `+time` namespace.** Either:
   (a) add `+bct/+time/*.m` skeletons in lockstep with Phase 1 (keeps
       the bidirectional symmetry rule from CLAUDE.md), or
   (b) accept that the `+time` namespace has no MATLAB-side reference
       and document the asymmetry in CLAUDE.md.
   (a) preserves the audit trail; (b) is faster but creates a
   precedent for future asymmetric namespaces.

## 11. Verification plan

End-to-end checks before declaring Phase 0 done:

1. **Bench parity** — `npm run bench:all && npm run bench:diff` against
   `bench/baselines.json`. Every method's verdict column unchanged.
   This is the safety net rule §11 in nxr-compute's CLAUDE.md.
2. **Native tests** — `test_eigen`, `test_cholesky_cache`,
   `test_cancellation`, `test_progress`, `test_field_generators`,
   `test_visualization_primitives` all pass on the renamed surface.
3. **WASM smoke** — `scripts/_smoke-wasm.mjs` round-trips an icosahedron
   eigenmode under the new `nxr.manifold.eigen.solve(...)` name.
4. **MEX smoke** — load fixture mesh in MATLAB, call
   `nxr.manifold.eigen.solve(K, M, 200)`, assert spectrum matches the
   pre-rename reference at `< 1e-12` absolute error.
5. **Three.js gallery** — open the existing mesh-tests gallery; the JS
   shim should mean zero changes to gallery code, all charts render
   correctly. If anything breaks, the shim is incomplete.
6. **Cortical-flow Electron app** — same: legacy IPC names still work,
   no renderer changes needed.

If any of (1)–(6) fail, Phase 0 is not done. Roll back via git revert and
diagnose before trying again.

---

## Appendix A: file-level git plan for Phase 0

Order of operations in the rename PR:

1. `git checkout -b refactor/api-namespace-tree`
2. Create new header files (empty or with namespace declarations).
3. Create new source folders.
4. `git mv src/<old>.cpp src/manifold/<group>/<old>.cpp` for each.
5. Rewrite namespace blocks in each moved `.cpp` (sed-friendly).
6. Move type/function declarations from `compute.h` into per-namespace
   headers. Keep `compute.h` as the shim.
7. Update `CMakeLists.txt` `target_sources` paths.
8. Update bindings (`addon.cpp`, `nxr_compute_wasm.cpp`,
   `nxr_compute_mex.cpp`, CLI `main.cpp`) to register new names + flat
   aliases.
9. Update tests' include paths.
10. Update `CLAUDE.md`.
11. Run all verification checks from §11.
12. PR review, merge.

Estimated: one focused day for steps 1–9, half a day for steps 10–12 and
verification iteration. The bench-diff step is the truth oracle — if it's
green, the refactor is non-functional; if it isn't, something moved that
shouldn't have.

## Appendix B: deferred but worth recording

Things that came up in discussion and are deliberately out of scope for
Phase 0 but worth banking:

- **Plan caching for FFT.** Phase 1 should add a `nxr::time::fft::PlanCache`
  analogous to `manifold::ops::CholeskyCache`, factor reused across calls.
  Mention in Phase 1 design doc.
- **Color and brush packages.** `+bct` has these; nxr-compute doesn't.
  Future namespaces if interactive UIs in the gallery need them.
- **Volume / image namespaces.** If volumetric MRI / image-processing
  ever enters scope (T1 segmentation, BEM mesh extraction from MRI, etc.),
  these would be new top-level namespaces. Currently nxr-neuro plans to
  delegate this to MNE-CPP, so no in-scope work.
- **MNE-CPP integration boundary.** Recorded in
  `docs/integration-lessons.md` and the `nxr-neuro` plan. Phase 0 has no
  MNE-CPP touchpoints.
