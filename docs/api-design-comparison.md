# nxr-compute API design — three-way comparison

Three API designs are in flight in this repo. None is fully implemented yet.
This document shows all three side by side as tree diagrams so you can
inspect them visually before deciding which one to adopt as canonical.

**Status of each design:**

| Design | Where | Implemented? |
|---|---|---|
| **A.** C++ refactor proposal | `docs/api-refactor-proposal.md` | No — header (`include/nxr/compute.h`) and source (`src/*.cpp`) still use the flat `nxr::compute::*` namespace. |
| **B.** May 6 JS shim          | `bindings/wasm/js/index.mjs` (commit `366fcd6`) | **Yes** — running in WASM. Some methods are TODO stubs. |
| **C.** Today's schema         | `docs/schema/{manifold,field}.schema.json` | No — design-only document. Bindings still match (B). |

The three designs share the same goal — replace a flat dispatch surface with
a domain-organised tree — but disagree on **how to group**, **how to name**,
and **what counts as a first-class object**.

---

## Design A — C++ refactor proposal

Full C++ namespace tree mirroring the MATLAB `+bct` toolbox. 8 sub-namespaces
under `manifold`, 5 under `field`, plus `core::` for cross-cutting types and
a planned future `time::` for signal processing. Includes a full mapping
table from every current symbol to its new home, header / source layout,
binding strategy, and a "namespace discipline" CI rule.

```
nxr::
├── core::                              cross-cutting infrastructure
│   ├── Error, ErrorCode
│   ├── CancellationToken
│   ├── ProgressObserver
│   └── storage helpers                 §11 flatten utilities
│
├── manifold::                          (was nxr::compute::*)
│   ├── ComputeContext                  top-level state container
│   │
│   ├── ops::
│   │   ├── MeshOperators
│   │   ├── DECOperators                d0, d1, hodge0, hodge1, hodge2, hodge1Inverse
│   │   ├── CholeskyCache
│   │   ├── assembleMesh
│   │   ├── assembleDEC
│   │   └── (gap) gradient, curl, divergence       ← matrix exports
│   │
│   ├── eigen::
│   │   ├── EigenResult
│   │   ├── solve
│   │   ├── normalize
│   │   └── removeDC
│   │
│   ├── solve::                         PDE solvers
│   │   ├── poisson
│   │   └── (gap) heat                  ← equation, not distance
│   │
│   ├── geometry::
│   │   ├── CurvatureResult, NormalType, FaceFrames
│   │   ├── curvatures
│   │   ├── vertexNormals
│   │   ├── faceFrames
│   │   └── parametrization::compute    ← BFF
│   │
│   ├── distance::                      heat-method + path
│   │   ├── HeatGeodesicSolver
│   │   ├── SignedHeatSolver, SignedHeatLevelSet
│   │   ├── compute                     ← heat-method geodesic distance
│   │   ├── signed                      ← signed heat distance
│   │   └── tracePath                   ← flip-out geodesic path
│   │
│   ├── transport::                     vector heat method
│   │   ├── VectorHeatSolver
│   │   ├── LogMapResult, LogMapStrategy
│   │   ├── parallel                    ← was vectorHeatTransport
│   │   ├── extendScalar                ← was vectorHeatExtendScalar
│   │   ├── logMap                      ← was vectorHeatLogMap
│   │   └── findCenter                  ← was vectorHeatFindCenter
│   │
│   ├── connection::                    direction fields
│   │   ├── DirectionFieldResult, SmoothVertexFieldResult
│   │   ├── trivial                     ← prescribed singularities
│   │   ├── smoothFace                  ← Knöppel-Crane on faces
│   │   └── smoothVertex                ← Knöppel-Crane on vertices
│   │
│   ├── decompose::                     algebraic decompositions
│   │   ├── HodgeResult
│   │   └── hodge                       ← Helmholtz/Hodge α + β + γ
│   │
│   ├── (gap) query::                   BFS, shortest paths
│   ├── (gap) transform::               gradient projection, eigen-basis projection
│   └── (gap) health::                  isManifold, hasBoundary, eulerCharacteristic
│
├── field::                             (+bct.+field analog)
│   ├── generate::
│   │   ├── delta
│   │   ├── randomVertexScalar, randomFaceScalar, randomOmega
│   │   ├── randomDecomposed1Form
│   │   ├── eigenmodeField
│   │   ├── heatDiffusion               ← was generateHeatDiffusion
│   │   └── dampedWave                  ← was generateDampedWave
│   │
│   ├── interp::
│   │   └── whitney                     ← edge 1-form → face vectors
│   │
│   ├── op::
│   │   └── gradient                    ← scalar field → face gradients
│   │
│   ├── stripes::                       Knöppel-Crane stripe patterns
│   │   ├── StripePatternResult
│   │   ├── compute
│   │   └── computeFreq
│   │
│   └── viz::
│       ├── IsolineResult, StreamlineResult
│       ├── isolines
│       └── streamlines
│
├── time::                              NEW — Phase 1, NOT YET IMPLEMENTED
│   ├── fft::                           forward, inverse, plan cache (pocketfft)
│   ├── wavelet::                       cwtMorlet
│   ├── filter::                        bandpass, FIR/IIR, Hilbert
│   └── window::                        hann, hamming, dpss
│
└── (future) graph::, image::, volume::, spectral::, brush::, kernel::
```

**Counts:** `core` (1) + `manifold` (8 + 3 gaps) + `field` (5) + `time` (4) =
**21 sub-namespaces** under `nxr::`.

---

## Design B — May 6 JS shim (currently shipping)

The C++ side is **still flat**; the shim adds a JS namespace tree on top of
the existing `ContextWrapper`. Six groups, verb-organised, designed to
work without waiting for the C++ refactor. The legacy flat `createContext()`
entry point is preserved unchanged.

```
mctx = nxr.createManifoldContext(verts, faces)
│
├── solve.                              PDE solvers + spectral + Hodge
│   ├── poisson(sourceVerts, sourceValues)
│   ├── heat(sources, sourceValues, timesteps, alpha)
│   ├── eigen(k, sigma)
│   └── hodge(omega)                    ← provisional grouping
│
├── operator.                           discrete operators (renamed at JS boundary)
│   ├── d0()
│   ├── d1()
│   ├── star0()                         ← was hodge0
│   ├── star1()                         ← was hodge1
│   ├── star2()                         ← was hodge2
│   ├── star1Inverse()                  ← was hodge1Inverse
│   ├── mass()
│   ├── stiffness()
│   └── laplacian()                     ← alias
│
├── query.                              localization on the manifold
│   ├── center(sourceVerts, p)          ← vector heat Karcher mean
│   ├── isoline(scalars, level)         ← single-level isoline
│   ├── line[STUB]                      ← {method: 'todo'} + console.warn
│   ├── circle[STUB]                    ← {method: 'todo'} + console.warn
│   └── region[STUB]                    ← {method: 'todo'} + console.warn
│
├── measure.                            scalar/vector measurements
│   ├── distance(sourceVerts)           ← heat-method geodesic distance
│   ├── distance.signed(curve, isLoop)  ← callable on the .distance fn
│   ├── curvature()                     ← full CurvatureResult struct
│   ├── normal(type)                    ← vertex normals
│   ├── frame()                         ← face frames struct
│   ├── area[STUB]
│   └── density[STUB]
│
├── uv.                                 coordinate atlases
│   ├── bff()                           ← BFF UV coordinates
│   ├── logMap(sourceVertex, strategy)  ← vector heat log map
│   ├── stripe(vertexField, freq, …)    ← uniform-frequency stripes
│   └── stripeFreq(vertexField, freqs, …) ← per-vertex frequencies
│
├── interpolate.                        propagating values across mesh
│   ├── transport(sourceVerts, sourceVectors)   ← vector heat parallel transport
│   ├── extend(sourceVerts, sourceValues)       ← scalar heat extension
│   ├── directionField(singVerts, singValues)   ← trivial connection
│   ├── smoothFaceField(nSym, alignToCurv)      ← Knöppel-Crane on faces
│   └── smoothVertexField(nSym, alignToCurv)    ← Knöppel-Crane on vertices
│
└── _flat                               ← escape hatch: legacy ContextWrapper

ALSO EXPOSED:  free-function form
  nxr.nxr.manifold.{group}.{method}(mctx, ...args)
  e.g. nxr.nxr.manifold.solve.eigen(mctx, 300)
```

**Counts:** **6 groups** flat under `mctx`. No `field` namespace. No `core`.
No `time`.

---

## Design C — today's schema

A design contract (`docs/schema/manifold.schema.json` + `field.schema.json`)
with the manifold and the field as two *sibling* top-level namespaces.
**10 groups under manifold**, plus the field as a first-class object with
its own constructor and operations.

```
MANIFOLD  (docs/schema/manifold.schema.json)
manifold/
├── attributes                          fingerprint, nxr_compute_version, schema_version
│
├── core/                               required inputs
│   ├── vertices                        [V, 3] float64
│   └── faces                           [F, 3] int32
│
├── embedding/                          world placement (three.js Object3D analogue)
│   ├── bounding_box                    {min[3], max[3]}
│   └── transform                       [4, 4] affine
│
├── topology/
│   └── halfedge/                       PLANNED (not yet exposed)
│       ├── twin, next, vertex, face, edge, is_boundary
│
├── geometry/                           per-element scalars / vectors (need embedding)
│   ├── face/                           normals, areas, centroids, frames(e1, e2)
│   ├── vertex/                         normals, vertex_areas
│   ├── edge/                           lengths, cotangent_weights, dihedral_angles
│   ├── curvature/                      gaussian, mean, principal_min, principal_max, principal_dir_max
│   └── singularities                   2 anchor vertex indices defining the global tangent frame
│
├── operators/                          d0, d1, hodge* canonical; rest derived
│   ├── d0                              canonical
│   ├── d1                              canonical
│   ├── hodge0                          canonical
│   ├── hodge1                          canonical
│   ├── hodge2                          canonical
│   ├── hodge1_inverse                  canonical
│   ├── stiffness                       derived: d0^T * hodge1 * d0
│   ├── mass                            alias: hodge0
│   ├── gradient                        derived: d0
│   └── divergence                      derived: d0^T * hodge1
│
├── eigen/
│   ├── attributes                      K, operator_ref, inner_product_ref, sigma, normalized, dc_removed
│   ├── values                          [K] float64
│   └── vectors                         [V, K] float64, vMajor (row-major)
│
├── parametrization/                    coordinate atlases (global UV + local frame fields)
│   ├── uv                              [V, 2] BFF
│   ├── cross_field                     [V] complex128  (4-RoSy)
│   ├── line_field                      [V] complex128  (2-RoSy)
│   └── stripes                         [2*N, 3] phase isolines
│
├── query/                              returns geometric loci
│   ├── point(v)                        → v   (identity)
│   ├── line(va, vb)                    → [N, 3] edge-flip geodesic
│   └── area(v, level)                  → contour segments of heat-distance level set
│
└── measure/                            returns metric scalars
    ├── distance(va, vb)                → scalar length of query.line
    └── area(v, level)                  → scalar area enclosed by query.area


FIELD  (docs/schema/field.schema.json — separate namespace)
field/
├── attributes                          manifold_ref, kind, domain, time_varying, T, units, source
├── values                              [D] (scalar) | [D, 3] (vector)   where D = V|E|F per domain
├── gradient()                          scalar+vertex → vector+face Field
├── divergence()                        vector / 1-form → scalar+vertex Field   (NOT YET WIRED)
├── isolines(...)                       scalar+vertex → polylines
├── streamlines(...)                    vector+face → polylines
├── hodge()                             edge 1-form → α + β + γ decomposition
└── poisson()                           scalar+vertex → scalar+vertex potential

(Field operations carry an `applies_when` precondition over the field's
attributes — e.g. gradient's applies_when is "kind == 'scalar' && domain == 'vertex'".)
```

**Counts:** **10 groups** under manifold (`attributes`, `core`, `embedding`,
`topology`, `geometry`, `operators`, `eigen`, `parametrization`, `query`,
`measure`) plus the field as a parallel namespace (1 attribute block + 1 data
slot + 6 operations).

---

## Where the same operation lands in each design

A few key methods, traced through all three.

| Operation | A — C++ proposal | B — May 6 shim | C — today's schema |
|---|---|---|---|
| Cotan Laplacian (matrix) | `manifold::ops::assembleMesh().stiffness` | `mctx.operator.stiffness()` | `manifold/operators/stiffness` |
| Hodge ★₁                  | `manifold::ops::assembleDEC().hodge1`     | `mctx.operator.star1()`      | `manifold/operators/hodge1` |
| Eigensolve               | `manifold::eigen::solve(...)`             | `mctx.solve.eigen(...)`      | `manifold/eigen.solve(...)` then read `.values`, `.vectors` |
| Poisson PDE              | `manifold::solve::poisson(...)`           | `mctx.solve.poisson(...)`    | `field.poisson()` on a scalar+vertex field |
| Hodge decomposition      | `manifold::decompose::hodge(...)`         | `mctx.solve.hodge(omega)`    | `field.hodge()` on an edge 1-form |
| Heat geodesic distance   | `manifold::distance::compute(...)`        | `mctx.measure.distance(v)`   | under-the-hood inside `query.area` and `measure.distance` |
| Signed heat distance     | `manifold::distance::signed(...)`         | `mctx.measure.distance.signed(...)` | under-the-hood |
| Flip-out path            | `manifold::distance::tracePath(...)`      | (no direct, has `_flat`)     | `manifold/query/line(va, vb)` |
| Path length (scalar)     | (none — derived by caller)                | (none — derived by caller)   | `manifold/measure/distance(va, vb)` |
| Heat diffusion (time series) | `field::generate::heatDiffusion(...)` | `mctx.solve.heat(...)`       | under-the-hood (deliberately not a field) |
| Random 1-form            | `field::generate::randomDecomposed1Form` | (not directly grouped)       | utility, under-the-hood |
| Whitney interpolation    | `field::interp::whitney(...)`             | (via _flat)                  | internal to `field.gradient` / `divergence` |
| Scalar gradient (Whitney)| `field::op::gradient(...)`                | (via _flat)                  | `field.gradient()` on scalar+vertex |
| BFF UV                   | `manifold::geometry::parametrization::compute` | `mctx.uv.bff()`         | `manifold/parametrization/uv` |
| Cross / line field       | `manifold::connection::smoothFace / smoothVertex` | `mctx.interpolate.smoothFaceField / smoothVertexField` | `manifold/parametrization/cross_field / line_field` |
| Stripe pattern           | `field::stripes::compute(...)`            | `mctx.uv.stripe(...)`        | `manifold/parametrization/stripes` |
| Trivial direction field  | `manifold::connection::trivial(...)`      | `mctx.interpolate.directionField(...)` | **open question** (schema §11.1 flags it) |
| Vector heat transport    | `manifold::transport::parallel(...)`      | `mctx.interpolate.transport(...)` | under-the-hood for global tangent frame |
| Log map                  | `manifold::transport::logMap(...)`        | `mctx.uv.logMap(...)`        | not exposed |
| Karcher mean             | `manifold::transport::findCenter(...)`    | `mctx.query.center(...)`     | not exposed |
| Curvatures               | `manifold::geometry::curvatures(ctx)`     | `mctx.measure.curvature()` *(bundle)* | per-slot: `manifold/geometry/curvature/{gaussian, mean, principal_min, principal_max, principal_dir_max}` |
| Vertex normals           | `manifold::geometry::vertexNormals`       | `mctx.measure.normal(type)`  | `manifold/geometry/vertex/normals` |
| Face frames              | `manifold::geometry::faceFrames`          | `mctx.measure.frame()`       | `manifold/geometry/face/frames` |
| Isolines                 | `field::viz::isolines(...)`               | `mctx.query.isoline(...)` *(single-level)* | `field.isolines(...)` |
| Streamlines              | `field::viz::streamlines(...)`            | (via _flat)                  | `field.streamlines(...)` on vector+face field |

---

## Naming differences

| What | A — C++ proposal | B — May 6 shim | C — today's schema |
|---|---|---|---|
| Hodge stars              | `hodge0`, `hodge1`, `hodge2`, `hodge1Inverse` (preserves existing struct field) | `star0`, `star1`, `star2`, `star1Inverse` (renamed at JS boundary) | `hodge0`, `hodge1`, `hodge2`, `hodge1_inverse` |
| Vertex Voronoi areas     | `vertexAreas` (struct field, untouched) | not directly exposed in 6-group tree | `vertex_areas` |
| Top-level groups under manifold | 8 + 3 gaps (`ops`, `eigen`, `solve`, `geometry`, `distance`, `transport`, `connection`, `decompose`, *gaps:* `query`, `transform`, `health`) | 6 (`solve`, `operator`, `query`, `measure`, `uv`, `interpolate`) | 10 (`attributes`, `core`, `embedding`, `topology`, `geometry`, `operators`, `eigen`, `parametrization`, `query`, `measure`) |
| Where Hodge decomposition lives | `manifold::decompose::hodge` (its own group) | `solve.hodge` (PDE solvers) | `field.hodge()` (operation on a 1-form field) |
| Where curvature lives    | `manifold::geometry::curvatures` (bundle) | `measure.curvature()` (bundle) | `manifold/geometry/curvature/{gaussian, mean, …}` (per slot) |
| Where direction fields live | `manifold::connection::{trivial, smoothFace, smoothVertex}` | `interpolate.{directionField, smoothFaceField, smoothVertexField}` | `manifold/parametrization/{cross_field, line_field}` + open question for `trivial` |
| Where vector heat lives  | `manifold::transport::{parallel, extendScalar, logMap, findCenter}` | split across 3 groups: `interpolate.{transport, extend}` + `uv.logMap` + `query.center` | under-the-hood; not exposed |
| Field as a namespace     | yes — `nxr::field::*` (5 sub-groups) | no — fields are inputs/outputs to ops on `mctx` | yes — `field/*` (parallel to `manifold/`); plus `Field` object via `nxr.createField(ctx, …)` |
| Time-series              | `nxr::time::*` reserved (Phase 1) | not addressed | not addressed |
| Helper scalars (V, E, F) | in result structs | as group counts on the wrapper (`nV()`, `nE()`, `nF()`) | in `attributes` (derivable from input shapes) |

---

## What each design has that the others don't

**A only — the C++ proposal:**

- Full source-tree reorganization (`src/{core,manifold,field}/...`) with header / source layout, `git mv` plan, CMake updates.
- `core::` namespace consolidating cross-cutting types (`Error`, `CancellationToken`, `ProgressObserver`, storage helpers).
- `nxr::time::*` reserved for FFT / wavelets / filters / windows (Phase 1).
- `decompose::` as Hodge's home — separated from PDE solvers and from field generators.
- Full backward-compat plan via `compute.h` shim with `using` aliases.
- Gaps explicitly marked: `manifold::query::`, `manifold::transform::`, `manifold::health::`, plus `solve::heat`.
- "Namespace discipline" CI rule preventing cross-namespace includes.
- Migration phases (0 = rename, 1 = `time::`, 2 = manifold gaps, 3+ = future namespaces) with verification plan (`bench:all && bench:diff`, native tests, WASM smoke, MEX smoke, three.js gallery, cortical-flow Electron).

**B only — the May 6 JS shim:**

- Actually shipping in WASM, used today.
- `mctx._flat` escape hatch: same underlying `ContextWrapper`, both surfaces callable.
- Free-function alternative form `nxr.nxr.manifold.{group}.{method}(mctx, …)`.
- TODO stubs that return `{method: 'todo', name}` and emit one-shot `console.warn`s.
- Renames `hodge*` to `star*` at the JS boundary.
- TypeScript declarations covering both surfaces; `tsc --strict` clean.
- README documents the namespace tree, one-line example per group, caching notes.

**C only — today's schema:**

- `field` as a first-class object with explicit `(kind, domain)` axes — concept absent from A and B (A has a `field::` *namespace* but not a Field *object*).
- `parametrization/` unifies global UV (BFF) with local coordinate fields (cross, line, stripes) under one branch — A splits these across `geometry::parametrization` + `connection::smooth*` + `field::stripes`; B splits them across `uv` + `interpolate`.
- `embedding/` for three.js Object3D semantics (bounding box + 4×4 transform) — A and B don't model world-placement.
- `geometry/singularities` — 2 reference vertex indices anchoring the global tangent frame.
- `applies_when` preconditions on field operations (predicate over field attributes; bindings throw `InvalidInput` on violation).
- `query` (returns loci) / `measure` (returns scalars) clean split — A puts these together under `distance::` + `geometry::`; B fuses them in `query` + `measure` groups but with overlapping semantics.
- Complex-number storage convention (`complex_interleaved`) baked into the dtype enum for n-RoSy fields.
- Group-level vs leaf-level `attributes` blocks throughout — every node carries metadata applicable to its scope.
- Symbolic memory-size formulas (`8 * V * 3`, `12 * nnz(operators/d0)`) for every slot.
- Hand-written JSON Schema validator (`*.meta.schema.json`) that catches drift at CI time, with shared `$defs` in `common.meta.schema.json` for forward-extensibility.

---

## Decision axes

If you want to pick one canonical design, the axes that matter most:

1. **8+5 (A) vs 6 (B) vs 10+1 (C) groups.** Fewer groups = each group is busier, easier to discover but harder to keep clean. More groups = clearer semantics but more navigation. A and C are close in spirit; B is the most compressed.
2. **Verb-organised (B) vs noun-organised (A, C).** B uses `solve / operator / query / measure / uv / interpolate` — these are verbs / outputs. A and C use `ops / eigen / geometry / operators / eigen / geometry` — these are nouns / contents. Verb organisation is more "what can I do?"; noun organisation is more "what does the manifold contain?"
3. **Curvature as bundle (A, B) vs per-slot (C).** Reading one value: bundle is wasted compute; per-slot is precise. Reading all five: bundle is one C++ call; per-slot needs memoisation.
4. **Hodge naming.** `hodge*` (A, C) preserves the existing C++ struct field name; `star*` (B) renames at the JS boundary. The today-conversation explicitly settled on `hodge*`.
5. **Field as namespace (A) vs Field as object (C) vs no Field (B).** A's `field::` is just where field-related *functions* live. C's `Field` is an *instance* — values + attributes + operations with preconditions. B has neither.
6. **Curvature / direction-field / stripe placement.** A scatters across `geometry / connection / stripes`; B groups under `measure` / `interpolate` / `uv`; C unifies under `parametrization`.
7. **Future `time::` namespace.** Only A reserves it. C deliberately stays mesh-focused for v1.
8. **Implementation cost.** A = large C++ rename (1–2 days mechanical + 1 day bench validation). B = already done. C = JS shim refactor (a few hours) + binding extension (Field constructor, per-slot getters).

---

## Recommended next move

Pick one of:

- **Adopt C as canonical, supersede A and B.** Update `api-refactor-proposal.md` to reference C's tree (or mark it superseded). Refactor the May 6 shim to match C — keep `createManifoldContext` as an alias for backward compat, but the primary surface becomes C's `createContext` + `createField`. Estimated effort: ~half a day of JS-side rework plus updating the proposal doc.
- **Adopt A as canonical, evolve B and C toward it.** Keep the C++ proposal as the long-term target; treat B as transitional; treat C as a sub-spec describing only the manifold side, due for revision to match A's exact group naming. Estimated effort: low in the short term, but you still owe the C++ refactor (1–2 days mechanical, plus bench).
- **Reconcile A and C into a single revised proposal.** Take C's structural insights (Field as object, parametrization unification, query/measure split, applies_when, embedding/) and merge them into A's broader plan (C++ rename, time::, source layout, migration phases). Retire B's grouping once A's binding strategy lands. Estimated effort: ~half a day to write the unified proposal; the C++ work remains.

My read: **C is the most recent considered design and the only one with an explicit Field object**, which is the only place fields-as-data really fit in a system where heat methods are under-the-hood. A is the most comprehensive long-term plan but doesn't have the Field object. B is the most pragmatic but locks in a verb-grouped structure that contradicts both A and C.

A reconciliation that promotes C's structure into A's broader plan is probably the right outcome.
