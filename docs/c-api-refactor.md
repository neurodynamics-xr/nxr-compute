# nxr-compute C API refactor — working document

**Status:** in progress. This document is built up incrementally, one
decision at a time. Each section reflects a choice that has been made
explicitly; gaps are deliberate — they mean "not yet discussed."

**Relation to other docs:**

- `api-design-comparison.md` — three-way comparison of the design
  shapes that have been in flight (A: C++ proposal, B: JS shim, C:
  schema). Background reading.
- `api-refactor-proposal.md` — the earlier full-tree proposal (Design
  A). Treated as input, not as a settled target. This document
  supersedes it once we reach a consistent design.

**Scope:** the C++ API surface of `nxr-compute` (`include/nxr/*.h`).
All downstream bindings (Node addon, WASM/Embind, MEX, CLI) are
expected to derive mechanically from this surface — i.e. the C++ tree
is the single source of truth. Binding-specific divergence (renames,
verb-grouped sugar) is explicitly out of scope.

---

## Top-level namespaces

Two top-level namespaces under `nxr::`:

```
nxr::
├── manifold::    [mesh / differential geometry]
└── field::       [scalar / vector / 1-form fields defined on a manifold]
```

`field` is split from `manifold` so that "what the surface *is*"
(topology, geometry, operators, eigendecomposition) stays separate
from "what lives *on* the surface" (heat distributions, gradients,
direction fields, isolines). The same field type should be reusable
across different manifolds; the same manifold should host arbitrarily
many fields.

Cross-cutting infrastructure (`Error`, `CancellationToken`,
`ProgressObserver`, storage helpers) — namespace not yet decided.

---

## Inside `manifold::` — what's settled

Five sub-namespaces and one class. Each sub-namespace exposes one
distinct kind of thing.

```
nxr::manifold::
├── Manifold                  [class — owns the GC mesh + geometry + nxr cache slots]
├── topology::                [indexing & localization — addresses, counts, adjacency, invariants]
├── geometry::                [intrinsic per-element quantities — areas, lengths, normals, curvatures, tangent bases]
├── ops::                     [matrix-valued operators — Laplacians, mass, DEC, connection]
├── query::                   [user-selected loci — points, polylines, regions]
└── measure::                 [scalar metrics of query loci — coords, lengths, areas]
```

The triplet **geometry / query / measure** captures three distinct
concerns:

- **geometry** — what GC computes intrinsically about every element of
  the surface. Per-element scalars/vectors. Always present once the
  mesh is set; no user choice involved.
- **query** — geometric loci that the user has pointed at: a vertex
  (point), a polyline between two vertices (edge-flip geodesic), a
  region (heat-distance level set centered on a vertex). The output is
  *the locus itself* (the geometric primitive — vertex handle,
  polyline, set of faces).
- **measure** — the *scalar metric* of a query locus: the coordinates
  of a point, the length of a line, the area of a region. Same input
  parameters as the matching `query` accessor; different return type.

`measure::*` is built on top of `query::*` plus pieces of
`geometry::*` (face areas etc.). The two are paired by signature:
`query::area(m, v, level)` returns a region; `measure::area(m, v, level)`
returns the scalar area of that region.

The names follow nxr's signal-processing framing: **the manifold is
the domain, the field is the signal, measurements characterize the
domain so the signals on it can be analysed.** `topology` answers
*where* on the domain; `measure` answers *what value* at that
location; `ops` produces the matrix transforms used to compute,
filter, and decompose signals.

### Why this split

The split mirrors GC's design at the return-type level but reframes
two of the three namespaces. The rule is mechanical: **the GC return
type decides the namespace**.

| Concern | GC source | Return type | nxr-compute namespace |
|---|---|---|---|
| Connectivity, addresses (Vertex / Edge / Face / Halfedge handles), counts, adjacency queries, topological invariants, (potentially) mesh edits | `ManifoldSurfaceMesh` | element handles, ints, bools | `manifold::topology` |
| Per-element scalars/vectors measured on the surface: positions, areas, normals, lengths, cotan weights, curvatures, tangent bases | `VertexPositionGeometry` `require*` fields | `VertexData<T>`, `EdgeData<T>`, `FaceData<T>`, `HalfedgeData<T>` (i.e. `MeshData<E,T>`) | `manifold::measure::{vertex,edge,face,curvature,...}` |
| Matrix-valued operators: `d0`, `d1`, `hodge0/1/2`, cotan Laplacian, mass matrices, connection Laplacians, Crouzeix-Raviart variants | `IntrinsicGeometryInterface` `require*` fields | `Eigen::SparseMatrix<double>` / `SparseMatrix<complex<double>>` | `manifold::ops` |

The earlier working name `manifold::geometry` (D2) is **superseded by
`manifold::measure`** — see D8 for the signal-processing framing that
motivates the rename.

Two structural properties this preserves:

1. **One topology, many geometries.** GC's `VertexPositionGeometry`
   takes a `SurfaceMesh&` by reference, so multiple geometries can
   share the same topology (original vs. deformed positions, rest vs.
   current state, time-varying surfaces). Mirroring the split at the
   namespace level preserves the option to expose this later.
2. **Topology-only operations stay free of geometry dependencies.**
   Mesh validation (`isManifold`, `eulerCharacteristic`, `genus`),
   adjacency queries, and BFS-style traversals are purely
   topological. Routing them through `manifold::topology` keeps them
   honest — no `VertexPositionGeometry` required.

### Caching architecture inside `manifold::ops::`

The operators namespace uses a **two-tier cache** matching who computes
what:

```
┌───────────────────────────────────────────────────────────────┐
│  manifold::ops::*  (accessors, return const SparseMatrix&)    │
│                                                                │
│   ┌─────────────────────────┐    ┌────────────────────────┐   │
│   │  Tier 1: GC's cache     │    │  Tier 2: nxr's cache   │   │
│   │  (require* system)      │    │  (per-context slots)   │   │
│   │                         │    │                        │   │
│   │  d0, d1                 │    │  symmetrized cotan L   │   │
│   │  hodge0/1/2 + inverses  │    │  barycentric mass      │   │
│   │  raw cotanLaplacian     │    │  consistentFEM mass    │   │
│   │  GC mass matrices       │    │  connectionLaplacian   │   │
│   │  GC connection Laplac.  │    │    (keyed by nSym,     │   │
│   │  Crouzeix-Raviart ops   │    │     format, reg)       │   │
│   └─────────────────────────┘    └────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

- **Tier 1: GC's `require*` system.** Used for accessors that are direct
  passthroughs of a GC quantity (`d0`, `d1`, `hodge0/1/2`, raw
  `cotanLaplacian`, GC's own mass matrices). The accessor calls
  `geometry.requireXxx()` once and returns `geometry.xxx` by const
  reference. No nxr-side storage.
- **Tier 2: nxr-side per-context cache slots.** Used for accessors that
  do post-processing on top of GC (`stiffness` = `(L + Lᵀ)/2`
  symmetrization), build something GC doesn't (Barycentric and
  ConsistentFEM mass variants), or are parameterized (connection
  Laplacian with `nSym`, `format`). The slot is a `unique_ptr` or
  `optional<SparseMatrix>` on a per-context cache object; populated
  lazily on first access; returned by const reference on subsequent
  calls.

Both tiers respect the same contract: **compute on first access, return
on subsequent**, lifetime bound to the context.

### What's deprecated by this

The current return-struct API is superseded:

| Today | Replaced by |
|---|---|
| `assembleMeshOperators(ctx)` returns `MeshOperators` struct (stiffness + mass + vertexAreas + normals + …, all copied) | Individual `manifold::ops::*()` accessors, each cached on its own |
| `assembleDECOperators(ctx)` returns `DECOperators` struct (6 sparse matrix copies from GC) | `manifold::ops::{d0, d1, hodge0, hodge1, hodge2, hodge1Inverse}()` returning `const SparseMatrix&` from GC's cache |
| Binding-side `shared_ptr<MeshOperators>` / `shared_ptr<DECOperators>` on `ContextHolder` | Removed — the two-tier cache replaces both |
| Binding-side `std::map<CLKey, ConnectionLaplacian>` for connection Laplacian | Moved into core as the tier-2 keyed slot |

### What goes where (placeholder — to be filled in as we discuss)

`manifold::topology::`
- (TBD) element addressing: `vertex(i)`, `face(i)`, `edge(i)`, `halfedge(i)` and the inverse `indexOf(...)`
- (TBD) counts: `nV()`, `nE()`, `nF()`
- (TBD) adjacency queries: neighbouring vertices/edges/faces of an element
- (TBD) topological invariants: `eulerCharacteristic`, `genus`, `isManifold`, `hasBoundary`, `isOriented`
- (TBD) mesh edits (collapse / split / flip), if ever exposed (out of scope for v1 — see D6)

`manifold::measure::`
- (TBD) **element-grouped measurements** (per the element they live on):
  - `vertex::{positions, normals, dualAreas, tangentBasis, ...}`
  - `edge::{lengths, cotanWeights, dihedralAngles, ...}`
  - `face::{areas, normals, tangentBasis, frames, ...}`
- (TBD) **semantic-family measurements** (where the family is more natural than the element):
  - `curvature::{gaussian, mean, principalMin, principalMax, principalDirMax, ...}`
  - possibly more semantic families as they accrue (e.g. transport vectors)

`manifold::ops::`
- (TBD) `d0`, `d1`, `hodge0`, `hodge1`, `hodge2`, `hodge1Inverse` (passthroughs)
- (TBD) `cotanLaplacian` (passthrough) and `laplacian.cotan` / `stiffness` (symmetrized — tier-2 derived)
- (TBD) `mass.vertex.{lumped, galerkin, voronoi, barycentric, consistentFEM, ...}`
- (TBD) `mass.face.{galerkin, ...}`, `mass.edge.{crouzeixRaviart, ...}`
- (TBD) `laplacian.{vertex, face, edge}` and `laplacian.connection.{vertex, face, edge}` (the last takes `nSym`, `format`)
- (TBD) interaction with the `require*()` lifetime model — open whether nxr exposes `release()` / `evict()` or treats cache as permanent until context destruction

---

## Decisions log

Append-only. Each entry: what was decided, why, what it rules out.

### D1 — Two top-level namespaces: `manifold` and `field`

- **Decided:** `nxr::manifold` and `nxr::field` are sibling top-level
  namespaces.
- **Why:** Things that the surface *is* vs. things that live *on* the
  surface are different concerns with different lifetimes and different
  reuse patterns. Same separation already exists in MATLAB's `+bct`
  toolbox (`+bct.+manifold`, `+bct.+field`) and in the design-C schema.
- **Rules out:** A single flat `nxr::compute::` umbrella. Also rules
  out putting fields inside the manifold (e.g. `manifold::field`),
  which would imply 1:1 ownership.

### D2 — `manifold::topology` wraps `ManifoldSurfaceMesh`; `manifold::geometry` wraps `VertexPositionGeometry`

- **Decided:** The two namespaces under `manifold::` correspond
  directly to geometry-central's two foundational classes.
- **Why:** GC made this split deliberately — connectivity and embedding
  are independently meaningful, and the `require*()` quantity cache
  lives on the geometry, not the mesh. Mirroring the split at the
  namespace level keeps our API legible and lets us preserve GC's
  one-topology-many-geometries pattern.
- **Rules out:** A combined `manifold::mesh` namespace that owns both.
  Also rules out putting derived quantities (cotan weights, curvatures)
  under `topology` — those depend on positions and live in `geometry`.

### D3 — `manifold::ops::*` is the namespace for matrix-valued operators

- **Decided:** A third sub-namespace under `manifold::`, alongside
  `topology` and `geometry`, dedicated to matrix-valued operators
  (Laplacians, mass matrices, DEC operators `d0` / `d1` / `hodge*`,
  connection Laplacians, Crouzeix-Raviart variants). It is an
  ergonomic accessor layer over geometry-central's matrix operators,
  not a reimplementation.
- **Why:** GC exposes 14+ matrix-valued operators on
  `IntrinsicGeometryInterface`; nxr-compute exposes ~5 today. A flat
  namespace doesn't scale to the full set, and the operators don't
  naturally fit under `geometry::` (which is per-element data) or
  `topology::` (which is purely connectivity). A dedicated `ops::`
  with hierarchical naming (`mass.vertex.lumped`,
  `laplacian.connection.vertex`) groups related families and leaves
  room for the full GC surface plus nxr-specific variants.
- **Rules out:** putting operators on `manifold::geometry::*` next to
  per-element scalars. Also rules out keeping the current return-
  struct pattern (`MeshOperators`, `DECOperators`) — see D7.

### D4 — Split rule between `geometry::` and `ops::` is by GC return type

- **Decided:** The structural boundary is mechanical:
  - GC return type `MeshData<E,T>` (`VertexData<T>`, `EdgeData<T>`,
    `FaceData<T>`, `HalfedgeData<T>`, `CornerData<T>`) →
    `manifold::geometry::{vertex,edge,face,...}::*`.
  - GC return type `Eigen::SparseMatrix<T>` →
    `manifold::ops::*`.
- **Why:** GC's own type system provides the boundary; no human
  judgment is needed for placement. `faceAreas` (per-face scalar) and
  `cotanLaplacian` (sparse matrix) are both "face-related," but the
  type difference is the structural difference — they live in
  different namespaces under this rule.
- **Rules out:** putting `vertexNormals` (a `VertexData<Vector3>`) in
  `manifold::ops::*` just because it's used during operator assembly.
  Also rules out putting `cotanLaplacian` in `manifold::geometry::*`
  just because it's "derived from positions."

### D5 — Anything intrinsic to the manifold lives in `manifold::*`; caller-supplied data lives in `field::*`

- **Decided:** The container types GC uses (`VertexData<T>` etc.)
  carry no semantic marker — the same template is used for GC's
  cached structural quantities and for user-defined per-element data.
  nxr-compute imposes the split:
  - Anything intrinsic to the surface (positions, areas, normals,
    cotan weights, curvatures, tangent bases, …) — derived purely
    from connectivity and positions — lives in `manifold::*`.
  - Anything caller-supplied (heat sources, user scalars, electrical
    potentials, time series, eigenmode coefficients, …) — owned by
    the caller or computed by an `ops` / `solve` call and returned
    — lives in `field::*`.
- **Why:** Keeps the rule sharp: *if GC computed it about the
  surface, it's in `manifold`. If the caller brought it (or computed
  it by operating on a field), it's in `field`.*
- **Rules out:** putting user-defined fields under
  `manifold::geometry::vertex::*` even though they share the same
  container type. Caller data never appears under `manifold::*`.
- **Edge case to flag:** Curvature. GC stores Gaussian curvature as
  `VertexData<double>` on the geometry — by D4, that places it in
  `manifold::geometry::vertex::gaussianCurvature`. Consumers who want
  to visualize curvature as a field do an explicit promotion (e.g.
  `field::fromManifoldQuantity(ctx, "gaussianCurvature")`). The rule
  is preserved: the *quantity* is intrinsic; treating it as a *field*
  is a caller decision.

### D6 — `ComputeContext` is immutable in v1; cache lifetime = context lifetime

- **Decided:**
  - `ComputeContext` is immutable after construction. Vertices,
    faces, and positions set once at construction. No mesh edits
    (collapse / split / flip). No position mutation. A caller wanting
    deformed geometry creates a new context.
  - Unparameterized ops return `const Eigen::SparseMatrix<double>&`
    from a single-slot cache.
  - Parameterized ops (e.g. `connectionLaplacian(nSym, format)`) take
    parameters and return from a keyed cache (or `shared_ptr` for
    large results), via a `std::map`-like structure keyed on the
    parameter tuple.
  - Cache lifetime equals context lifetime. No manual `release()` /
    `evict()` in the v1 API; caches free when the context dies.
- **Why:** Mesh and position mutation are the only ways to invalidate
  cached results during a context's lifetime. Ruling them out for v1
  eliminates the entire class of stale-cache correctness problems and
  matches how nxr-compute is already used (every binding builds a
  fresh `ComputeContext` per mesh and discards on mesh change).
- **Rules out:** in-place mesh edits, position deformation, manual
  cache eviction. Also rules out a multi-geometry `ComputeContext`
  (one topology, many geometries) at the type level for v1 — keep
  the bundled `topology + geometry` shape we have today, defer the
  GC-style separation as a v2 question.

### D7 — Two-tier caching: GC's `require*` for what GC computes; nxr-side per-context slots for what nxr computes; ops accessors return `const SparseMatrix&`

- **Decided:**
  - **Tier 1 (GC's cache).** For accessors that are direct pass-
    throughs of a GC quantity — `d0`, `d1`, `hodge0/1/2`,
    `hodge1Inverse`, raw `cotanLaplacian`, GC's own mass matrices,
    GC's connection Laplacians, Crouzeix-Raviart variants — the
    accessor calls `geometry.requireXxx()` once and returns
    `geometry.xxx` by const reference. No nxr-side storage.
  - **Tier 2 (nxr's cache).** For accessors that do post-processing
    on top of GC, build something GC doesn't, or take parameters —
    symmetrized stiffness `(L + Lᵀ)/2`, the Barycentric and
    ConsistentFEM mass variants, the connection Laplacian with
    `nSym` + Real2N expansion + regularization — nxr-compute owns a
    cache slot on a per-context cache object. Slot is populated
    lazily on first access; returned by const reference on
    subsequent calls.
  - **Ops accessors return `const SparseMatrix<double>&` by
    default.** Eliminates the matrix copies in the current
    `assembleMeshOperators` / `assembleDECOperators` pattern.
    Consumers that need an owned copy (notably the eigensolver,
    which hands `K` and `M` to Spectra's `SymGEigsShiftSolver`) take
    an explicit copy at the call site.
  - **Legacy return-struct API is deprecated.**
    `assembleMeshOperators` / `assembleDECOperators` and the
    binding-side `shared_ptr<MeshOperators>` / `shared_ptr<DECOperators>`
    holders go away. The binding-side `std::map<CLKey, ConnectionLaplacian>`
    cache moves into the core as the tier-2 keyed slot.
- **Why:** Verified by reading `src/mesh_operators.cpp:220-234`: the
  current code triggers GC's cache and then *copies* 6 sparse
  matrices out into the `DECOperators` return struct on every call.
  The two-tier pattern eliminates these copies. The split matches
  who-computes-what: GC owns the cache for what GC computes; nxr
  owns the cache for what nxr computes. No single layer is forced to
  duplicate the other.
- **Rules out:** a single unified nxr-side cache that duplicates GC's
  state. Also rules out copying on every accessor call (consumers
  that need owned matrices copy explicitly — the exception, not the
  default).
- **Caveat:** the hidden-first-call-cost risk and the GC-bundle
  cascade (one DEC access builds all 8 matrices) are inherited from
  GC's design. Mitigated by accessor naming convention (parentheses
  signal "may compute") and per-accessor docs of what the trigger
  cascades into.

### D8 — `manifold::geometry` (D2) is renamed to `manifold::measure`; nxr's framing is signal-processing

- **Decided:**
  - The sub-namespace previously called `manifold::geometry`
    (per D2 / D4) is renamed to `manifold::measure`. Its content is
    unchanged — same wrappers over the same GC `MeshData<E,T>`
    quantities — but the name reflects what the namespace exposes
    rather than which GC class it wraps.
  - Inside `manifold::measure::*`, two organizing principles coexist:
    - **Element-grouped:** `measure::vertex::*`, `measure::edge::*`,
      `measure::face::*` for measurements naturally addressed by
      element type (areas, lengths, normals, tangent bases, ...).
    - **Semantic-family-grouped:** `measure::curvature::{gaussian,
      mean, principalMin, principalMax, principalDirMax}` for
      measurements whose semantic family is more discoverable than
      their element type. New semantic groups can be added as the
      surface grows.
  - The split between `manifold::topology` (indexing / localization)
    and `manifold::measure` (values at addresses) is also drawn
    explicitly. `topology` answers *where on the surface* (element
    handles, counts, adjacency, invariants); `measure` answers
    *what value at that location* (the per-element scalars and
    vectors GC's `require*` system computes).
- **Why (signal-processing framing):** nxr-compute is being built
  as a signal-processing library on manifolds. The vocabulary that
  fits this domain:
  - **Manifold** = the domain on which signals are defined.
  - **Field** = a signal living on the manifold (caller-supplied or
    solver-produced, see D5).
  - **Measurement** = an intrinsic quantity of the manifold that
    characterizes the domain — used to set up and interpret the
    signals on it (areas for L² inner products, curvature for shape
    descriptors, normals for projection / display, lengths for
    metric queries).
  - **Operator** = a matrix transform applied to signals (Laplacian
    for smoothing / spectral analysis, mass for inner products,
    `d0` / `d1` / `hodge*` for exterior calculus, connection
    Laplacian for direction fields).
  - The names `topology` / `measure` / `ops` map directly to the
    things a signal-processing user reaches for; `geometry` was
    accurate but conflated indexing with measurement.
- **Rules out:** `manifold::geometry` as a public name. Also rules
  out putting curvature under an element-grouped path (e.g.
  `manifold::measure::vertex::gaussianCurvature`) when the curvature
  family is more discoverable as a semantic group — though the
  element-grouped path may exist as an alias if useful.
- **Revises:** D2 (the second half — the `geometry` half — is
  renamed). D4's split rule still applies: `MeshData<E,T>` →
  `manifold::measure::*`. D5 unchanged in spirit: intrinsic
  measurements (`manifold::measure::*`) vs. caller-supplied signals
  (`field::*`).

### D9 — `ctx` is the parameter name for `Manifold&`; the C++ API is free-functions-in-namespaces, not methods-on-handle

- **Decided:**
  - Every function in the `nxr::manifold::*` / `nxr::field::*`
    namespaces takes a `Manifold& m` as its first parameter:
    `nxr::manifold::ops::d0(m)`,
    `nxr::manifold::geometry::face::areas(m)`,
    `nxr::manifold::ops::laplacian::connection::vertex(m, nSym, format)`.
  - The variable name `m` replaced `ctx` during Chunk A's rename.
    `ctx` had no semantic content; `m` is short, conventional, and
    refers to the Manifold instance.
  - The C++ API surface is the **namespace tree**, not member
    methods on `Manifold`. Bindings are free to wrap with method-
    on-handle ergonomics (`mctx.manifold.ops.d0()` calling through
    to `nxr::manifold::ops::d0(mctx)` underneath) — that's a binding
    choice, not the C++ contract.
- **Why:** The persistent handle is unavoidable (D6 + D7 require
  somewhere to own the GC mesh + geometry + nxr cache across calls).
  Free-functions-with-handle matches GC's own shape (GC functions
  take `Geometry&`), keeps the namespace tree as the primary API
  surface (per D3 / D4), and lets multiple Manifolds coexist
  trivially. Method-on-handle would force the namespace tree to be
  documentation scaffolding rather than the actual API.
- **Rules out:** thread-local or global current-Manifold dispatch
  (breaks reentrancy). Also rules out method-on-handle as the
  primary C++ API.

### D10 — Class location: `nxr::manifold::Manifold`

- **Decided:** The class formerly known as `ComputeContext` (renamed
  to `Manifold` during the Chunk-A pass) lives inside the
  `nxr::manifold::` namespace. Full qualified name:
  `nxr::manifold::Manifold`.
- **Why:** "Manifold" is the semantic name for what the class
  represents (the per-mesh handle that owns GC mesh + geometry + nxr
  cache). Co-locating with the `manifold::` sub-namespace tree keeps
  the type and the functions that operate on it in one place. The
  doubled token (`manifold::Manifold`) is a standard C++ idiom
  (`std::vector` inside `std::`, `boost::filesystem::path` inside
  `boost::filesystem::`).
- **Rules out:** `nxr::Manifold` (top-level) — that would conflict
  with D1's `nxr::manifold::` namespace declaration.

### D11 — Three sibling sub-namespaces: `geometry`, `query`, `measure` (revises D8)

- **Decided:** Inside `manifold::*`, three sub-namespaces serve
  distinct roles:
  - `manifold::geometry::*` — intrinsic per-element data (areas,
    lengths, normals, curvatures, tangent bases). Wraps GC's
    `MeshData<E,T>` quantities. Always present once the mesh is set.
  - `manifold::query::*` — user-selected geometric loci. Returns the
    primitive itself (vertex handle, polyline, region). Takes user
    parameters (vertex ids, distance thresholds, etc.).
  - `manifold::measure::*` — scalar metrics of query loci. Same
    parameters as the matching `query::*` accessor; returns a number
    (coordinates, length, area).
- **Pairing:** `query::point` ↔ `measure::point`, `query::line` ↔
  `measure::line`, `query::area` ↔ `measure::area`. The query returns
  the locus, the measure returns the scalar.
- **Revises:** D8. D8 had renamed `geometry → measure` under the
  signal-processing framing. D11 keeps `geometry` as D2 originally had
  it (intrinsic per-element data) AND reassigns `measure` to the new
  query-pair role. D8's signal-processing vocabulary (manifold =
  domain, field = signal, ops = transforms acting on signals) still
  applies at the namespace level.
- **`query::line` mechanism:** edge-flip geodesic path (the polyline
  primitive in GC, available today via `tracePath`). Heat-method
  distance is a separate concept exposed under `manifold::distance::*`.

### D12 — No visualization layer; `field::extract::*` instead of `field::viz::*`

- **Decided:** nxr-compute is a computation library, not a
  visualization library. The library returns geometric data
  (polylines, regions, fields, scalars) but never visualization
  primitives (colors, geometry buffers, render commands, materials).
  Visualization belongs in the consuming app.
- **Rules out:** A `field::viz::*` namespace (which the earlier
  proposal had carried over from MATLAB's `+bct.+field.+viz`).
  Functions that compute geometric loci derived from fields
  (`isolines`, `streamlines`) live under `field::extract::*` —
  *extract* signals "pull a geometric primitive out of a field," with
  no rendering concern.

### D13 — Full namespace placement for all current public symbols

- **Decided:** Mapping from current `nxr::compute::*` symbols to the
  target tree. Renamed-but-not-yet-relocated symbols (Chunk A's work)
  noted in parentheses.

  `manifold::eigen::`
  - `EigenResult`, `solve` (← `solveEigenmodes`), `normalize` (← `normalizeEigenmodes`), `removeDC`

  `manifold::solve::`
  - `poisson` (← `solvePoisson`)

  `manifold::distance::`
  - `heat` (← `geodesicDistance`), `signed` (← `signedHeatDistance`),
    plus the solver classes `HeatGeodesicSolver`, `SignedHeatSolver`,
    `SignedHeatLevelSet`

  `manifold::transport::`
  - `parallel` (← `vectorHeatTransport`), `extendScalar`
    (← `vectorHeatExtendScalar`), `logMap` (← `vectorHeatLogMap`),
    `findCenter` (← `vectorHeatFindCenter`), plus `VectorHeatSolver`,
    `LogMapResult`, `LogMapStrategy`. **The `vectorHeat*` prefix is
    dropped**; the family name is `transport`.

  `manifold::connection::`
  - `trivial` (← `directionField`), `smoothFace` (← `smoothFaceField`),
    `smoothVertex` (← `smoothVertexField`), plus
    `DirectionFieldResult`, `SmoothVertexFieldResult`

  `manifold::decompose::`
  - `hodge` (← `hodgeDecompose`), plus `HodgeResult`

  `manifold::parametrization::`
  - `bff` (← `uvCoordinates`), `stripes::compute` (← `stripePattern`),
    `stripes::computeFreq` (← `stripePatternFreq`), plus
    `StripePatternResult`

  `manifold::geometry::`
  - `vertex::{positions, normals, dualAreas, tangentBasis, …}` (existing accessors + GC fields)
  - `edge::{lengths, cotanWeights, dihedralAngles, …}`
  - `face::{areas, normals, frames, tangentBasis, …}` (← `faceFrames` → `face::frames`; `FaceFrames`)
  - `curvature::{gaussian, mean, principalMin, principalMax, principalDirMax}` (← `curvatures` returns a bundle today; split into per-slot accessors)
  - `NormalType` (← from current `vertexNormals(m, type)`)

  `manifold::ops::`
  - `d0`, `d1`, `hodge0`, `hodge1`, `hodge2`, `hodge1Inverse` (already passthroughs)
  - `laplacian::{cotan, connection::{vertex, face, edge}}`
  - `mass::{voronoi, barycentric, consistentFEM, galerkin}` (← `MassMatrixVariant` enum becomes the leaf names)
  - `CholeskyCache` (factor cache — stays internal or exposed under `ops::factor::*`; TBD)
  - `ConnectionLaplacianOptions`, `ConnectionLaplacian`, `ConnectionDomain`, `ConnectionLaplacianFormat`

  `manifold::query::`
  - `point(m, v)` — new — trivial
  - `line(m, va, vb)` — wraps `tracePath` (edge-flip geodesic)
  - `area(m, v, level)` — new composition (heat distance + iso-extract)

  `manifold::measure::`
  - `point(m, v)` — new — returns `Vector3` from `geometry::vertex::positions`
  - `line(m, va, vb)` — new — polyline length of `query::line`
  - `area(m, v, level)` — new — face-area sum inside `query::area`

  `manifold::topology::`
  - `nV`, `nE`, `nF` (currently methods on `Manifold`; could stay as methods or become free functions)
  - (future) adjacency queries, invariants

  `field::generate::`
  - `delta`, `randomVertexScalar`, `randomFaceScalar`, `randomOmega`,
    `randomDecomposed1Form`, `eigenmodeField`, `heatDiffusion`
    (← `generateHeatDiffusion`), `dampedWave` (← `generateDampedWave`)

  `field::interp::`
  - `whitney` (← `whitneyInterpolate`)

  `field::op::`
  - `gradient` (← `scalarGradient`)

  `field::extract::`
  - `isoline` (← `isolines`; possibly singular form)
  - `streamline` (← `traceStreamlines`)

  `nxr::core::` (cross-cutting infrastructure)
  - `Error`, `ErrorCode`
  - `CancellationToken`
  - `ProgressObserver`

- **Note on `MassMatrixVariant`:** the enum currently selects which
  mass matrix `assembleManifoldOperators` builds. Under D13, the
  enum becomes redundant — each variant has its own accessor under
  `manifold::ops::mass::{voronoi, barycentric, consistentFEM, galerkin}`.
  The enum + factory pair is deprecated.

- **Note on `ManifoldOperators` / `DECOperators` structs:** also
  deprecated under D7. Consumers call individual accessors instead
  of receiving a bundled struct.

---

---

## Open questions

Things that have come up but not been answered. Move to "Decisions
log" once settled.

- **Naming hierarchy under `manifold::ops::`.** Depth and shape are
  not yet fixed. Candidates include flat (`hodge0`, `hodge1`,
  `cotanLaplacian`, `vertexLumpedMassMatrix`) vs. semantic
  (`hodge.vertex` / `hodge.edge` / `hodge.face`,
  `laplacian.{vertex,face,edge,connection}`,
  `mass.{vertex,face,edge}.{lumped,galerkin,barycentric,...}`).
  Trade-off: deeper hierarchy is more discoverable but the C++
  syntax gets verbose; numeric subnames (`hodge::0`) aren't legal
  C++ identifiers and need an escape hatch.
- **Where does cross-cutting infrastructure live** (`Error`,
  `CancellationToken`, `ProgressObserver`, §11 storage helpers)?
  Candidates: `nxr::core::*`, `nxr::*` (top-level, no sub-namespace).
- **Where does the eigendecomposition sit?** It depends on operators,
  not directly on `geometry`. Candidates: `manifold::eigen::*` (per
  the earlier proposal), `manifold::ops::eigen::*`, sibling under
  `manifold::*`.
- **Where do other solvers go** — Poisson PDE, heat geodesic
  distance, signed heat, vector heat transport, log map, Karcher
  mean, Hodge decomposition, direction fields (trivial, smooth),
  flip-out path tracing, BFF UV? The earlier proposal had separate
  sub-namespaces (`solve::`, `distance::`, `transport::`,
  `connection::`, `decompose::`, `geometry::parametrization::`);
  whether to keep that depth or flatten is open.
- **Where does `CholeskyCache` (factor cache, distinct from matrix
  cache) live in the new tree?** Internal to solvers (hidden), or
  exposed as `manifold::ops::factor::*`?
- **Tier-2 cache implementation.** Where do the per-context cache
  slots physically live in code — a new `OpsCache` type stored on
  `ComputeContext`? A `mutable` member of `ComputeContext` itself?
  A separate `OpsAccessor` object?
- **Should there be a `Session` / `Handle` type in the core** that
  owns the stateful holder (currently lives in each binding's
  `ContextHolder` / `ContextWrapper`)? D6 implicitly answers "yes"
  by bundling caches on the context, but the explicit name and
  shape of that type aren't decided.
- **Naming: `topology` vs `mesh` vs `connectivity`** for the first
  sub-namespace; `geometry` vs `embedding` for the second. (Current
  working choice: `topology` and `geometry`, mirroring GC's own
  terminology.)
- **How do `manifold::topology` and `manifold::geometry` relate at
  the type level** — separate types that reference each other, or a
  combined `ComputeContext` that contains both? D6 says we keep
  them bundled in v1; v2 may revisit if a use case needs the
  one-topology-many-geometries shape GC supports.
- **Edge case from D5: how does field-from-measurement promotion work?**
  A consumer wanting to treat a manifold measurement (e.g.
  `manifold::measure::curvature::gaussian(ctx)`) as a field for
  downstream signal processing — colormap, gradient, spectral
  projection — needs an explicit promotion. Proposed sketch was
  `field::fromManifoldMeasurement(ctx, ...)`; shape and name not
  settled.
- **`ComputeContext` rename.** D9 keeps the name as-is but flags
  candidates: `Manifold`, `Domain`. The latter matches the signal-
  processing framing in D8 (the manifold is the *domain* on which
  signals live). Not blocking for the rest of the design but worth
  resolving before any C++ source-level rename lands.

---

## What's not in this document yet

By design. Each will be added when discussed:

- The full set of sub-namespaces under `manifold::` (ops, eigen,
  solve, distance, transport, connection, decompose, …).
- The full set of sub-namespaces under `field::`.
- Naming for individual functions and types.
- Header / source layout.
- Binding strategy (how the C++ tree projects into Node / WASM / MEX / CLI).
- Migration plan from the current flat `nxr::compute::*`.
- Backward-compatibility shims.

These are intentionally deferred until the foundation is settled.
