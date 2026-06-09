# Manifold geometry facets — GC-faithful facet structure + gauge workflow + operators access

**Date:** 2026-06-09
**Status:** Design — approved in discussion, pending spec review
**Scope:** C++ internal API, plus one thin MEX addition — an `operators` string-dispatch
command (operators already cross to MATLAB). The geometry consumer bundles
(`topology/geometry/gauge` struct shapes) are unchanged this pass.
**Builds on:** the intrinsic-Delaunay Phases 1–2 (`operatorGeometry()` switchable intrinsic
interface) — this design *names and generalizes* that mechanism.

---

## 1. Goal

Restructure `nxr::manifold::Manifold`'s public surface to mirror geometry-central's
conceptual levels — topology + the three nested geometry interfaces — but **rooted at
the embedding**, because every consumer (Brainstorm/MATLAB, three.js/WASM, CLI) starts
from `(vertices, faces)` and never from a bare metric. The reorganization gives a
**clean, type-enforced separation between intrinsic and extrinsic quantities**, which:

- makes operator assembly source its geometry **mechanically** (the facet owns the
  intrinsic-vs-embedded decision) instead of by per-function convention, and
- makes intrinsic-Delaunay "one mesh everywhere" (Phase 3) a **pointer move** rather
  than a per-operator rewrite.

This is an **additive** C++ refactor: the existing `mesh()`, `geometry()`,
`operatorGeometry()` accessors and the MEX/WASM bindings keep working unchanged on top
of the new facet layer.

## 2. The six structures (embedding-rooted composition)

geometry-central's class nesting is *capability subtyping* (`Embedded : Extrinsic :
Intrinsic : Base`, richest IS-A poorest). nxr-compute's facets are a *composition /
ownership* model over the same underlying GC objects: the embedding is the substrate
you always have; the others are facets read off it. The two views coexist — the GC
objects keep their inheritance internally; the facets are typed views over them.

Three categories: **geometry** (topology/embedded/intrinsic/extrinsic — per-element
*data* + metric), **gauge** (frame *transforms*), **operators** (assembled *matrices*).
Keeping operators in their own structure (not scattered through the geometry facets) is
deliberate — operators are a distinct category with a distinct lifecycle (lazy build,
independent cache, optional release) and a sanctioned MATLAB export path (§5b).

| Structure | Category | Backing | Under `intrinsicDelaunay=true` |
|---|---|---|---|
| `topology` | geometry | `ManifoldSurfaceMesh` | fixed |
| `embedded` | geometry | `VertexPositionGeometry` (Embedded interface) | **fixed** — always the input embedding |
| `intrinsic` | geometry | `VertexPositionGeometry` by default; `SignpostIntrinsicTriangulation` when normalized | **swappable — the Delaunay seam** |
| `extrinsic` | geometry | `VertexPositionGeometry` (Extrinsic interface) | fixed |
| `gauge` | transforms | derived: embedded frames + intrinsic connection | grid follows active gauge; LC grid fixed (φ_v=0) |
| `operators` | matrices | derived: sources from the geometry facets + active gauge | cotan/mass/connection certified-PSD; DEC/trivial original (Phase 3) |

**Key asymmetry (a feature, not a gap):** `SignpostIntrinsicTriangulation` implements
*only* `IntrinsicGeometryInterface`. So `intrinsic` is the one switchable facet;
`embedded`/`extrinsic` are *always* the original `VertexPositionGeometry`. The API
encodes that "the embedding of the Delaunay mesh" is not a thing (geodesic edges have
no straight-line embedding). `intrinsic` is *initialized from* the embedding but is an
independent object — defaults to the embedding's induced metric, swaps to the signpost
triangulation when normalized. Non-normalized ⇒ byte-identical (the facet is a view on
the same embedded object).

## 3. Facet contents (element-first interior)

Canonical shape is `manifold.<facet>.<element>.<quantity>` — element-first inside each
facet, matching the existing house style (`MeshGeometry` and the MEX bundles already
group `vertex.* / edge.* / face.* / halfedge.* / corner.*`).

Geometry facets hold *data* (per-element quantities + the metric); the assembled
*matrices* (Laplacians, DEC, mass, Hodge, connection, covariant) live in the `operators`
structure (§5b), which sources from these facets. The geometry facet a quantity lives in
records *where its operator sources from* — that is the operator-sourcing rule.

### `topology`
- connectivity / element counts / indices (`vertexIndices`, `edgeIndices`, …)
- *(sources `operators.dec.{d0,d1}` and `operators.laplacian.graph` — combinatorial,
  metric-free)*

### `embedded` (substrate — always the input)
- `vertex.position`, `vertex.normal`, `vertex.grid` (the complex frame `c = e1+i·e2`)
- `face.normal`, `face.grid`, `face.centroid`
- **`grid` is the canonical frame name.** `c = e1+i·e2` *is* the orthonormal frame in
  compact complex form; the explicit `{e1, e2, n}` basis is a *view* of it
  (`n = Re(c)×Im(c)`, exact), not a sibling field. No separate `frame` vs `grid` split.

### `intrinsic` (metric — Delaunay-swappable)
- `vertex.dualArea`, `vertex.angleSum`
- `edge.length`, `edge.cotanWeight`; `halfedge.cotanWeight`
- `halfedge.vectorInVertex`, `halfedge.vectorInFace` (intrinsic tangent reps)
- `halfedge.transportAlong`, `halfedge.transportAcross` (the connection)
- *(sources the intrinsic operators: `operators.laplacian.cotan`, `operators.mass.*`,
  `operators.hodge.*`, and the `K` of `operators.laplacian.connection`. Delaunay-swappable,
  so those operators are certified-PSD under normalization. **Phase 3:** the DEC Hodge
  stars source here too, restoring `cotanL == d0ᵀ★₁d0`.)*

### `extrinsic` (extrinsic curvature — fixed)
- `edge.dihedralAngle`
- `vertex.principalDir`, `vertex.curvature2RoSy` (the deviatoric `q` + mean `H`),
  expressed in the `embedded` frame.

### `gauge` (transforms — derived layer)
- the realized `grid` per gauge choice (euclidean / Levi-Civita / trivial)
- the surface↔world map (`G_intrinsic = G·cᵀ`) and gauge↔gauge rotation (`exp(iφ_v)`)
- *(sources the gauge operators `operators.laplacian.{connection,covariant}` — the active
  gauge selects which connection `K` they're built in)*
- **Dependency note:** `gauge` consumes `embedded` (frames) + `intrinsic` (connection/
  transport). It is a *sibling for access* but a *higher layer for derivation* — not a
  peer of the geometry facets in the dependency graph.

### Root convenience aliases (raw input only)
`manifold.vertexPositions`, `manifold.faces` — direct accessors for the literal,
facet-agnostic input. Everything *derived* stays under its facet; only the raw input is
hoisted, so the intrinsic/extrinsic separation is never punctured.

## 4. C++ API shape

`Manifold` gains five facet accessors returning lightweight typed view objects. The
existing accessors remain as the backing and are unchanged.

```cpp
class Manifold {
public:
    // --- existing (unchanged; the facets are layered over these) ---
    ManifoldSurfaceMesh&          mesh();
    VertexPositionGeometry&       geometry();            // embedded substrate
    IntrinsicGeometryInterface&   operatorGeometry();    // intrinsic, Delaunay-swappable
    bool                          isIntrinsicDelaunay() const;

    // --- new facet accessors (additive) ---
    TopologyFacet   topology();
    EmbeddedFacet   embedded();
    IntrinsicFacet  intrinsic();    // wraps operatorGeometry() — swappable
    ExtrinsicFacet  extrinsic();
    GaugeFacet      gauge();        // the active gauge (see §5)
    OperatorsFacet  operators();    // assembled matrices, lazy + cached (see §5b)

    // --- raw-input aliases ---
    const Eigen::MatrixXd& vertexPositions() const;
    const Eigen::MatrixXi& faces() const;
};
```

Facet objects are thin (hold a `Manifold&` / the relevant GC interface ref + lazy
caches); they expose element sub-accessors (`embedded().vertex().grid()`,
`intrinsic().edge().cotanWeight()`). They are **views, not owners** — lifetime is tied
to the `Manifold`, consistent with the existing view-field lifetime contract.

The facet layer is where the **operator sourcing rule** lives: assembly functions take a
facet (or the manifold + a facet selector) and read geometry from it, so the
intrinsic-vs-embedded choice is structural. `intrinsic()` is the Delaunay-swappable
source for cotan/mass/connection/DEC; `embedded()` is the fixed source for
frames/normals/positions; `extrinsic()` for dihedral/curvature.

## 5. Gauge construction workflow

A gauge is identified by `(type, singularities?)`. Three usage patterns collapse to **one
mechanism with two binding times** — there is a single trivial-connection solve and a
single cache key `(type, singularities)`.

| Pattern | Call | Default/active gauge |
|---|---|---|
| 1 | `Manifold(V, F)` | Levi-Civita (no singularity input needed) |
| 2 | `Manifold(V, F, singularities)` | **trivial** — consistent grid becomes the manifold's canonical frame field |
| 3 | `Manifold(V, F)` → `gauge('trivial', singularities)` | LC default; trivial computed + cached on request |

- **Levi-Civita** (and its euclidean variant) is the zero-config reference gauge; grid
  frames come straight from the per-vertex tangent bases.
- **Trivial** is a derived gauge: `integrateTrivialGaugeRotations` gives the per-vertex
  `φ_v` that realize the globally-consistent (flat-except-at-singularities) frame field;
  `assembleTrivialConnectionLaplacian` is the operator in that gauge. Machinery already
  exists in the library.
- Patterns 2 and 3 produce the **identical** trivial gauge for the same singularities —
  pattern 2 is eager (pre-warms the cache, promotes to default); pattern 3 is lazy
  (returns a cached gauge value).

### Active-gauge state
- **Default is fixed at construction:** LC for patterns 1/3-baseline, trivial for
  pattern 2.
- **Optional explicit `setGauge(type, singularities?)`** re-points the active gauge later
  (cheap — selects/realizes a cached facet). Ad-hoc `gauge(type, sing)` *requests* return
  gauge *values* without mutating the default unless `setGauge` is used.

```cpp
Manifold(const double* V, int nV, const int32_t* F, int nF,
         bool intrinsicDelaunay = false);                       // pattern 1
Manifold(const double* V, int nV, const int32_t* F, int nF,
         const std::map<int,double>& singularities,
         bool intrinsicDelaunay = false);                       // pattern 2

GaugeFacet gauge();                                             // active gauge
GaugeFacet gauge(GaugeType type,
                 const std::map<int,double>& singularities = {}); // request a value
void       setGauge(GaugeType type,
                    const std::map<int,double>& singularities = {}); // re-point default
```

### Gauss–Bonnet validation (hard guard)
Trivial connections exist **iff the singularity indices sum to the Euler characteristic
χ** (`Σ index == χ`). This is a solvability condition, not a preference — the
trivial-connection Poisson solve has no solution otherwise. The constructor / `gauge` /
`setGauge` **validate `Σ index == χ`** and throw `Error(ErrorCode::InvalidInput, …)` with
a hint naming χ and the supplied sum when violated. "Two vertices" is the χ=2 case (a
closed genus-0 surface, e.g. an inflated/closed hemisphere, two +1 singularities); the
API takes a `{vertex → index}` map and validates the sum generally (a disk is χ=1; handles
differ).

## 5b. Operators access (`manifold/operators`)

The assembled matrices live in their own `operators` structure — a single source of
truth, lazily built and independently cached, that replaces today's scattered
`assembleX` free functions + per-binding `ensure*` slots.

### Three consumers
1. **nxr-compute's own solvers** (`solveEigenmodes`, `solvePoisson`, `hodgeDecompose`,
   direction fields) read operators by name.
2. **geometry-central** (assembly inputs).
3. **MATLAB directly** — a user pulls a named operator as native sparse and runs their
   own `eig`/`eigs`/`\`. *Single source of truth ⇒ the matrix exported to MATLAB is
   byte-identical to the one nxr's internal solver uses, so internal and user-side
   results are cross-checkable.*

### Taxonomy (maps ~1:1 to GC cached-member families)
| Key | Returns | Type | Sources from |
|---|---|---|---|
| `laplacian.cotan` | cotan Laplacian | real sparse | `intrinsic` |
| `laplacian.connection` | vertex connection L (in active gauge) | **complex** sparse | `intrinsic` + active `gauge` |
| `laplacian.covariant` | 3-frame covariant L (`3N×3N`) | real sparse | `intrinsic` + active `gauge` |
| `laplacian.graph` | graph Laplacian `D−A` | real sparse | `topology` |
| `dec` | `{d0, d1}` bundle | real sparse | `topology` |
| `mass.{lumped,galerkin}` | mass matrix | real sparse | `intrinsic` |
| `hodge.{h0,h1,h2,h1inv}` | Hodge stars | real sparse | `intrinsic` |

Coarse key returns a **bundle**, fine key returns **one matrix**: `operators('dec')` →
`{d0,d1}`; `operators('mass','lumped')` → one matrix; `operators('laplacian','connection')`
→ one matrix.

### Dual surface (typed C++ core + string MEX veneer)
- **C++ core — typed accessors** (type safety: connection is complex, cotan is real, DEC
  is a struct — a single string-dispatched C++ return would force `std::variant`/erasure):
  ```cpp
  OperatorsFacet operators();   // on Manifold
  // operators().laplacian().cotan()        -> const Eigen::SparseMatrix<double>&
  // operators().laplacian().connection()   -> const Eigen::SparseMatrix<std::complex<double>>&
  // operators().dec()                      -> const DECOperators&
  // operators().mass().lumped()            -> const Eigen::SparseMatrix<double>&
  ```
  Each accessor lazily builds + caches its operator independently (GC `require` model).
- **MEX veneer — string dispatch** (in scope *now*, because operators already cross to
  MATLAB via the existing `.operators` opt-in): `nxr_compute('operators', h, 'laplacian',
  'connection')` → native MATLAB sparse (real via `eigenSparseToMx`, complex via
  `eigenComplexSparseToMx`). This is the clean single entry point replacing
  "dig through `geometry.operators.laplacian`". The string keys map onto the typed core.
  *(The geometry/gauge consumer bundles themselves stay C++-only this pass — only this
  thin operators command is added on the MATLAB side.)*

### Lazy build + release (GC `require`/`unrequire` memory model)
- **Independent per-operator caching decouples today's `MeshOperators` bundle** — the key
  memory win. Requesting `laplacian.cotan` no longer also builds mass + normals
  (today `assembleManifoldOperators` fuses `{cotanLaplacian, mass, vertexDualAreas,
  vertexNormals, totalArea}`).
- **Optional explicit `operators().release(key)`** drops a heavy cached operator
  mid-session (mirrors GC `unrequire`); lazy-build-then-hold-for-handle-lifetime is the
  default. MATLAB-exported copies are independent of the C++ cache (MATLAB owns its copy).

### Matched-pair convention for eigensolves (correctness)
"Eigensolve of a Laplacian" is almost always the **generalized** problem `K φ = λ M φ`,
not `eig(K)`. The facet documents and eases the pairing: **cotan Laplacian ↔ vertex mass**
(lumped/Galerkin); **connection Laplacian ↔ vertex mass**. Pulling `K` without `M` is the
easiest way to get subtly wrong eigenvalues. **PSD caveat:** raw signed cotan weights are
indefinite on non-Delaunay meshes, so `'smallestabs'` can misbehave — for a well-posed
smallest-eigenvalue solve in MATLAB, create the handle with `intrinsicDelaunay=true`
(certified-PSD `K` + matched `M`, Phases 1–2).

## 6. Backward compatibility

- **Additive only.** `mesh()`, `geometry()`, `operatorGeometry()`, `isIntrinsicDelaunay()`
  stay and keep their semantics; the facets are views layered over them.
- **MEX/WASM unchanged.** The `topology/geometry/gauge/bundle` consumer bundles continue
  to use the existing accessors. Re-cutting the consumer surface into five facets is
  explicitly out of scope for this pass (a separate, consumer-facing decision).
- **Existing assembly call sites** continue to compile; they are migrated to source from
  facets incrementally (the migration is what delivers the mechanical-sourcing payoff,
  but it preserves numerical output — non-normalized is byte-identical).

## 7. Intrinsic-Delaunay interaction

- `intrinsic()` is the named, switchable facet — exactly today's `operatorGeometry()`.
  cotan/mass/vertex-connection already route through it (Phases 1–2) ⇒ certified-PSD
  under normalization.
- **DEC + trivial gauge** still ride the original-mesh DEC under normalization (Phase-2
  deferral). So pattern 2/3 + `intrinsicDelaunay` gives the trivial gauge on the original
  connectivity while cotan/LC-connection are intrinsic. The facet model makes Phase 3
  (route DEC through `intrinsic()`, restore `cotanL == d0ᵀ★₁d0`) a pointer move — but
  Phase 3 itself is out of scope here.

## 8. Error handling

- Gauss–Bonnet violation → `Error(ErrorCode::InvalidInput)` with χ + supplied sum in the
  hint.
- Requesting `gauge('trivial')` with no singularities → `Error(ErrorCode::InvalidInput)`
  ("trivial gauge requires a singularity map; Levi-Civita needs none").
- Facet quantity that requires an embedding on a context that lacks one — N/A here
  (`Manifold` is always embedded), but `intrinsic()` must never be asked for an extrinsic
  quantity (compile-time: extrinsic quantities are simply not on the `IntrinsicFacet`).

## 9. Test plan

- **Facet identity / non-normalized byte-identity:** for a fixture, every facet quantity
  read via the new accessor equals the value read via the existing accessor
  (`embedded().vertex().grid()` == `vertexGrid(m)`, `intrinsic().edge().cotanWeight()`
  == `operatorGeometry().edgeCotanWeights`, etc.). Non-normalized: `intrinsic` facet ==
  embedded-induced (byte-identical).
- **Operator sourcing:** assembling cotan/connection via the `intrinsic()` facet under
  `intrinsicDelaunay=true` reproduces the certified-PSD result from Phase 1/2 (re-use the
  rhombus fixture + PSD certificate).
- **Gauge workflow:**
  - Pattern 1: `Manifold(V,F)` → `gauge()` is Levi-Civita; grid == LC frames.
  - Pattern 2: `Manifold(V,F,sing)` → `gauge()` is trivial; grid == trivial-realized
    (rotated by `φ_v`); equals the pattern-3 result for the same singularities.
  - Pattern 3: `Manifold(V,F)` → `gauge('trivial',sing)` value == pattern-2 grid; default
    unchanged until `setGauge`.
  - `setGauge('trivial',sing)` re-points the default; `gauge()` then returns trivial.
- **Gauss–Bonnet guard:** singularities summing to χ succeed; a wrong sum throws
  `InvalidInput` with χ in the message. Trivial with empty singularities throws.
- **Operators access:** `operators().laplacian().cotan()` == today's
  `assembleManifoldOperators(m).cotanLaplacian`; `operators().dec()` == `assembleDECOperators`;
  the connection/covariant == the Phase-2 results. Independent caching: requesting
  `laplacian.cotan` does **not** populate the mass cache (assert via a build counter or
  by observing mass is absent). `release(key)` drops the cached entry (next request
  rebuilds). MEX: `nxr_compute('operators', h, 'laplacian', 'cotan')` returns a native
  sparse equal to the `geometry.operators.laplacian` from the existing opt-in; complex
  for `'connection'`. Matched-pair eigensolve: `eigs(K, M)` in MATLAB on the exported
  `(cotan, mass.lumped)` reproduces `solveEigenmodes` eigenvalues (cross-check).
- **Backward compat:** existing native tests (`test_connection_laplacian`,
  `test_geometry_bundle`, `test_intrinsic_delaunay`) and MEX tests (`test_operators`,
  `test_bundle`, `test_intrinsic_delaunay.m`) pass unchanged.

## 10. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | hierarchy direction | embedding-rooted composition (input is always embedded), not GC's capability nesting |
| 2 | keep `embedded` as a facet | yes — GC 1:1 fidelity + exceptionless sourcing rule; dissolving it invites flattening swappable per-element fields to root |
| 3 | `extrinsic` facet | own facet (GC fidelity) |
| 4 | interior grouping | element-first (`<facet>.<element>.<quantity>`), matching house style |
| 5 | grid vs frame | `grid` canonical (compact complex frame); explicit `{e1,e2,n}` is a view |
| 6 | root aliases | only raw input (`vertexPositions`, `faces`) |
| 7 | scope | C++ only; MEX bundles + Phase-3 DEC out of scope |
| 8 | gauge patterns | one mechanism, two binding times (eager at ctor / lazy at request) |
| 9 | active gauge | default fixed at construction + optional explicit `setGauge` |
| 10 | trivial solvability | validate `Σ index == χ` (Gauss–Bonnet) — hard `InvalidInput` guard |
| 11 | operators home | own `manifold/operators` structure (matrices), separate from geometry facets (data) + gauge (transforms) |
| 12 | operators surface | typed C++ core (type-safe) + string-dispatch MEX veneer (in scope now — operators already cross to MATLAB) |
| 13 | operators lifecycle | independent per-operator lazy cache (decouples the `MeshOperators` bundle) + optional `release()` (GC `require`/`unrequire` model) |
| 14 | operators consumers | nxr solvers + GC + **direct MATLAB export** (single source of truth ⇒ byte-identical, cross-checkable); document matched `(K, M)` pairing + PSD caveat |

## 11. Out of scope / deferred

- Re-cutting the MEX/WASM *geometry* consumer bundles (`topology/geometry/gauge` struct
  shapes) into the facet structure (consumer-facing — separate pass). **Exception:** the
  thin `operators` string-dispatch MEX command **is** in scope, because operators already
  cross to MATLAB and the workflow needs direct named export.
- Phase 3: routing DEC + trivial connection through `intrinsic()` (fully-intrinsic
  bundle, restores the cross-DEC identity under normalization).
- Auto-placement / suggestion of singularities (user supplies; we validate).
- A MATLAB `handle`-class wrapper exposing the facets (application-side, like the JS
  wrappers).
