# Manifold geometry facets — GC-faithful five-facet structure + gauge construction workflow

**Date:** 2026-06-09
**Status:** Design — approved in discussion, pending spec review
**Scope:** C++ internal API only. No MEX/WASM consumer-bundle change in this pass.
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

## 2. The five facets (embedding-rooted composition)

geometry-central's class nesting is *capability subtyping* (`Embedded : Extrinsic :
Intrinsic : Base`, richest IS-A poorest). nxr-compute's facets are a *composition /
ownership* model over the same underlying GC objects: the embedding is the substrate
you always have; intrinsic / extrinsic / gauge are facets read off it. The two views
coexist — the GC objects keep their inheritance internally; the facets are typed views
over them.

| Facet | Backing GC object(s) | Under `intrinsicDelaunay=true` |
|---|---|---|
| `topology` | `ManifoldSurfaceMesh` | fixed |
| `embedded` | `VertexPositionGeometry` (Embedded interface) | **fixed** — always the input embedding |
| `intrinsic` | `VertexPositionGeometry` by default; `SignpostIntrinsicTriangulation` when normalized | **swappable — the Delaunay seam** |
| `extrinsic` | `VertexPositionGeometry` (Extrinsic interface) | fixed |
| `gauge` | derived: embedded frames + intrinsic connection | connection follows `intrinsic`; grid fixed (φ_v=0) |

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

### `topology`
- connectivity / element counts / indices (`vertexIndices`, `edgeIndices`, …)
- combinatorial differential operators `dec.d0` `[E×V]`, `dec.d1` `[F×E]` (incidence —
  metric-free)
- graph Laplacian `D−A = d0ᵀd0`

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
- operators: cotan Laplacian, mass (lumped/Galerkin), vertex connection Laplacian's `K`,
  Hodge stars `hodge0/1/2/h1inv`
- **(Phase 3)** the intrinsic DEC Hodge stars route here too — restoring
  `cotanL == d0ᵀ★₁d0` under normalization by construction.

### `extrinsic` (extrinsic curvature — fixed)
- `edge.dihedralAngle`
- `vertex.principalDir`, `vertex.curvature2RoSy` (the deviatoric `q` + mean `H`),
  expressed in the `embedded` frame.

### `gauge` (transforms — derived layer)
- the realized `grid` per gauge choice (euclidean / Levi-Civita / trivial)
- the surface↔world map (`G_intrinsic = G·cᵀ`) and gauge↔gauge rotation (`exp(iφ_v)`)
- operators: connection Laplacian (complex) and `covariantLaplacian` (`3N×3N` real)
- **Dependency note:** `gauge` consumes `embedded` (frames) + `intrinsic` (connection/
  transport). It is a *sibling for access* but a *higher layer for derivation* — not a
  peer of the other three in the dependency graph.

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

## 11. Out of scope / deferred

- Re-cutting MEX/WASM consumer bundles into five facets (consumer-facing — separate pass).
- Phase 3: routing DEC + trivial connection through `intrinsic()` (fully-intrinsic
  bundle, restores the cross-DEC identity under normalization).
- Auto-placement / suggestion of singularities (user supplies; we validate).
- A MATLAB `handle`-class wrapper exposing the facets (application-side, like the JS
  wrappers).
