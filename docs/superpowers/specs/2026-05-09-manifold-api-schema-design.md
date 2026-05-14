# Manifold API schema — design

**Status:** approved 2026-05-09
**Author:** Diellor Basha (with Claude as scribe)
**Repository:** `nxr-compute`

> **Note (2026-05-13):** the `implemented_by` examples below show
> `ctx.assembleMeshOperators().stiffness` because they predate the
> field-rename pass (commit `2dbb95b`). In the current bindings the
> path is `ctx.assembleMeshOperators().cotanLaplacian`; similarly
> `.vertexAreas` → `.vertexDualAreas`, `.normals` → `.vertexNormals`.
> The schema slot names themselves (`stiffness`, `mass`, etc.) are
> mathematical concept names and remain unchanged in
> `manifold.schema.json`. This spec is preserved as design rationale;
> the live schema and `docs/schema/CONVENTIONS.md` have the current
> examples.

## Files produced

| File | Role |
|---|---|
| `docs/schema/common.meta.schema.json` | Shared $defs (dtype, storage, leaf_slot, operation_slot, binding_map, group_attributes) reused by every NXR namespace meta-schema |
| `docs/schema/manifold.schema.json` | The manifold contract |
| `docs/schema/manifold.meta.schema.json` | JSON Schema (Draft 2020-12) that validates `manifold.schema.json` (imports `common.meta.schema.json`) |
| `docs/schema/field.schema.json` | The field contract — scalar/vector fields defined on a manifold |
| `docs/schema/field.meta.schema.json` | JSON Schema validating `field.schema.json` |
| `docs/schema/CONVENTIONS.md` | Short prose reference for authors and reviewers |
| `docs/superpowers/specs/2026-05-09-manifold-api-schema-design.md` | This document — design rationale |

---

## 1. Goal

Define a single design-time contract that captures:

1. **What a manifold is** — the hierarchy of slots (data + operations)
   a surface-mesh manifold can hold.
2. **What every slot looks like** — symbolic shape, dtype, storage layout,
   dependency graph, producing operation, memory cost, complexity.
3. **How DEC-derived slots compose** — in symbolic formula form, plus a
   binding reference per shell (cpp, wasm, mex, node_addon) to the actual
   implementation.

The schema is the contract by which other NXR components (nxr-viewer in
the first instance, future nxr-graph / nxr-recording schemas, downstream
analysis apps) are designed. It is the canonical artifact that every
component can point at and say "this is what a manifold gives you."

## 2. Non-goals

- **No runtime role.** Bindings do not parse the schema on init.
  Validation, dispatch, and marshalling stay hand-written in
  `bindings/{node,wasm,mex}/`.
- **No code generation.** Nothing in `bindings/` or `src/` is generated
  from the schema. Drift between bindings and schema is caught by
  human review, not by tooling, in this version.
- **No instance data.** The schema describes the shape of a manifold,
  never an actual manifold's bytes. Buffer sizes are always symbolic
  (`8 * V * 3`, never `120000`).
- **No file I/O concerns.** Zarr, HDF5, FreeSurfer parsing live in
  `cxf-io`. The schema does not enumerate persistence formats.

## 3. Background

`nxr-compute` exposes a rich C++ surface (`include/nxr/compute.h`)
covering mesh operators, DEC, eigensolvers, Hodge decomposition,
geodesics, vector heat method, signed heat method, smooth direction
fields, stripe patterns, parametrization, curvature, isolines and
streamlines. Three binding shells (`bindings/{node,wasm,mex}/`)
re-export the surface in their host's idiom.

Today there is no single document a downstream component author can
read to understand the full shape of "what a manifold gives you."
`include/nxr/compute.h` describes function signatures, not the
hierarchy; `docs/extensions.md` proposes a namespace reorganization
but is C++-centric and partially superseded; `bindings/wasm/js/index.d.ts`
describes the JS surface only.

The manifold API schema fills that gap.

## 4. Layout

```
docs/schema/
├── common.meta.schema.json      ← shared $defs (dtype, storage, leaf_slot, …)
├── manifold.schema.json         ← the manifold contract
├── manifold.meta.schema.json    ← validates manifold.schema.json
├── field.schema.json            ← the field contract (scalar/vector fields on a manifold)
├── field.meta.schema.json       ← validates field.schema.json
└── CONVENTIONS.md               ← prose reference
```

Each file is hand-authored. CI may add a JSON Schema validator step
(`ajv-cli` or equivalent) so the meta-schema catches drift, but that
is a follow-up.

`common.meta.schema.json` is the cross-namespace language. Every
namespace meta-schema (`manifold.meta`, `field.meta`, future `graph.meta`,
`recording.meta`, `volume.meta`) imports its primitive definitions —
`dtype`, `storage`, `slot_path`, `binding_map`, `leaf_slot`,
`operation_slot`, `group_attributes` — via `$ref` to keep the schemas
structurally consistent without duplicating the definitions.

## 5. The slot tree

The manifold is a hierarchical namespace. Group nodes are organisational;
data lives only at the leaves. Every node — group or leaf — carries an
`attributes` block.

```
manifold/
├── attributes               ← header for the whole object
│
├── core/                    ← inputs; everything else is derivable
│   ├── attributes
│   ├── vertices             ← [V, 3] float64
│   └── faces                ← [F, 3] int32
│
├── embedding/               ← three.js Object3D-style world placement
│   ├── attributes
│   ├── bounding_box         ← { min[3], max[3] } in local coords
│   └── transform            ← [4, 4] float64 affine
│
├── topology/
│   ├── attributes           ← euler_characteristic, n_boundary_loops, n_edges, orientable, is_closed
│   └── halfedge/{attributes, twin, next, vertex, face, edge, is_boundary}
│
├── geometry/
│   ├── attributes
│   ├── face/{attributes, normals, areas, centroids, frames}
│   ├── vertex/{attributes, normals, vertex_areas}
│   ├── edge/{attributes, lengths, cotangent_weights, dihedral_angles}
│   ├── curvature/{attributes, gaussian, mean, principal_max, principal_min, principal_dir_max}
│   └── singularities        ← 2 reference vertices anchoring the global tangent frame
│
├── operators/               ← d0, d1, star* canonical; rest derived
│   ├── attributes
│   ├── d0, d1
│   ├── hodge0, hodge1, hodge2, hodge1_inverse
│   ├── stiffness            ← derived: d0^T * hodge1 * d0
│   ├── mass                 ← alias: hodge0
│   ├── gradient             ← derived: d0
│   └── divergence           ← derived: d0^T * hodge1
│
├── eigen/
│   ├── attributes           ← K, operator_ref, inner_product_ref, sigma, normalized, dc_removed
│   ├── values               ← [K] float64
│   └── vectors              ← [V, K] float64, vMajor (row-major)
│
├── parametrization/         ← coordinate atlases (global UV + local frame fields + stripes)
│   ├── attributes
│   ├── uv                   ← [V, 2] float64 — BFF global parametrization
│   ├── cross_field          ← [V] complex128 — 4-RoSy face frames
│   ├── line_field           ← [V] complex128 — 2-RoSy vertex frames
│   └── stripes              ← [2*N, 3] float64 — phase isolines
│
├── query/                   ← localization (returns geometric loci)
│   ├── attributes
│   ├── point                ← (v) → v
│   ├── line                 ← (va, vb) → [N, 3] edge-flip geodesic
│   └── area                 ← (v, level) → contour segments of heat-distance level set
│
└── measure/                 ← metric scalars (returns numbers)
    ├── attributes
    ├── distance             ← scalar length of query.line
    └── area                 ← scalar area enclosed by query.area level set
```

### 5.1 Why DEC operators are siblings of stiffness/gradient

The six DEC operators (`d0`, `d1`, `hodge0..2`, `hodge1_inverse`) are the
canonical building blocks. Every other linear operator in the schema
is a composition of these via formulas like `d0^T * hodge1 * d0`.
Putting them in a `dec/` subgroup buries them; they are the simplest
operators on the manifold, not a special case. Other components
(stiffness, gradient, divergence, Hodge Laplacian on k-forms, MFT, etc.)
sit alongside them and reference them via the `formula` attribute.

### 5.2 Why fields are not a manifold property

Heat diffusion, damped wave, Hodge decomposition outputs and analytic
field generators are *under-the-hood machinery* for computing
distances (heat method), globally consistent tangent frames (vector
heat method), and parametrizations (BFF + signed heat). They are not
properties of the manifold itself — a manifold without a heat-diffusion
buffer attached is no less a manifold. The C++ library still computes
them; the schema simply does not enumerate them as slots.

The exception is `parametrization/`: cross-fields, line-fields and
stripes are coordinate atlases, not data fields, and are exposed
because consumers (nxr-viewer) need to know they exist as
coordinate systems on the manifold.

### 5.3 Why query and measure are split

`query/` returns *loci* (a vertex, a polyline, a contour). `measure/`
returns *scalars* derived from those loci (the length of a polyline,
the area enclosed by a contour). The split mirrors the natural
categorical distinction between "where is X" and "how big is X" and
keeps each operation's `output` schema simple.

## 6. The attribute set

Every leaf slot (data buffer) carries this contract:

| Attribute | Required | Example | Notes |
|---|---|---|---|
| `description` | yes | `"Cotangent Laplacian, V×V, PSD symmetrized"` | one-line prose |
| `shape` | yes | `"[V, 3]"`, `"[V, V] sparse"`, `"[K]"` | symbolic |
| `dtype` | yes | `"float64"` | enum (§7.1) |
| `storage` | yes | `"row_major"` | enum (§7.2) |
| `memory_bytes` | yes | `"8 * V * 3"` | symbolic formula (§7.3) |
| `depends_on` | yes | `["core/vertices", "core/faces"]` | array of slot paths |
| `formula` | only if derived | `"d0^T * hodge1 * d0"` | symbolic, ASCII (§7.4) |
| `implemented_by` | yes if non-derived | `{ cpp, wasm, mex, node_addon }` | per-binding entry |
| `complexity` | yes | `"O(V^1.5)"` | big-O for the producing op |
| `errors` | optional | `["NonManifold"]` | subset of `nxr::compute::ErrorCode` |
| `cancellable` | optional, default `false` | `true` | accepts `CancellationToken` |
| `progress` | optional, default `false` | `true` | emits `ProgressObserver` updates |
| `units` | optional | `"meters"` | scientific provenance |
| `references` | optional | `["Crane et al. 2013"]` | algorithmic citations |

Group-level `attributes` carry only the meta facts that apply to the
group as a whole — counts that do not belong to any single leaf,
configuration choices that gate the whole group (e.g., `eigen/attributes`
holds `K`, `operator_ref`, `inner_product_ref`, `sigma`, `normalized`,
`dc_removed`).

The root `manifold/attributes` is the canonical header: `fingerprint`
(stable hash of vertices+faces), `nxr_compute_version`, `created_at`.

### 6.1 Operations under query/ and measure/

Operations are verbs, not buffers, so their attributes describe a
signature instead of a layout:

| Attribute | Example |
|---|---|
| `inputs` | `[{ "name": "v", "type": "vertex_index" }, { "name": "level", "type": "float64" }]` |
| `output` | `{ "shape": "[N, 3]", "dtype": "float64", "storage": "row_major" }` |
| `formula` | `"area(v, level) = ∫_{D(v) ≤ level} dA"` (optional, prose-math) |
| `implemented_by` | per-binding map (same shape as data slots) |
| `complexity`, `cancellable`, `progress`, `errors`, `references` | same |

## 7. Conventions

### 7.1 dtype enum

```
float64, float32, int32, int64, uint8, uint32, complex128, complex64, bool
```

`complex128` and `complex64` represent IEEE-style complex numbers
stored as interleaved real / imaginary pairs (§7.5). `bool` is
materialized as `uint8` in every binding; the dtype tag exists for
documentation only.

### 7.2 storage enum

```
row_major, col_major, sparse_coo, sparse_csc, sparse_csr,
diagonal, complex_interleaved, struct, scalar
```

- **`row_major`** / **`col_major`** — dense N-D buffers.
- **`sparse_coo`** — `{ row, col, data, rows, cols, nnz }` triplets;
  the WASM and addon bindings standardize on this.
- **`sparse_csc`** / **`sparse_csr`** — Eigen/MATLAB native; admitted
  for completeness, not currently used by any binding.
- **`diagonal`** — sparse matrix where only the main diagonal is
  populated; binding may flatten to a length-N dense vector.
- **`complex_interleaved`** — implies dtype is `complex64` or
  `complex128`; storage layout is `re_0, im_0, re_1, im_1, …`
  matching `std::complex<T>*` cast to `T*`, NumPy `np.complex128`,
  MATLAB native complex, and Eigen `Matrix<std::complex<T>, …>`.
- **`struct`** — leaf is a record of named sub-slots, each with its
  own attributes (e.g., `geometry/face/frames` is a struct of `e1`,
  `e2`).
- **`scalar`** — single value, no shape.

### 7.3 memory_bytes formula vocabulary

```
V    — n_vertices
E    — n_edges
F    — n_faces
H    — n_halfedges (= 2*E for closed manifolds; less for boundary meshes)
K    — n_eigenmodes (only meaningful inside eigen/)
N    — operation-specific (e.g., number of segments in a streamline,
                            polyline points in a geodesic path)
T    — number of timesteps (in time-varying contexts)
nnz(<slot_path>) — number of non-zeros in a sparse slot
```

Constants follow C-like arithmetic: `8 * V * 3`, `12 * nnz(operators/d0)`.
The "12" in the latter is `8 + 4 + 4` (8-byte data + 4-byte row + 4-byte
col) per non-zero in COO storage, plus a small overhead for the
`{rows, cols, nnz}` header that we ignore at this granularity.

The schema does not specify the exact `nnz` formula for each operator —
that depends on the mesh and is a runtime quantity. `nnz(operators/d0)`
is left symbolic for the same reason `V` is.

### 7.4 formula syntax

ASCII-only, mathematical-notation-light. Recognised tokens:

- `*` — matrix or scalar multiplication (left-associative)
- `+`, `-` — addition / subtraction
- `^T` — transpose (postfix)
- `^-1` — inverse (postfix)
- `<slot_name>` — reference to another slot at the same level or
  resolved by `depends_on`. Bare names within `operators/` resolve to
  `operators/<name>`; otherwise use a slash-separated path.
- Parentheses for grouping.

Examples:

```
d0^T * hodge1 * d0                                  ← stiffness
d0                                                 ← gradient
d0^T * hodge1                                       ← divergence
(d0^T * hodge1 * d0)^-1 * f                         ← Poisson solve  (notional)
```

Formulas are documentation-grade. They are not parsed or executed.

### 7.5 Complex types

Complex scalars and complex vectors appear naturally in the
Knöppel-Crane n-RoSy formulation, where the smoothest direction field
is the principal eigenvector of a complex connection-Laplacian, and in
the stripe-pattern phase function. The schema represents these as:

```json
{
  "shape": "[V]",
  "dtype": "complex128",
  "storage": "complex_interleaved",
  "memory_bytes": "16 * V"
}
```

Across bindings, this round-trips as:

| Binding | Type |
|---|---|
| C++ | `Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>` (length V) |
| N-API addon / WASM | `Float64Array` of length `2 * V`, interleaved re/im |
| MEX | MATLAB native complex column vector of length V |

The same `complex_interleaved` convention applies to higher-rank
complex tensors: a `[V, 3] complex128` is `24 * V` bytes laid out as
`re_{v=0,c=0}, im_{v=0,c=0}, re_{v=0,c=1}, im_{v=0,c=1}, …`.

### 7.6 `implemented_by` shape

```json
{
  "cpp":        "<symbol or expression in the nxr::compute namespace>",
  "node_addon": "<method on the JS handle, dot-notation if returning a struct field>",
  "wasm":       "<method on the JS handle, dot-notation if returning a struct field>",
  "mex":        "<command name, dot-notation for struct fields>"
}
```

Each value is a string. The string format is intentionally loose —
it exists to point a reader at the right binding entry, not to be
parsed. Examples:

```
cpp:        "nxr::compute::assembleMeshOperators().stiffness"
node_addon: "ctx.assembleMeshOperators().stiffness"
wasm:       "ctx.assembleMeshOperators().stiffness"
mex:        "nxr_compute('assembleMeshOperators', verts, faces).stiffness"
```

For derived slots (those with a `formula`), `implemented_by` may be
omitted — the formula is sufficient to tell a binding author how to
realize the slot. Where the C++ surface happens to expose the derived
quantity directly (for performance), `implemented_by` may be present
*and* the formula present — both are valid and the binding may choose.

## 8. Worked example — `operators/stiffness`

```json
"stiffness": {
  "attributes": {
    "description": "Cotangent Laplacian (PSD, symmetrized).",
    "shape": "[V, V] sparse",
    "dtype": "float64",
    "storage": "sparse_coo",
    "memory_bytes": "12 * nnz(operators/stiffness)",
    "depends_on": ["operators/d0", "operators/hodge1"],
    "formula": "d0^T * hodge1 * d0",
    "implemented_by": {
      "cpp":        "nxr::compute::assembleMeshOperators().stiffness",
      "node_addon": "ctx.assembleMeshOperators().stiffness",
      "wasm":       "ctx.assembleMeshOperators().stiffness",
      "mex":        "nxr_compute('assembleMeshOperators', V, F).stiffness"
    },
    "complexity": "O(F)",
    "errors": ["NonManifold", "GeometryDegenerate"],
    "units": "dimensionless",
    "references": ["Pinkall & Polthier 1993", "Meyer et al. 2003"]
  }
}
```

Reading this:

- A consumer designing an nxr-viewer panel knows it needs a sparse
  V×V float64 buffer, COO triplets, ~12 bytes per non-zero.
- A binding author updating `bindings/wasm/src/nxr_compute_wasm.cpp`
  knows the WASM method is `ctx.assembleMeshOperators().stiffness`
  and that the formula `d0^T * hodge1 * d0` is what the underlying C++
  computes — the binding does not need to invent a separate
  `assembleStiffness()` entry.
- A reviewer comparing the schema to actual library behaviour can
  cross-check: the binding returns COO; the C++ comment says
  "PSD, symmetrized"; both match the schema.

## 9. Worked example — `query/line` and `measure/distance`

```json
"line": {
  "attributes": {
    "description": "Edge-flip (Sharp & Crane 2020) geodesic between two vertices.",
    "inputs": [
      { "name": "va", "type": "vertex_index" },
      { "name": "vb", "type": "vertex_index" }
    ],
    "output": { "shape": "[N, 3]", "dtype": "float64", "storage": "row_major", "memory_bytes": "8 * N * 3" },
    "depends_on": ["core/vertices", "core/faces", "topology/halfedge"],
    "implemented_by": {
      "cpp":        "nxr::compute::tracePath",
      "node_addon": "ctx.tracePath",
      "wasm":       "ctx.tracePath",
      "mex":        "nxr_compute('tracePath', V, F, va, vb)"
    },
    "complexity": "O(diameter / mean_edge_length)",
    "errors": ["InvalidInput"],
    "references": ["Sharp & Crane SIGGRAPH Asia 2020"]
  }
}
```

```json
"distance": {
  "attributes": {
    "description": "Metric length of the query/line geodesic between two vertices.",
    "inputs": [
      { "name": "va", "type": "vertex_index" },
      { "name": "vb", "type": "vertex_index" }
    ],
    "output": { "shape": "[]", "dtype": "float64", "storage": "scalar", "memory_bytes": "8" },
    "depends_on": ["query/line"],
    "formula": "Σᵢ |line(va, vb)[i+1] − line(va, vb)[i]|",
    "complexity": "O(diameter / mean_edge_length)",
    "units": "world units (same as core/vertices)",
    "references": ["Sharp & Crane SIGGRAPH Asia 2020"]
  }
}
```

`measure/distance` has no `implemented_by` because it is purely
derived: any consumer can sum the segment lengths of `query/line`'s
output. A binding may choose to expose it as a one-call convenience
for performance — that is a binding-level decision, transparent to
the schema.

## 10. The field schema — sibling namespace

A field is a scalar or vector quantity defined on a manifold. Fields
are first-class objects in the NXR contract: `manifold.operators.d0`
sits alongside `field.gradient` in the same idiom (the slot path is
the API path).

### 10.1 What goes in `field/`

A field is for **user data** — anything a consumer creates from
measurement, simulation, generators, or as the derived output of
another field operation. The internal heat-method machinery
(`computeGeodesicDistance`, `vectorHeatTransport`, `signedHeatDistance`,
`generateHeatDiffusion`, etc.) is **not** modelled as fields; those
remain under-the-hood as machinery used by `manifold/query/*` and the
manifold's internal solvers.

### 10.2 Field slot tree

```
field/
├── attributes               ← header (manifold_ref, kind, domain, time_varying, T, units, source)
├── values                   ← the data buffer; shape depends on (kind, domain)
├── gradient                 ← scalar vertex field → face vector field
├── divergence               ← vector or 1-form → scalar vertex field
├── isolines                 ← scalar vertex field → polyline contours
├── streamlines              ← face vector field → polyline traces
├── hodge                    ← edge 1-form → α + β + γ decomposition
└── poisson                  ← scalar vertex field → harmonic potential
```

### 10.3 Polymorphic shape

A field's `values` slot has a polymorphic shape because (kind, domain)
varies per field instance:

| kind   | domain | shape    | memory_bytes  |
|---|---|---|---|
| scalar | vertex | `[V]`    | `8 * V`       |
| scalar | edge   | `[E]`    | `8 * E`       |
| scalar | face   | `[F]`    | `8 * F`       |
| vector | vertex | `[V, 3]` | `8 * V * 3`   |
| vector | edge   | `[E, 3]` | `8 * E * 3`   |
| vector | face   | `[F, 3]` | `8 * F * 3`   |

The schema records this as `[D] (scalar) | [D, 3] (vector)` in the
`shape` attribute of `field/values`, where D is V, E, or F per
`attributes.domain`. This is documentation; a downstream consumer
authoring an `nxr-viewer` panel reads `attributes.kind` /
`attributes.domain` to resolve the concrete shape at the call site.

### 10.4 `applies_when` preconditions

Each field operation carries an `applies_when` predicate over the
field's attributes. Examples:

| Operation | `applies_when` |
|---|---|
| `gradient`     | `kind == 'scalar' && domain == 'vertex'` |
| `divergence`   | `kind == 'vector' || (kind == 'scalar' && domain == 'edge')` |
| `isolines`     | `kind == 'scalar' && domain == 'vertex'` |
| `streamlines`  | `kind == 'vector' && domain == 'face'` |
| `hodge`        | `domain == 'edge'` |
| `poisson`      | `kind == 'scalar' && domain == 'vertex'` |

The predicates are documentation, not parsed. Bindings throw
`InvalidInput` if a consumer calls an operation on a field whose
attributes don't satisfy the precondition.

### 10.5 Worked example — `field/gradient`

```json
"gradient": {
  "attributes": {
    "description": "Gradient of a scalar vertex field, returned as per-face 3D vectors via Whitney interpolation of d0(values).",
    "applies_when": "attributes.kind == 'scalar' && attributes.domain == 'vertex'",
    "inputs": [],
    "output": {
      "shape": "[F, 3]",
      "dtype": "float64",
      "storage": "row_major",
      "memory_bytes": "8 * F * 3"
    },
    "depends_on": ["values"],
    "formula": "gradient(values) on face f = (1 / 2 A_f) Σ (values_k - values_i) (n × e_ji)",
    "implemented_by": {
      "cpp":        "nxr::compute::scalarGradient(ctx, values)",
      "node_addon": "ctx.scalarGradient(values)",
      "wasm":       "ctx.scalarGradient(values)",
      "mex":        "nxr_compute('scalarGradient', V, F, values)"
    },
    "complexity": "O(F)",
    "errors": ["InvalidInput"],
    "units": "values-units / world units"
  }
}
```

### 10.6 Mapping to today's bindings

The field schema is largely a renaming of operations the current
WASM / addon / MEX bindings already expose. The intended dotted-path
API drops the verbose `compute*` prefixes:

| Today's API (e.g. WASM)                        | Schema path                  |
|---|---|
| `ctx.scalarGradient(values)`                    | `field.gradient` (with `field.values = values`) |
| `ctx.computeIsolines(values, n, lo, hi)`        | `field.isolines(num_levels=n, min=lo, max=hi)` |
| `ctx.traceStreamlines(faceField, …)`            | `field.streamlines(num_seeds=…, …)` |
| `ctx.hodgeDecompose(omega)`                     | `field.hodge` (with `field.values = omega`, `domain='edge'`) |
| `ctx.solvePoisson(verts, vals)`                 | `field.poisson` (with `field.values` densely populated) |

Bindings can converge on the schema's idiom over time without
breaking existing call sites — the old methods stay; the new dotted
API is sugar.

## 11. Meta-schema responsibilities

The two namespace meta-schemas (`manifold.meta.schema.json`,
`field.meta.schema.json`) are Draft 2020-12 JSON Schemas that import
shared $defs from `common.meta.schema.json` via `$ref`. Each
meta-schema enforces:

1. **Top-level structure.** Manifold must have `attributes`, `core`,
   `embedding`, `topology`, `geometry`, `operators`, `eigen`,
   `parametrization`, `query`, `measure`. Field must have
   `attributes`, `values`, `gradient`, `divergence`, `isolines`,
   `streamlines`, `hodge`, `poisson`. Both close `additionalProperties`
   at the top level so unknown sections are flagged.
2. **Leaf-slot shape.** Each leaf's `attributes` must contain
   `description`, `shape`, `dtype`, `storage`, `memory_bytes`,
   `depends_on`, `complexity`, plus either `formula` or `implemented_by`.
3. **Operation-slot shape.** Each operation's `attributes` must
   contain `description`, `inputs`, `output`, `complexity`, plus
   either `formula` or `implemented_by`. `applies_when` is optional
   (used by field operations to document field-kind preconditions).
4. **dtype / storage enums.** Closed enums per §7.1 / §7.2, defined
   once in `common.meta.schema.json`.
5. **`implemented_by` keys.** `cpp` is required; `node_addon`, `wasm`,
   `mex`, `cli` are optional but must be strings if present.
6. **`depends_on` paths.** Slash-separated strings rooted at a
   top-level section. Cross-namespace paths use a leading namespace
   prefix (e.g. `manifold/operators/d0` from inside a field schema).
   The meta-schema does not currently verify resolution — that is a
   static analysis follow-up.

`common.meta.schema.json` is the cross-namespace contract: when
`graph.schema.json` and `recording.schema.json` arrive, they
import the same `$defs` (`leaf_slot`, `operation_slot`, `dtype`,
`storage`, `binding_map`) so all schemas stay structurally consistent.

## 12. Out of scope (deferred)

- **Actual instance values.** A separate "manifold instance"
  document type, holding the concrete buffers for a specific mesh,
  is a future Zarr/HDF5 schema problem.
- **Path validation.** Verifying that every `depends_on` resolves to
  an existing slot needs a small static analyser; the meta-schema
  cannot express this constraint cleanly.
- **Cross-namespace references.** When `graph.schema.json` arrives,
  some manifold slots may want to reference graph slots (e.g., a
  sub-graph induced on the manifold). Cross-references will need
  their own `$id` / `$ref` convention.
- **Versioning.** A `version` field at the schema root captures
  the contract version; semantic-versioning policy (when does a
  change require a major bump?) is unwritten.
- **Provenance auditing.** The schema describes what *can* exist;
  it does not enforce that a manifold instance carries a specific
  set of slots. Tooling that walks an instance and checks "this
  manifold has eigen but no parametrization, is that OK?" is a
  separate concern.

## 13. Open questions

1. **Does `parametrization/` truly belong as one branch, or should
   global UV (BFF) and local frame fields (cross/line/stripes) split?**
   Locked as one branch for v1 on the basis that they are all "putting
   coordinates on the surface." Re-evaluate if a downstream consumer
   needs to enumerate "all global parametrizations" without the local
   frame fields.

2. **Should `embedding/` carry `position[3] + quaternion[4] + scale[3]`
   alongside `transform[4×4]`?** Locked on the matrix form for v1.
   Decomposition can be added as additional sibling slots with formulas
   relating them.

3. **`measure/distance` redefinition.** The current attribute defines
   distance as the polyline length of `query/line`. The heat-method
   geodesic distance (`computeGeodesicDistance` in C++) is a different,
   per-vertex scalar field — it does not belong in `measure/` because
   its output is a buffer, not a scalar. If a consumer later needs
   "the geodesic distance between two vertices" without first calling
   `query/line`, that is a performance optimisation; both compute the
   same quantity.

## 14. References

- `include/nxr/compute.h` — the C++ surface this schema describes
- `bindings/wasm/js/index.d.ts` — the existing JS-facing type
  declarations, partial reference for the `implemented_by.wasm`
  values
- `docs/architecture.md` — the four-layer architecture
- `docs/extensions.md` — earlier proposal, partly superseded
- Knöppel & Crane, *Globally Optimal Direction Fields*, SIGGRAPH 2013
- Sharp & Crane, *You Can Find Geodesic Paths in Triangle Meshes by
  Just Flipping Edges*, SIGGRAPH Asia 2020
- Sharp, Soliman, Crane, *The Vector Heat Method*, SIGGRAPH 2019
- Sawhney & Crane, *Boundary First Flattening*, ACM TOG 2017
