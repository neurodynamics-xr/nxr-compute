# NXR schema conventions

A short reference for authors and reviewers working with the NXR
component schemas. Today: `manifold.schema.json` (the surface-mesh
contract) and `field.schema.json` (scalar / vector fields defined on
a manifold). Future: `graph.schema.json`, `recording.schema.json`,
`volume.schema.json`. All NXR schemas share a common primitive
library in `common.meta.schema.json` ($defs for `dtype`, `storage`,
`leaf_slot`, `operation_slot`, `binding_map`, `group_attributes`).

The full design rationale lives in
`docs/superpowers/specs/2026-05-09-manifold-api-schema-design.md`.
This document is the quick-reference companion.

## File map

```
docs/schema/
├── common.meta.schema.json      ← shared $defs (dtype, storage, leaf_slot, …)
├── manifold.meta.schema.json    ← validates manifold.schema.json
├── manifold.schema.json         ← the manifold contract
├── field.meta.schema.json       ← validates field.schema.json
└── field.schema.json            ← the field contract
```

---

## What a schema is

A **design-time contract** that other NXR components are built
against. Not loaded at runtime, not used for code generation. The
schema documents the shape of an object (its slots, their types,
their dependencies, their producing operations) so that nxr-viewer
and any other consumer can be designed against a stable target.

Bindings (`nxr_compute_addon.node`, `nxr_compute.wasm`,
`nxr_compute.mexw64`) stay hand-written. Drift between bindings and
schema is caught by human review.

## Anatomy of a slot

Every node in the slot tree — group or leaf — carries an
`attributes` block. Group `attributes` describe the group; leaf
`attributes` describe one piece of data or one operation.

```json
{
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
        "cpp":        "nxr::compute::assembleMeshOperators().cotanLaplacian",
        "node_addon": "ctx.assembleMeshOperators().cotanLaplacian",
        "wasm":       "ctx.assembleMeshOperators().cotanLaplacian",
        "mex":        "nxr_compute('assembleMeshOperators', V, F).cotanLaplacian"
      },
      "complexity": "O(F)"
    }
  }
}
```

## Required attributes

For **data leaves**:

- `description` — one-line prose
- `shape` — symbolic, e.g. `[V, 3]`, `[V, V] sparse`, `[K]`
- `dtype` — see enum below
- `storage` — see enum below
- `memory_bytes` — symbolic formula, e.g. `8 * V * 3`
- `depends_on` — array of slot paths (may be empty)
- `complexity` — big-O for the producing op
- Either `formula` (derived slot) or `implemented_by` (non-derived)

For **operations** (under `query/`, `measure/`):

- `description`
- `inputs` — array of `{ name, type }`
- `output` — `{ shape, dtype, storage, memory_bytes? }`
- `complexity`
- Either `formula` or `implemented_by`

## Optional attributes

- `errors` — subset of `nxr::compute::ErrorCode` enumerator names
- `cancellable` — bool, default `false`
- `progress` — bool, default `false`
- `units` — `"meters"`, `"radians"`, `"dimensionless"`, etc.
- `references` — array of citation strings

## dtype enum

```
float64   float32
int32     int64     uint8     uint32
complex64 complex128
bool
```

`bool` is materialized as `uint8` everywhere; the dtype tag is
documentation only.

## storage enum

| Value | Meaning |
|---|---|
| `row_major` | Dense N-D buffer, row-major layout |
| `col_major` | Dense N-D buffer, column-major layout |
| `sparse_coo` | `{ row, col, data, rows, cols, nnz }` triplets |
| `sparse_csc` | Eigen / MATLAB native CSC |
| `sparse_csr` | CSR (admitted; not currently used by any binding) |
| `diagonal` | Sparse with only the main diagonal populated; bindings may flatten to length-N dense |
| `complex_interleaved` | dtype is complex; layout is `re_0, im_0, re_1, im_1, …` |
| `struct` | Record of named sub-slots, each with own attributes |
| `scalar` | Single value, no shape |

## memory_bytes vocabulary

Symbolic formulas use these tokens:

```
V   — n_vertices
E   — n_edges
F   — n_faces
H   — n_halfedges (= 2*E for closed manifolds; less for meshes with boundary)
K   — n_eigenmodes (only inside eigen/)
N   — operation-specific (e.g. number of streamline segments, polyline points)
T   — number of timesteps (in time-varying contexts)
nnz(<slot_path>) — non-zeros in a sparse slot
```

Constants follow C-style arithmetic. Examples:

```
8 * V * 3                       — V×3 float64
12 * nnz(operators/d0)          — COO triplets, 8B data + 4B row + 4B col
16 * V                          — V complex128 values, interleaved
4 + 8                           — small struct of one int32 + one float64
```

`nnz(...)` is left symbolic; the schema does not predict non-zero
counts.

## formula syntax

ASCII-only. Recognised tokens:

- `*` — multiplication (matrices or scalars)
- `+`, `-` — addition / subtraction
- `^T` — transpose (postfix)
- `^-1` — inverse (postfix)
- `<name>` — slot reference (bare name within `operators/`,
  slash-path otherwise)
- Parentheses for grouping

Examples:

```
d0^T * hodge1 * d0                 — stiffness
d0                                — gradient
d0^T * hodge1                      — divergence
hodge0^-1 * d0^T * hodge1 * d0      — Laplace–Beltrami acting on 0-forms
```

Formulas are documentation. They are not parsed or executed.

## Complex storage

Complex slots use `dtype: complex64` or `dtype: complex128` together
with `storage: complex_interleaved`. Layout is interleaved real /
imaginary pairs:

```
[re_0, im_0, re_1, im_1, ...]
```

This matches `std::complex<T>*` cast to `T*`, NumPy `np.complex128`,
MATLAB native complex, and Eigen `Matrix<std::complex<T>, …>`. A
shape of `[V]` complex128 is `16 * V` bytes; a shape of `[V, 3]`
complex128 is `24 * V` bytes (each component is one complex pair).

## implemented_by shape

```json
{
  "cpp":        "nxr::compute::assembleMeshOperators().cotanLaplacian",
  "node_addon": "ctx.assembleMeshOperators().cotanLaplacian",
  "wasm":       "ctx.assembleMeshOperators().cotanLaplacian",
  "mex":        "nxr_compute('assembleMeshOperators', V, F).cotanLaplacian"
}
```

`cpp` is required. `node_addon`, `wasm`, `mex` are optional — a
binding without an entry simply does not expose the slot today.
String content is documentation; not parsed.

For derived slots (those with a `formula`), `implemented_by` may be
omitted. The formula is enough for a binding author to realize the
slot. If C++ also exposes it directly (for performance), both keys
may be present.

## depends_on paths

Slash-separated, rooted at a top-level section:

```
core/vertices
operators/d0
geometry/face/normals
topology/halfedge/twin
```

The meta-schema currently does not resolve these — a static analyser
that walks the schema and verifies every `depends_on` target exists
is on the follow-up list.

## Field schema — the sibling of manifold

`field.schema.json` describes a scalar or vector field defined on a
manifold. Every field has:

- `attributes` — header carrying `manifold_ref` (URI of the host
  manifold), `kind` ('scalar' | 'vector'), `domain` ('vertex' |
  'edge' | 'face'), and optionally `time_varying`, `T`, `units`,
  `source`.
- `values` — the data buffer (shape determined by kind/domain).
- Six operations applicable to fields, each carrying an
  `applies_when` precondition documenting which (kind, domain)
  combinations are valid:
  - `gradient` — scalar vertex field → face-vector field
  - `divergence` — vector or 1-form → scalar vertex field
  - `isolines` — scalar vertex field → polyline contours
  - `streamlines` — face-vector field → polyline traces
  - `hodge` — edge 1-form → α + β + γ decomposition
  - `poisson` — scalar vertex field → potential

A field is always defined *on* a manifold — there is no field
without a manifold. Fields are for **user data** (measurement,
simulation output, generator output, derived from another field).
Internal heat-method intermediates (heat diffusion, vector heat,
signed heat) used to compute distances, parallel transport, or the
global tangent frame are **not** fields; they live under-the-hood as
unnamed machinery inside `manifold/query/*` and the manifold's
internal solvers.

The slot-tree convention `manifold.operators.d0` (path) ↔
`ctx.operators.d0` (API) extends to fields:
`field.gradient` (path) ↔ `field.gradient(...)` (API).

## Authoring checklist

Before committing changes to `manifold.schema.json` or `field.schema.json`:

- [ ] Each new leaf has all required attributes (§ "Required attributes")
- [ ] `dtype` is one of the enum values
- [ ] `storage` is one of the enum values
- [ ] `memory_bytes` uses only the documented symbolic vocabulary
- [ ] `depends_on` paths are slash-separated and rooted at a top-level section
- [ ] If derived: `formula` is present and uses the documented syntax
- [ ] If non-derived: `implemented_by.cpp` is present
- [ ] Group `attributes` carry only group-wide facts, not data
- [ ] Run `ajv validate -s manifold.meta.schema.json -r common.meta.schema.json -d manifold.schema.json`
      (or the equivalent for field) before opening a PR
