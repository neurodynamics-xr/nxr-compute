# Field Type System — design

**Date:** 2026-06-25
**Status:** design (approved in brainstorming; pending spec review)
**Area:** new `include/nxr/field_registry.h` / `src/field_registry.cpp`; extends `include/nxr/operator_registry.h` / `src/operator_registry.cpp`; `fieldInfo` in MEX + WASM bindings.

---

## 1. Motivation

The operator registry (`docs/.../2026-06-25-operator-registry-metadata-design.md`)
gave nxr-compute a machine-checkable type system for *operators*. It typed
**half** of the picture. The other half — the **fields** those operators act on —
remains untyped: every field is a bare `Eigen::VectorXd`/`MatrixXd` whose meaning
(bundle, domain, n-form degree, representation, gauge) is implicit by convention,
and the flatten step at the binding boundary erases even the element-domain type
that geometry-central's `MeshData<E,T>` carried internally. Correctness is caller
responsibility; nothing checks that a tangent operator receives a tangent field,
that a 1-form is edge-shaped, or that an `immersion` quaternion field isn't a
4-component ambient vector.

This design adds the **dual of the operator registry**: a machine-checkable type
system for fields. It is the natural next step toward nxr-compute as a full
**field-analysis** engine (not just geometric shape analysis), oriented to
neuroscience fields on the cortical surface — scalar activations, tangent vector
fields, and ambient ℝ³ current-source vectors (Brainstorm-style).

### Goals (v1)

- A `FieldDescriptor` schema over controlled vocabularies (reusing the operator
  registry's enums), making a field's spatial type explicit and queryable.
- A catalogue of canonical **Field variants** — the "Field registry" mirroring
  `OperatorVariant`.
- **Routing**: operators declare `input_field`/`output_field`; `requireField`
  validates (the inverse of `requireBundle`); `operatorsAccepting` queries.
- A **conversion graph** of representation transitions, each wired to the existing
  function that already implements it (no new numerics).
- `fieldInfo` in MEX + WASM, mirroring `operatorInfo`.

### Non-goals (v1)

- **No rewiring** of existing field-producing/consuming functions to take/return a
  typed `Field`. They keep bare-Eigen signatures; the typed layer wraps/validates
  at chosen boundaries. Additive, zero breakage — exactly how the operator
  registry shipped.
- **No time axis.** Time is a *future orthogonal subsystem* — a temporal container
  that is the time-axis counterpart to `MeshData`'s spatial containers (§3). v1 is
  single-time-step spatial fields only. The sole guardrail: a Field is defined as
  "a typed payload over a domain," **never** as "a length-`nElements` vector," so a
  future temporal container composes *over* spatial fields without reopening the
  spatial type.
- **No halfedge/corner domains** (§6). Vertex/edge/face cover every cortical *data*
  field; halfedge/corner data is gauge/operator plumbing. `Domain` extends
  additively if a real halfedge/corner data field appears.
- **Frames are excluded as a category** (§4). A frame is a *coordinate system*, not
  a field; it lives in the gauge layer and is only *referenced* by a field's
  `representation`/`gauge` axes.
- No new conversion numerics; no `Field<Scalar>` container threaded through APIs.

---

## 2. Relationship to geometry-central and the operator registry

The full stack this completes:

```
┌──────────────────────────────────────────────────────────────────┐
│  Field registry (THIS)        ⇄        Operator registry (DONE)    │  semantic
│  types DATA:                           types OPERATORS:            │  layer
│  bundle·domain·n_form·field_type       bundle·holonomy·order·…     │  (duals,
│  ·representation·gauge·nSym            + input_field/output_field  │  cross-linked)
│        requireField(field, op)  ==  inverse of requireBundle       │
├──────────────────────────────────────────────────────────────────┤
│  MeshData<E,T> containers          │   DEC ops · halfedge · geometry │ base layer
│  (data-on-domain primitive;        │   · gauge/transport/direction   │ (structural +
│   the Field's internal backing)    │     field algorithms)           │  container
└──────────────────────────────────────────────────────────────────┘
```

`MeshData<E,T>` is a *container* primitive (domain × payload `T`), not a semantic
field type — it cannot express `bundle`, `n_form`, `representation`, or the
intrinsic↔ambient distinction, and it does not survive the flatten to a bare Eigen
array. The Field registry restores that type information on the Eigen/cross-binding
side and adds the semantic axes `MeshData` never had. It reuses geometry-central's
tangent-bundle machinery (tangent bases, `transportVectorsAlongHalfedge`, the
`computeSmoothest*DirectionField` algorithms) rather than reinventing it; those
algorithms become the canonical `tangent`-bundle generators in the Field system.

---

## 3. Time: the deliberate guardrail (not built in v1)

Time will later be a **first-class orthogonal subsystem** — a temporal container
that is to the time axis what `MeshData` is to the spatial mesh. v1 builds none of
it and assumes single-time-step fields. The only thing v1 must *not* do is
foreclose it. Concretely:

- A Field is **"a typed payload over a domain"**, not "a vector." The descriptor
  describes one element's payload (`components_per_element`) + the domain; it has
  **no time axis and no sample/stack count**.
- A future time series is "a sequence/stack of spatial fields," owned by the
  temporal subsystem, composed *over* the spatial Field type — never by widening
  the spatial descriptor.
- Validation/routing code is written against the spatial descriptor and the
  per-element payload, so wrapping fields in a temporal container later is additive.

This means: today an `nV×1` scalar field is a `scalarVertex` Field; later an
`nV×nT` field is a *temporal container of* `scalarVertex` Fields — same spatial
type, new orthogonal axis. Nothing in v1 asserts a field is one-dimensional in a
way that blocks this.

---

## 4. `FieldDescriptor` schema

The spatial type of a field. Controlled-vocab enums; reuses the operator-registry
enums where they exist (`Domain`, `Bundle`, `FieldType`, `Gauge`).

| Axis | Vocabulary | Meaning |
|---|---|---|
| `domain` | `vertex \| edge \| face` | where DOFs live (reuse `Domain`; halfedge/corner deferred) |
| `bundle` | `scalar \| tangent \| ambient \| immersion` | what the field IS (reuse `Bundle`) |
| `field_type` | `real \| complex \| quaternion` | payload arithmetic (reuse `FieldType`) |
| `n_form` | `na \| zero \| one \| two` | DEC degree; meaningful for `bundle=scalar` (0-form vertex / 1-form edge / 2-form face), `na` otherwise |
| `representation` | `na \| world \| local_frame \| intrinsic_complex \| quaternion_interleaved` | component layout *relative to* a coordinate system (never stores one) |
| `gauge` | `na \| euclidean \| levi_civita \| trivial` | *which* coordinate system the `local_frame`/`intrinsic_complex` representation references (reuse `Gauge`) |
| `nSym` | int (default 1) | n-RoSy order for tangent direction fields (1 vector / 2 line / 4 cross) — a parameter, not a separate variant |

`components_per_element` is **derived** (scalar=1; tangent=1 complex / 2 real;
ambient=3; immersion=4), used for shape validation, not stored.

### Frames are coordinate systems, not fields

A frame (`[e1|e2|n]`, the complex `grid` `c=e1+i·e2`) is the **basis a field's
components are expressed in** — it lives in the gauge/geometry layer (`gauge()`
facet, `vertexFrames`, the `grid`). It is **not** a Field variant. The Field system
only *references* a coordinate system: `representation` says which layout
(`world` = global Cartesian; `local_frame` = per-element `[e1,e2,n]`;
`intrinsic_complex` = tangent field as `(e1·coord)+i(e2·coord)`;
`quaternion_interleaved` = `4v+c`/`4f+c`), and `gauge` says which frame/connection
convention. The frame itself is never copied into a Field.

---

## 5. Field variant catalogue

Named canonical variants — the "Field registry," mirroring `OperatorVariant`. Each
is a `FieldVariant{ id, label, descriptor, notes }` with `byId`/`where` queries.

| `id` | domain | bundle | field_type | n_form | representation | gauge | notes |
|---|---|---|---|---|---|---|---|
| `scalarVertex` | vertex | scalar | real | zero | na | na | 0-form / vertex function |
| `oneFormEdge` | edge | scalar | real | one | na | na | DEC 1-form |
| `twoFormFace` | face | scalar | real | two | na | na | per-face scalar / 2-form |
| `tangentVertex` | vertex | tangent | complex | na | intrinsic_complex | levi_civita | n-RoSy via `nSym` |
| `tangentFace` | face | tangent | complex | na | intrinsic_complex | levi_civita | n-RoSy via `nSym` |
| `ambientVertexWorld` | vertex | ambient | real | na | world | na | ℝ³ in global Cartesian (e.g. leadfield) |
| `ambientVertexLocal` | vertex | ambient | real | na | local_frame | levi_civita | ℝ³ in per-vertex frame `[a;b;c]` |
| `ambientEdge` | edge | ambient | real | na | local_frame | levi_civita | covariant-difference output (3E) |
| `ambientFaceWorld` | face | ambient | real | na | world | na | ℝ³ per face (gradient/whitney output) |
| `immersionVertex` | vertex | immersion | quaternion | na | quaternion_interleaved | na | `4v+c` shape-spinor |
| `immersionFace` | face | immersion | quaternion | na | quaternion_interleaved | na | `4f+c` shape-spinor |

(n-RoSy line/cross fields are `tangentVertex`/`tangentFace` with `nSym=2`/`4`, not
separate variants. Further ambient/tangent variants extend additively.)

---

## 6. Operator coupling (extend `OperatorVariant`)

Extend `OperatorVariant` with two fields:

```cpp
std::string input_field;    // Field-variant id the operator consumes
std::string output_field;   // Field-variant id it produces
```

Populated per operator. Representative mappings:

| operator | input_field | output_field |
|---|---|---|
| `laplaceBeltrami`, `graphLaplacian` | `scalarVertex` | `scalarVertex` |
| `faceLaplacianGreenGauss`, `faceLaplacian2Form` | `twoFormFace` | `twoFormFace` |
| `leviCivitaConnectionLaplacian`, `trivialConnectionLaplacian` | `tangentVertex` | `tangentVertex` |
| `flatCovariantLaplacian`, `productCovariantLaplacian` | `ambientVertexLocal` | `ambientVertexLocal` |
| `covariantGradient` | `ambientVertexLocal` | `ambientEdge` |
| `faceGradient` | `twoFormFace` | `ambientFaceWorld` |
| `relativeDirac` | `immersionVertex` | `immersionVertex` |
| `extrinsicDirac`, `intrinsicDirac` | `immersionVertex` | `immersionFace` |
| `relativeFaceDirac` | `immersionFace` | `immersionFace` |
| `extrinsicFaceDirac`, `intrinsicFaceDirac` | `immersionFace` | `immersionVertex` |
| `massLumped`, `massGalerkin` | `scalarVertex` | `scalarVertex` |
| `d0` | `scalarVertex` | `oneFormEdge` |
| `d1` | `oneFormEdge` | `twoFormFace` |
| `hodge0`/`hodge1`/`hodge2`/`hodge1inv` | scalar form of degree | dual-degree form |

### Routing API (`field_registry.h`)

```cpp
// Structural compatibility: a field matches an operator's input iff its descriptor
// agrees on domain, bundle, field_type, n_form, representation (gauge & nSym are
// advisory parameters, not match keys).
bool fieldMatches(const FieldDescriptor& f, const FieldDescriptor& expected);

// Inverse of requireBundle: throw Error(InvalidInput) if `f` is not admissible
// as the input of operator `operatorId`.
void requireField(const FieldDescriptor& f, std::string_view operatorId);

// Query: which operators accept a field of this descriptor as input.
std::vector<std::string> operatorsAccepting(const FieldDescriptor& f);

// Optional shape validation against a mesh: rows == nElements(domain)*components.
void validateFieldShape(const FieldDescriptor& f, int rows, int nV, int nE, int nF);
```

### Cross-reference integrity

Every `OperatorVariant.input_field` / `output_field` must name a Field variant that
exists in the catalogue. Enforced by a test (`test_field_registry`) that walks the
operator registry and asserts each referenced id resolves via `fieldById` — the
field analogue of the operator-registry completeness guard.

---

## 7. Conversion graph (declared + wired to existing impls)

Typed edges between Field variants, each pointing at the existing function that
implements it. No new numerics in v1; missing routes are catalogued as
`declared, unimplemented`.

```cpp
struct ConversionEdge {
    std::string from;        // Field-variant id
    std::string to;          // Field-variant id
    std::string impl;        // existing function name (or "" if unimplemented)
    bool        implemented; // false ⇒ declared-only
};
const std::vector<ConversionEdge>& conversionGraph();
```

| from → to | impl |
|---|---|
| `ambientVertexWorld` ↔ `ambientVertexLocal` | `differential::liftToWorld` / `liftToFrame` |
| `ambientVertexWorld` → intrinsic (`ambientVertexLocal`) | `G·cᵀ` (`differential::covariantGradient` correspondence) |
| `oneFormEdge` → `ambientFaceWorld` | `field::interp::whitney` |
| `scalarVertex` → `ambientFaceWorld` | `field::op::gradient` (`scalarGradient`) |
| `tangentVertex` (complex) ↔ real-2N | connection-Laplacian `lowerToReal2N` embedding |

Each edge's `from`/`to` must resolve to a catalogued variant, and each
`implemented` edge's `impl` must name a real function (test-checked by string
presence in the API; full call-through is a later phase).

---

## 8. Bindings

- **MEX:** `nxr_compute('fieldInfo', id)` → struct of the Field variant's metadata
  strings (mirrors `operatorInfo`). `operatorInfo` additionally surfaces
  `input_field` / `output_field`.
- **WASM:** `manifold.fieldInfo(id)` → the same object shape (Embind), parity with
  MEX verified.
- `requireField` / `operatorsAccepting` are **C++-internal** in v1 (validation +
  query); only `fieldInfo` and the extended `operatorInfo` cross the binding.

---

## 9. Files & enforcement

- **New:** `include/nxr/field_registry.h` (`FieldDescriptor`, `FieldVariant`,
  `ConversionEdge`, the catalogue accessors `fieldRegistry()`/`fieldById`/
  `fieldsWhere`, the routing API of §6, `conversionGraph()`),
  `src/field_registry.cpp` (the catalogue + conversion table + impls),
  `test/test_field_registry.cpp`.
- **Modified:** `operator_registry.h` (`OperatorVariant` += `input_field`,
  `output_field`), `operator_registry.cpp` (populate them),
  `bindings/mex/src/nxr_compute_mex.cpp` (`fieldInfo` + surface op I/O),
  `bindings/wasm/src/nxr_compute_wasm.cpp` (`fieldInfo`), `CMakeLists.txt`,
  `CLAUDE.md`.
- **Tests (`test_field_registry.cpp`):** variant catalogue completeness (every id
  resolves); operator I/O cross-reference integrity (every operator's
  `input_field`/`output_field` resolves); conversion edges reference real variants
  and (for implemented edges) real functions; `requireField` accept/reject cases
  (e.g. a `tangentVertex` rejected by `laplaceBeltrami`, an `immersionVertex`
  rejected where `ambient` is required); `validateFieldShape` length checks.
- **Binding tests:** MEX `fieldInfo` round-trip; WASM `fieldInfo` smoke; field
  metadata parity between the two.

---

## 10. Deferred

- The time/temporal subsystem (the orthogonal time-axis container; §3).
- Halfedge/corner domains.
- Rewiring existing functions to take/return a typed `Field`; a `Field<Scalar>`
  container; a `MeshData`↔Field marshaling bridge as a first-class API.
- Conversion *call-through* (v1 declares + names impls; invoking conversions
  through the registry is a later phase).
- `requireField`/`operatorsAccepting` exposed to bindings.
