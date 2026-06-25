# Vector-bundle covariant operators — design

**Date:** 2026-06-25
**Status:** design (approved in brainstorming; pending spec review)
**Area:** `src/covariant_differential.cpp` / `src/connection_laplacian.cpp` (Cell A);
new extrinsic-Weitzenböck assembly (Cell B); `include/nxr/compute.h` (`OperatorId`,
facet/API); `operator_registry.{h,cpp}` + `field_registry.{h,cpp}`; MEX + WASM `operators`.

---

## 1. Motivation — closing the vector-bundle covariant derivatives

The intrinsic / extrinsic / flat decomposition of a covariant derivative (the Gauss
formula `d_X Y = ∇^LC_X Y + II(X,Y)·N`) appears on three bundles in nxr-compute:
tangent, ambient (ℝ³), and immersion (quaternion/spinor). The **spinor** bundle is
complete — it has first-order intrinsic (`intrinsicDirac`) and extrinsic
(`extrinsicDirac`) operators and their squares. The **vector/tangent** side has two
gaps:

| cell | bundle | what exists | what's missing (this spec) |
|---|---|---|---|
| **A** intrinsic, 1st-order | tangent (complex) | `leviCivitaConnectionLaplacian` (2nd) | the first-order root `d^∇` (√ of the connection Laplacian) |
| **B** extrinsic, 2nd-order | ambient (ℝ³) | `flatCovariantLaplacian`/`gradient3D` (flat); `productCovariantLaplacian` | `extrinsicWeitzenbockLaplacian` `Δ₃+W_extrinsic` (catalogued `planned`) |

**Use cases.** Cell A: differentiating an *intrinsic* tangent field (smooth direction
field, fibre orientation, tangential flow) with the surface's own connection — today
only the energy (Laplacian) is available, not the gradient itself. Cell B: a
curvature-aware ambient operator that folds in *how the cortex bends in space* (the
shape operator), as opposed to the flat `gradient3D`/`flatCovariantLaplacian` that
treat the field as raw ℝ³. Together they bring the vector bundle to the same
intrinsic/extrinsic completeness the spinor bundle already has.

**There is no separate "first-order extrinsic covariant derivative."** The extrinsic
term `II·N` is a zeroth-order (algebraic) shape-operator endomorphism, not a
derivative, so it enters only as the curvature term in the 2nd-order Cell B — the
deliberate asymmetry vs the spinor bundle.

---

## 2. Cell A — `connectionGradient` (intrinsic first-order covariant gradient `d^∇`)

### Construction

The connection Laplacian is assembled (`src/connection_laplacian.cpp`) per halfedge
as `weight·(z_i − ρ_ij z_j)` with `ρ_ij = transportVectorsAlongHalfedge.pow(nSym)`
and `weight = edgeCotanWeight` — i.e. it is already `(d^∇)ᴴ ⋆₁ d^∇` waiting to be
factored. Cell A builds the factor directly, the complex/curved analogue of
`gradient3D`'s real flat stencil:

```
d^∇ : tangentVertex [V] (complex)  →  tangentEdge [E] (complex)
per oriented edge e: i→j           δ_e = z_j − ρ_ij · z_i
                                   ρ_ij = transportVectorsAlongHalfedge[·].pow(nSym)
```

with `⋆₁ = diag(edgeCotanWeight)`. By construction the **`squares_to` identity**

```
(d^∇)ᴴ ⋆₁ d^∇  ==  connectionLaplacian      (relation: exact, per nSym and per gauge)
```

holds, mirroring `gradient3D`'s `GᵀWG == flatCovariantLaplacian`.

### Parameterization (decided)

- **nSym** — yes (1 vector / 2 line / 4 cross). `ρ_ij.pow(nSym)`, exactly as the
  Laplacian. Needed because the consumers (smooth direction fields) use nSym 2/4, and
  the `squares_to` identity must hold for the same nSym the Laplacian was built with.
- **Active gauge** — yes. Built from whichever transport the active gauge produces
  (Levi-Civita default; trivial gauge → the trivial-connection root), so `squares_to`
  holds in any gauge the connection Laplacian supports. This makes Cell A **two
  curated registry variants**, `leviCivitaConnectionGradient` /
  `trivialConnectionGradient`, mirroring the Laplacian's gauge split (same
  `OperatorId`, different active gauge).
- **Vertex domain only** (v1). Tangent data lives on vertices; the face / edge-CR
  connection domains are additive later (they are separate `domain` values already on
  the Laplacian).

### Storage / representation

`d^∇` is complex `[E×V]`; the input is `tangentVertex` (complex per vertex,
`intrinsic_complex`), the output is a complex tangent 1-form per edge — a **new field
variant `tangentEdge`** (the tangent-bundle sibling of `ambientEdge`). The complex
matrix crosses bindings as COO with `realData`/`imagData` (the existing convention for
the complex connection Laplacian).

---

## 3. Cell B — `extrinsicWeitzenbockLaplacian` (`planned` → `built`)

### Construction shape

```
extrinsicWeitzenbockLaplacian = flatCovariantLaplacian   (Δ₃, ambient, [3N×3N] real)
                              + W_extrinsic               (zeroth-order shape-operator endomorphism)
```

`W_extrinsic` is the **extrinsic curvature endomorphism** built from the Gauss-map
data the extrinsic Dirac already uses (per-face vertex-normal differences
`(N_r − N_q)/2A`), assembled as a per-vertex symmetric `3×3` real block, mass-weighted
by vertex dual area. `bundle = ambient`, `holonomy = extrinsic_curved`,
`order = second`, `role = laplacian`, `field_type = real`, `domain = vertex`.

### Open discretization, bounded by two anchors (decided)

The **exact discrete form of `W_extrinsic`** (sign, area weighting, whether/how the
normal-tilt off-diagonal couples into the normal component) is an **implementation
design point**, developed against the MATLAB reference. The spec fixes correctness via
two verification anchors that the discretization must hit:

1. **Cross-bundle anchor.** Acting on the ambient/vector embedding, the operator must
   agree (to `< 1e-9` on the test mesh) with the **vector part of the immersion
   squared Dirac** `2·relativeDirac(½) = Δ₄ + D_N` — i.e. the same Gauss-formula /
   Weitzenböck decomposition, read on the ℝ³ (imaginary-quaternion) components.
   *(Caveat: the quaternionic `leftMulImag` mixes scalar↔imaginary, so this is the
   vector projection of the spinor square, not a literal invariant restriction — the
   anchor is the agreement to tolerance, and the plan documents the exact projection.)*
2. **Gauss-formula / internal anchor.** The extrinsic content equals the tangent↔normal
   coupling that the difference `flatCovariantLaplacian − productCovariantLaplacian`
   exposes (Product drops the shape-operator coupling that Ambient keeps). Both
   operators are already built, so this is a zero-new-dependency internal consistency
   check on `W_extrinsic`.

### Cross-link

Cell B has **no** `squares_to`/`square_of` — it is a sum `Δ₃ + W` whose extrinsic term
is algebraic, with no clean first-order root (per §1). (`productCovariantLaplacian`
similarly carries no cross-link; this is consistent.)

---

## 4. Registry & field integration

### Operator registry (`operator_registry.{h,cpp}`)

- **`OperatorId::ConnectionGradient`** (new enum) → curated variants
  `leviCivitaConnectionGradient` (`holonomy intrinsic_curved`, `gauge levi_civita`) and
  `trivialConnectionGradient` (`holonomy flat`, `gauge trivial`, `singular chi_defects`),
  both `bundle tangent`, `field_type complex`, `order first`, `role gradient`,
  `domain edge` (edge←vertex), each `squares_to` its matching connection-Laplacian
  variant (`relation exact`). Field I/O `tangentVertex → tangentEdge`. Added to the
  `variantIdsFor` no-`default` switch (compile-time completeness).
- **`OperatorId::ExtrinsicWeitzenbock`** (new enum, replaces the `Gradient3D`
  placeholder on the existing `extrinsicWeitzenbockLaplacian` entry); flip
  `status: planned → built`; field I/O already `ambientVertexLocal → ambientVertexLocal`.
  Repoint `variantIdsFor(ExtrinsicWeitzenbock)` to it and drop the placeholder note on
  `Gradient3D`.

### Field registry (`field_registry.{h,cpp}`)

- New variant **`tangentEdge`** — `domain edge, bundle tangent, field_type complex,
  n_form na, representation intrinsic_complex, gauge levi_civita`. Notes: "complex
  tangent 1-form on edges; connectionGradient output." Catalogue + the existing
  completeness/integrity tests pick it up automatically.

---

## 5. Verification

A new `test_vector_covariant_operators.cpp` (native), on a closed mesh (icosphere):

**Cell A:**
- `squares_to`: assemble `d^∇` and the connection Laplacian for nSym ∈ {1,2}; assert
  `(d^∇)ᴴ diag(edgeCotanWeight) d^∇ == connectionLaplacian` to `< 1e-9`.
- gauge tracking: in the trivial gauge (singularities Σ=χ), assert `(d^∇)ᴴ⋆₁d^∇ ==
  trivialConnectionLaplacian`.
- annihilation: on a flat patch (or up to curvature tolerance), a transported-constant
  tangent field maps near zero (documented as curvature-bounded, not exact-zero on a
  curved mesh — there is no global parallel field).
- registry: `operatorById("leviCivitaConnectionGradient").squares_to == "leviCivitaConnectionLaplacian"`,
  field I/O resolves, `tangentEdge` in the catalogue.

**Cell B:**
- cross-bundle anchor (§3.1) and Gauss-formula anchor (§3.2), both `< 1e-9`.
- registry: `status == built`, `op_id == ExtrinsicWeitzenbock`, `variantIdsFor`
  completeness still compiles.

The existing `test_operator_registry` / `test_field_registry` completeness +
cross-reference tests must stay green (the new `OperatorId`s force `variantIdsFor`
entries via `-Werror=switch`).

---

## 6. Bindings

- **MEX/WASM `operators`**: `connectionGradient` as a new family (returns complex COO;
  takes an `nSym` opt and tracks the handle's active gauge), and
  `extrinsicWeitzenbock` as a `laplacian` subtype (or its own family) returning real
  COO/sparse. `operatorInfo`/`fieldInfo` surface the new entries automatically from the
  registries. No new bespoke marshaling — reuse the complex-COO path the connection
  Laplacian already uses (Cell A) and the real-sparse path (Cell B).

---

## 7. Files

- **Modify:** `src/connection_laplacian.cpp` or `src/covariant_differential.cpp`
  (assemble `d^∇`; reuse the transport/weight reads already there), `include/nxr/compute.h`
  (`OperatorId` += `ConnectionGradient`, `ExtrinsicWeitzenbock`; facet/API decls),
  the relevant `*.cpp` for the operators facet, `operator_registry.{h,cpp}`,
  `field_registry.{h,cpp}`, `bindings/mex/src/nxr_compute_mex.cpp`,
  `bindings/wasm/src/nxr_compute_wasm.cpp`, `CMakeLists.txt`, `CLAUDE.md`.
- **Create:** new extrinsic-Weitzenböck assembly source (Cell B), `test/test_vector_covariant_operators.cpp`.

---

## 8. Scope, sequencing, deferred

- **Sequence Cell A first.** It is a clean exact-root construction (factoring an
  existing operator); it ships independently. Cell B's `W_extrinsic` discretization is
  open (bounded by the §3 anchors) and may iterate — Cell A must not be blocked on it.
- **Deferred:** face / edge-CR domains for `connectionGradient`; a first-order extrinsic
  derivative (does not exist — §1); the tangent-bundle **tensor** (rank-2) variant and
  the per-component face Jacobian (separate cataloguing question raised by the
  vector-gradient discussion, out of scope here); exposing `requireField` to bindings.
