# Operator Registry & Metadata — design

**Date:** 2026-06-25
**Status:** design (approved in brainstorming; pending spec review)
**Area:** `include/nxr/compute.h`, `include/nxr/facets.h`, new `include/nxr/operator_registry.h` / `src/operator_registry.cpp`

---

## 1. Motivation

nxr-compute has grown a *zoo* of operators — cotan / graph / connection (Levi-Civita,
trivial) / covariant (product, ambient) Laplacians, the first- and second-order
Dirac family (vertex/face, intrinsic/extrinsic, the τ-graded relative Dirac),
gradients, masses, DEC. The names are ambiguous, several **distinct** operators
hide behind a **single** `OperatorId`, and the relationships between them
(`D` squares to `E`, `intrinsicDirac` squares to `cotanL`, …) are tribal
knowledge rather than queryable data.

This causes real, structural errors. The sharpest is the **LJC trap**: the
quaternionic `immersion`-bundle operators (relative Dirac) are *not*
interchangeable with the `ambient` ℝ³ operators, yet nothing today bars an
`immersion` operator from being mis-used as a current differentiator on a
leadfield.

This design introduces a **machine-checkable operator registry**: one canonical
C++ table that assigns every operator a stable identifier, a human-readable
label, and a set of **controlled-vocabulary** metadata fields. The registry is
the single source of truth, usable both **internally** (the eigensolver's
natural-mass lookup, the LJC-trap guard) and **across bindings** (MEX/WASM
`operatorInfo`).

### Goals

- A stable, curated name + label for every operator ("Laplace–Beltrami",
  "Levi-Civita connection Laplacian", "Extrinsic Weitzenböck Laplacian", …).
- Controlled-vocabulary metadata that disambiguates operators *structurally*
  (`bundle`, `holonomy`, `order`, `field_type`, `role`, `domain`, …) so
  machine checks replace prose.
- Verified cross-links (`squares_to` / `square_of`) encoding the operator
  identities, with a `relation` qualifier distinguishing exact equality from
  principal-part agreement.
- Compile-time guarantee that no operator can exist without metadata, plus a
  numerical test that the claimed identities actually hold.
- Build the one named-but-missing operator: the **extrinsic Weitzenböck
  Laplacian** `Δ₃ + D_N` (ambient sibling of the immersion squared Dirac).

### Non-goals (this version)

- Renaming the `OperatorId` enum or any internal C++ symbol. The new vocabulary
  is **additive** — curated names live as registry fields keyed by the existing
  enum. Only binding dispatch strings adopt the new names (a binding version bump).
- A MATLAB `handle`-class wrapper (application-side, as before).
- Implementing the deferred frame-coupled α-spectral / shell operator.

---

## 2. Metadata schema (controlled vocabularies)

Every registry entry is one `OperatorVariant`. Free text is confined to `label`
and `notes`; everything else is a controlled enum so it is machine-checkable.

| Field | Vocabulary | Meaning |
|---|---|---|
| `id` | curated lowerCamel string (the **key**) | stable identifier, e.g. `leviCivitaConnectionLaplacian` |
| `label` | free text | human display name, e.g. "Levi-Civita connection Laplacian" |
| `bundle` | `scalar \| tangent \| ambient \| immersion` | what the field is. **`immersion` ≠ `ambient`** — this is the LJC-trap guard axis |
| `holonomy` | `flat \| combinatorial \| intrinsic_curved \| extrinsic_curved \| graded` | curvature carried into the operator. `flat` → differentiation (no holonomy artifact); `*_curved` → spectral basis (curvature graded into spectrum); `combinatorial` → metric-free; `graded` → a τ-family spanning intrinsic↔extrinsic |
| `order` | `zeroth \| first \| second` | `first` = Dirac/√ or a differential (d, gradient); `second` = Laplacian; `zeroth` = mass/Hodge metric |
| `role` | `laplacian \| connection_laplacian \| dirac \| gradient \| exterior_derivative \| mass \| hodge_star` | what *kind* of operator (distinguishes a mass from a Laplacian, which `order` alone cannot) |
| `field_type` | `real \| complex \| quaternion` | storage arithmetic. quaternion ⇒ 4-component (`4v+c` / `4f+c` interleaved); ambient ⇒ 3-component; complex ⇒ tangent 2-component |
| `domain` | `vertex \| face \| edge` (+ optional `→` source for rectangular ops) | where the DOFs live; rectangular operators record `codomain←domain` |
| `singular` | `none \| chi_defects{count, locations}` | forced singularities. `chi_defects` flags the trivial-gauge Σindex=χ defects |
| `gauge` | `n/a \| euclidean \| levi-civita \| trivial` | gauge the operator is assembled in (disambiguates Levi-Civita vs trivial connection L) |
| `coupling` | `n/a \| product \| ambient` | covariant-Laplacian coupling (disambiguates flat vs product) |
| `square_of` / `squares_to` | `{ target: id, relation }` | cross-link to the operator's root (second-order) or square (first-order) |
| `relation` | `exact \| principal_part` | whether the cross-link is exact matrix equality or scalar/principal-part agreement only (drives what the numerical test asserts) |
| `natural_mass` | `id \| identity \| <id>⊗I₄ \| diag(faceArea)⊗I₄` | the generalized mass its eigenproblem is posed against (folds in `eigenProblemFor`) |
| `tau` | `n/a \| [0,1]` + `presets` | for graded families: the parameter range and named presets |
| `status` | `built \| planned` | `planned` ⇒ `operatorInfo` returns metadata but assembly throws `NotImplemented` |
| `notes` | free text | caveats (e.g. closed-mesh-only; DEC identity not valid under intrinsicDelaunay) |

### Schema additions beyond the original six (recorded)

- **`domain`** — required: vertex Dirac vs face Dirac, and the three connection-L
  domains, are otherwise indistinguishable.
- **`role`** — required: `order` cannot separate a `mass` (zeroth) from a Laplacian.
- **`gauge`**, **`coupling`** — required: disambiguate the variants hiding behind
  one `OperatorId` (see §3).
- **`holonomy: combinatorial`** — added value so `graphLaplacian` (metric-free) is
  distinct from `flat` (flat *metric*, e.g. trivial connection / flat covariant).
- **`holonomy: graded`** + `tau` — added for the τ-families.
- **`relation`** qualifier — added because not every `squares_to` link is exact
  (the intrinsic Dirac agrees with `laplaceBeltrami` only on its scalar part).

---

## 3. The `OperatorVariant` key — resolving the `OperatorId` coarseness

`OperatorId` is **too coarse** to key the registry: three IDs each fan out to
several genuinely distinct operators.

| `OperatorId` | fans out to |
|---|---|
| `LaplacianConnection` | `{levi-civita, trivial}` gauge × `{vertex, face, edge}` domain |
| `LaplacianCovariant` | `{ambient, product}` coupling |
| `Dirac` / `DiracFace` | a τ-continuum (graded family) |

**Decision.** The registry is keyed on a stable `OperatorVariant` — the curated
`id` string — which **resolves** internally to a `(OperatorId, gauge, coupling,
domain, τ)` tuple. The `OperatorId` enum is unchanged; the registry adds the
finer addressing on top.

```cpp
struct OperatorVariant {
    std::string id;                 // stable key, e.g. "trivialConnectionLaplacian"
    std::string label;
    Bundle      bundle;
    Holonomy    holonomy;
    Order       order;
    Role        role;
    FieldType   field_type;
    Domain      domain;
    Singular    singular;
    Gauge       gauge;              // n/a unless connection L
    Coupling    coupling;           // n/a unless covariant L
    CrossLink   square;             // {target id, relation}  (square_of OR squares_to)
    std::string natural_mass;
    TauSpec     tau;                // n/a unless graded
    Status      status;
    std::string notes;

    // Resolution to the assembly path:
    OperatorId  op_id;              // the (possibly shared) enum slot
    // gauge/coupling/domain/τ above complete the address
};
```

Lookup is by `id`; `registry.where(predicate)` supports filtering by any facet
(e.g. `where(bundle == immersion && order == first)`).

---

## 4. The full registry

Grouped by `bundle`. `→` = `squares_to`, `√←` = `square_of`; `(e)` = relation
`exact`, `(p)` = `principal_part`. All `field_type` real unless noted.

### Scalar bundle (block 1)

| `id` | label | holonomy | order | role | domain | cross-link | `OperatorId` |
|---|---|---|---|---|---|---|---|
| `laplaceBeltrami` | Laplace–Beltrami (cotan) | intrinsic_curved | second | laplacian | vertex | √← `intrinsicDirac` (p) | `LaplacianCotan` |
| `graphLaplacian` | Graph Laplacian (d₀ᵀd₀) | **combinatorial** | second | laplacian | vertex | √← `d0` (e) | `LaplacianGraph` |
| `faceLaplacianGreenGauss` | Face Laplacian (Green–Gauss dual) | intrinsic_curved | second | laplacian | face | √← `faceGradient` (e) | `LapFace` |
| `faceLaplacian2Form` | Face Laplacian (DEC 2-form, d₁⋆₁⁻¹d₁ᵀ) | intrinsic_curved | second | laplacian | face | — (relFaceDirac τ=0 anchor) | *(internal `cacheTwoFormLaplacian_`)* |

### Tangent bundle (complex, 2-comp; `nSym` param)

| `id` | label | holonomy | order | role | domain | singular | gauge | `OperatorId` |
|---|---|---|---|---|---|---|---|---|
| `leviCivitaConnectionLaplacian` | Levi-Civita connection (Bochner) Laplacian | intrinsic_curved | second | connection_laplacian | vertex/face/edge | none | levi-civita | `LaplacianConnection` |
| `trivialConnectionLaplacian` | Trivial connection Laplacian | flat | second | connection_laplacian | vertex | **chi_defects** | trivial | `LaplacianConnection` |

### Ambient bundle (real, 3-comp ℝ³)

| `id` | label | holonomy | order | role | domain | cross-link | `OperatorId` |
|---|---|---|---|---|---|---|---|
| `flatCovariantLaplacian` | Flat covariant Laplacian (ambient) | flat | second | laplacian | vertex | √← `covariantGradient` (e) | `LaplacianCovariant` (ambient) |
| `productCovariantLaplacian` | Product covariant Laplacian (tan⊕nor) | intrinsic_curved | second | laplacian | vertex | — | `LaplacianCovariant` (product) |
| `covariantGradient` | Covariant gradient (flat transport) | flat | first | gradient | edge←vertex | → `flatCovariantLaplacian` (e) | `Gradient3D` |
| `faceGradient` | Face gradient (Green–Gauss) | intrinsic_curved | first | gradient | face | → `faceLaplacianGreenGauss` (e) | `GradFace` |
| `extrinsicWeitzenbockLaplacian` | Extrinsic Weitzenböck Laplacian (Δ₃+D_N) | **extrinsic_curved** | second | laplacian | vertex | √← *(ambient extrinsic Dirac root)* | **NEW** (`status: planned` until built) |

### Immersion bundle (quaternion, 4-comp shape-spinor)

| `id` | label | holonomy | order | role | domain | cross-link | `OperatorId` |
|---|---|---|---|---|---|---|---|
| `intrinsicDirac` | Intrinsic Dirac (1st-order) | intrinsic_curved | first | dirac | face←vertex | → `laplaceBeltrami` (p) | `DiracIntrinsicD` |
| `extrinsicDirac` | Extrinsic Dirac (1st-order, Gauss map) | extrinsic_curved | first | dirac | face←vertex | → `relativeDirac`@τ=1 (e) | `DiracD` |
| `relativeDirac` | Relative Dirac (vertex τ-family) | **graded** | second | dirac | vertex | √← `extrinsicDirac` @τ=1 (e) | `Dirac` |
| `intrinsicFaceDirac` | Intrinsic face Dirac (1st-order) | intrinsic_curved | first | dirac | vertex←face | → `faceLaplacian2Form` (p) | `DiracFaceIntrinsicD` |
| `extrinsicFaceDirac` | Extrinsic face Dirac (1st-order) | extrinsic_curved | first | dirac | vertex←face | → `relativeFaceDirac`@τ=1 (e) | `DiracFaceD` |
| `relativeFaceDirac` | Relative face Dirac (τ-family) | **graded** | second | dirac | face | √← `extrinsicFaceDirac` @τ=1 (e) | `DiracFace` |

**Relative-Dirac presets** (atlas over the same τ-family — convenience args, not
separate variants):

```
relativeDirac      tau∈[0,1]    natural_mass = massGalerkin ⊗ I₄
  intrinsic → τ=0        (cotanL ⊗ I₄ ; pure intrinsic Δ₄)
  extrinsic → τ=1        (E = DᵀW_F D ; pure extrinsic D_N)
  squared   → τ=½, ×2    (Δ₄ + D_N = D², the Weitzenböck square; spectrally = 2·relativeDirac(½))

relativeFaceDirac  tau∈[0,1]    natural_mass = diag(faceArea) ⊗ I₄
  intrinsic → τ=0        (faceLaplacian2Form ⊗ I₄)
  extrinsic → τ=1        (Ẽ = D̃ᵀW_V D̃)
  squared   → τ=½, ×2
```

> **Preset vs first-order, no collision.** The preset words `intrinsic`/`extrinsic`
> are *arguments to the second-order `relativeDirac` family*. The first-order
> operators `intrinsicDirac` / `extrinsicDirac` are distinct `id`s with their own
> roots. The `order` field separates them; bindings dispatch the presets as
> `relativeDirac(arg)`, never as standalone ids.

### Metrics & exterior calculus (`role`-tagged; `holonomy` n/a)

| `id` | label | order | role | domain | `OperatorId` |
|---|---|---|---|---|---|
| `massLumped` | Lumped (barycentric) mass | zeroth | mass | vertex | `MassLumped` |
| `massGalerkin` | Galerkin (FEM) mass | zeroth | mass | vertex | `MassGalerkin` |
| `d0` | Exterior derivative d₀ (grad) | first | exterior_derivative | edge←vertex | `Dec` |
| `d1` | Exterior derivative d₁ (curl) | first | exterior_derivative | face←edge | `Dec` |
| `hodge0` / `hodge1` / `hodge2` / `hodge1inv` | Hodge stars ⋆₀/⋆₁/⋆₂/⋆₁⁻¹ | zeroth | hodge_star | vertex/edge/face | `Dec` |

---

## 5. Cross-link semantics & the `relation` qualifier

The `squares_to` / `square_of` links encode the verified operator identities.
The `relation` qualifier tells the numerical test (§7) what to assert:

| cross-link | relation | identity the test asserts |
|---|---|---|
| `extrinsicDirac → relativeDirac@τ=1` | exact | `DᵀW_F D == E` |
| `extrinsicFaceDirac → relativeFaceDirac@τ=1` | exact | `D̃ᵀW_V D̃ == Ẽ` |
| `covariantGradient → flatCovariantLaplacian` | exact | `GᵀWG == L3_ambient` (<1e-9, existing anchor) |
| `faceGradient → faceLaplacianGreenGauss` | exact | `G̃ᵀ⋆_F G̃ == K̃` |
| `graphLaplacian √← d0` | exact | `d₀ᵀd₀ == graphL` |
| `relativeDirac@τ=0 √← d0⊗I₄` | exact | on embedded mesh; **not** under intrinsicDelaunay (note) |
| `relativeDirac@τ=½ ×2 == D²` | exact | the squared-Dirac preset identity |
| `intrinsicDirac → laplaceBeltrami` | **principal_part** | scalar part of `D_intᵀW_F D_int == cotanL` (full square = Δ+curvature) |
| `intrinsicFaceDirac → faceLaplacian2Form` | **principal_part** | scalar part only |

Without the qualifier the test would over-assert on the intrinsic links (whose
full square is `Δ + curvature`, not the bare Laplacian).

---

## 6. New operator: extrinsic Weitzenböck Laplacian `Δ₃ + D_N`

The one named-but-missing operator. It is the **ambient (3-comp) sibling** of
the immersion squared Dirac `Δ₄ + D_N` — the same "Laplacian + extrinsic shape
term, un-blended" construction on the ℝ³ bundle instead of the quaternion bundle.

`Δ₃ + D_N` is not literally addable as written (`Δ₃` is 3-comp ambient; `D_N` is
4-comp quaternionic). The construction:

```
extrinsicWeitzenbockLaplacian = flatCovariantLaplacian        (Δ₃, the flat full-frame ambient covariant L)
                              + W_extrinsic                    (zeroth-order extrinsic curvature endomorphism)
```

where `W_extrinsic` is the **ℝ³ / imaginary-quaternion restriction** of the
shape-operator term `D_N` uses — built from the *same* Gauss-map normal
differences as `extrinsicDirac`, assembled as a per-vertex symmetric 3×3
endomorphism (mass-weighted), not the convex blend.

- `bundle: ambient`, `holonomy: extrinsic_curved`, `order: second`, `role: laplacian`,
  `field_type: real`, `domain: vertex`, `singular: none`.
- **Built-in correctness anchor:** its action on the ambient/imaginary part must
  match the corresponding ℝ³ block of the immersion squared Dirac
  (`2·relativeDirac(½)` restricted to the imaginary quaternion components) — a
  cross-bundle consistency check analogous to `GᵀWG == L3_ambient`.

**Open implementation detail** (resolved in the implementation plan, not here):
the exact discrete form of `W_extrinsic` (sign, area weighting, whether the
normal-tilt off-diagonal couples into the normal component). The spec fixes the
*target* (the cross-bundle anchor above); the plan fixes the discretization that
hits it.

---

## 7. Enforcement

Two independent mechanisms, per the locked decision.

1. **Compile-time completeness.** The registry is built from an exhaustive
   `switch (OperatorId)` (or a `static_assert` on the entry-array length keyed by
   `OperatorId` count). Adding an `OperatorId` enumerator without a registry
   entry **fails to compile**. The compiler is the enforcer for the enum-backed
   operators; the variant-only rows (gauge/coupling/τ fan-out, the planned
   Weitzenböck) are covered by a completeness unit test that asserts every
   documented `id` resolves.

2. **Numerical `squares_to` test.** A new `test_operator_registry.cpp` walks
   every entry whose `square` cross-link has `relation: exact`, assembles both
   operators, and asserts matrix equality to `< 1e-9`; for `relation:
   principal_part`, it asserts equality of the scalar/principal block only. The
   `relativeDirac@τ=½ ×2 == D²` and the Weitzenböck cross-bundle anchor are
   included. This makes the metadata's claimed identities *checked*, not trusted.

---

## 8. C++ surface & binding exposure

- **New files:** `include/nxr/operator_registry.h` (the `OperatorVariant` struct,
  the controlled-vocab enums, the `registry()` accessor + `byId(id)` /
  `where(pred)` queries) and `src/operator_registry.cpp` (the table + the
  exhaustive-switch completeness guard).
- **Internal consumers:** `solve::eigenProblemFor` reads `natural_mass` from the
  registry instead of carrying its own switch (single source of truth). A new
  `registry::requireBundle(id, Bundle)` helper is the **LJC-trap guard** — a
  call site that demands an `ambient` differentiator can assert
  `requireBundle(id, Bundle::ambient)` and an `immersion` operator throws
  `Error(InvalidInput)` with a clear hint.
- **MEX:** new command `nxr_compute('operatorInfo', id)` → struct of the metadata
  fields (controlled-vocab values as strings). The existing
  `nxr_compute('operators', h, family, subtype)` dispatch gains the curated `id`s
  as the preferred names (old strings retained as aliases this version; the
  alias table itself lives in the registry as a `legacy_aliases` field).
- **WASM:** `manifold.operatorInfo(id)` mirrors the MEX struct; the existing
  `manifold.operators(family, arg)` accepts the curated ids.
- **N-API addon:** unchanged (does not expose the operators surface today).

---

## 9. Disambiguation decisions (recorded)

The suite layout surfaced five gaps; resolutions:

1. **`OperatorId` too coarse** → stable `OperatorVariant` key resolving to
   `(OperatorId, gauge, coupling, domain, τ)`. §3.
2. **τ-family has no single holonomy** → `holonomy: graded` + `tau` with
   `intrinsic`/`extrinsic`/`squared` presets. §4.
3. **Two "K̃ face Laplacians"** → distinct ids `faceLaplacianGreenGauss`
   (`G̃ᵀ⋆G̃`) vs `faceLaplacian2Form` (`d₁⋆₁⁻¹d₁ᵀ`). §4.
4. **`graphLaplacian` metric-free** → `holonomy: combinatorial`. §2.
5. **`Δ₃ + D_N` ill-typed as written** → defined as `flatCovariantLaplacian +
   W_extrinsic` with a cross-bundle correctness anchor. §6.

---

## 10. Testing

- `test_operator_registry.cpp` — completeness (every `OperatorId` + every
  documented `id` resolves) and the numerical cross-link identities (§7).
- Extend `bindings/mex/test/` with an `operatorInfo` round-trip asserting the
  controlled-vocab strings for a representative operator per bundle.
- The new `extrinsicWeitzenbockLaplacian` gets its own assembly test against the
  cross-bundle anchor (§6).
- No regression to existing operator outputs — the registry is additive; the
  only behavioral change is `eigenProblemFor` sourcing `natural_mass` from the
  registry (verified byte-identical to its prior switch).

---

## 11. Deferred

- Frame-coupled α-spectral / shell operator (unchanged deferral).
- Boundary treatment (infinite-potential-well) for the face Diracs and the
  Weitzenböck (closed-mesh v1).
- MATLAB `handle`-class wrapper.
- Routing the intrinsic *data* facet through `operatorGeometry()` under
  intrinsicDelaunay (Phase 3) — affects the `relativeDirac@τ=0 √← d0⊗I₄` note.
