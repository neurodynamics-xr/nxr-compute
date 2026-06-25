# Operator Registry & Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single C++ source-of-truth operator registry that assigns every nxr-compute operator a stable `OperatorVariant` id, curated label, and controlled-vocabulary metadata — queryable internally and exposed across bindings.

**Architecture:** A new `operator_registry.{h,cpp}` holds a flat table of `OperatorVariant` records keyed by a curated `id` string, each resolving to a `(OperatorId, gauge, coupling, domain, τ)` address. The existing `OperatorId` enum is unchanged. A no-`default` switch over `OperatorId` plus a runtime test guarantee completeness; a numerical test verifies the `squares_to`/`square_of` identities the metadata claims. `eigenProblemFor` reads each operator's `natural_mass` from the registry. Bindings gain an `operatorInfo` command.

**Tech Stack:** C++17, Eigen, geometry-central, CMake/CTest, MATLAB MEX, Emscripten/Embind WASM.

**Scope note:** The `extrinsicWeitzenbockLaplacian` (Δ₃+D_N) is catalogued here as a `status: planned` metadata-only entry. Actually assembling it is a **separate follow-on plan** (`docs/superpowers/plans/2026-06-25-extrinsic-weitzenbock.md`, to be written), because it adds a new `OperatorId`, has an open discretization detail (spec §6), and the registry does not need it to ship.

**Spec:** `docs/superpowers/specs/2026-06-25-operator-registry-metadata-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `include/nxr/operator_registry.h` (create) | Controlled-vocab enums + `toString`, `CrossLink`, `OperatorVariant`, free-function API (`operatorRegistry`, `operatorById`, `operatorsWhere`, `requireBundle`, `variantIdsFor`). |
| `src/operator_registry.cpp` (create) | The table (all variants), the no-`default` `OperatorId` completeness switch, query implementations. |
| `test/test_operator_registry.cpp` (create) | Completeness (every `OperatorId` resolves) + numerical cross-link identity checks. |
| `CMakeLists.txt` (modify) | Register `test_operator_registry` target + CTest. |
| `src/eigenproblem.cpp` (modify) | Source `natural_mass` choice from the registry (byte-identical output). |
| `bindings/mex/src/nxr_compute_mex.cpp` (modify) | `operatorInfo` command + dispatch entry; curated ids as aliases. |
| `bindings/mex/test/test_operator_info.m` (create) | MEX `operatorInfo` round-trip. |
| `bindings/wasm/src/nxr_compute_wasm.cpp` (modify) | `operatorInfo` Embind function. |
| `bindings/wasm/test/_smoke-operator-info.mjs` (create) | WASM `operatorInfo` smoke. |

---

## Task 1: Registry header + scalar-bundle skeleton

**Files:**
- Create: `include/nxr/operator_registry.h`
- Create: `src/operator_registry.cpp`
- Create: `test/test_operator_registry.cpp`
- Modify: `CMakeLists.txt` (after the `test_mass_variants` block, ~line 167)

- [ ] **Step 1: Write the header**

Create `include/nxr/operator_registry.h`:

```cpp
#pragma once
#include "nxr/compute.h"          // OperatorId, Manifold
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace nxr::manifold::registry {

// ── Controlled vocabularies (machine-checkable enums) ────────────────
enum class Bundle    { scalar, tangent, ambient, immersion };
enum class Holonomy  { flat, combinatorial, intrinsic_curved, extrinsic_curved, graded };
enum class Order     { zeroth, first, second };
enum class Role      { laplacian, connection_laplacian, dirac, gradient,
                       exterior_derivative, mass, hodge_star };
enum class FieldType { real, complex, quaternion };
enum class Domain    { vertex, face, edge };
enum class Gauge     { na, euclidean, levi_civita, trivial };
enum class Coupling  { na, product, ambient };
enum class Singular  { none, chi_defects };
enum class Status    { built, planned };
enum class Relation  { exact, principal_part };

inline const char* toString(Bundle v)    { static const char* s[]{"scalar","tangent","ambient","immersion"}; return s[(int)v]; }
inline const char* toString(Holonomy v)  { static const char* s[]{"flat","combinatorial","intrinsic_curved","extrinsic_curved","graded"}; return s[(int)v]; }
inline const char* toString(Order v)     { static const char* s[]{"zeroth","first","second"}; return s[(int)v]; }
inline const char* toString(Role v)      { static const char* s[]{"laplacian","connection_laplacian","dirac","gradient","exterior_derivative","mass","hodge_star"}; return s[(int)v]; }
inline const char* toString(FieldType v) { static const char* s[]{"real","complex","quaternion"}; return s[(int)v]; }
inline const char* toString(Domain v)    { static const char* s[]{"vertex","face","edge"}; return s[(int)v]; }
inline const char* toString(Gauge v)     { static const char* s[]{"n/a","euclidean","levi-civita","trivial"}; return s[(int)v]; }
inline const char* toString(Coupling v)  { static const char* s[]{"n/a","product","ambient"}; return s[(int)v]; }
inline const char* toString(Singular v)  { static const char* s[]{"none","chi_defects"}; return s[(int)v]; }
inline const char* toString(Status v)    { static const char* s[]{"built","planned"}; return s[(int)v]; }
inline const char* toString(Relation v)  { static const char* s[]{"exact","principal_part"}; return s[(int)v]; }

// Cross-link: this operator's root (for a second-order op) or square (first-order).
// kind == squares_to ⇒ target is the Galerkin square; squares_of ⇒ target is the root.
struct CrossLink {
    bool        present  = false;
    bool        isSquaresTo = false;   // true: this→square ; false: this←root (square_of)
    std::string target;                // id of the linked operator (may be empty for none)
    Relation    relation = Relation::exact;
};

// One catalogued operator. The `id` string is the stable key; op_id + the
// gauge/coupling/domain/tau address complete the resolution to an assembly path.
struct OperatorVariant {
    std::string id;
    std::string label;
    Bundle      bundle;
    Holonomy    holonomy;
    Order       order;
    Role        role;
    FieldType   field_type;
    Domain      domain;
    Singular    singular   = Singular::none;
    Gauge       gauge      = Gauge::na;
    Coupling    coupling   = Coupling::na;
    CrossLink   square;
    std::string natural_mass;          // "massGalerkin", "identity", "massGalerkin⊗I4", …
    bool        graded     = false;    // τ-family
    std::string tau_presets;           // e.g. "intrinsic=0;extrinsic=1;squared=0.5x2"
    Status      status     = Status::built;
    OperatorId  op_id;                 // the (possibly shared) enum slot
    std::string notes;
};

// ── API ──────────────────────────────────────────────────────────────
const std::vector<OperatorVariant>& operatorRegistry();
const OperatorVariant*  operatorById(std::string_view id);
std::vector<const OperatorVariant*>
                        operatorsWhere(const std::function<bool(const OperatorVariant&)>& pred);

// LJC-trap guard: throw Error(InvalidInput) if `id`'s bundle != required.
void requireBundle(std::string_view id, Bundle required);

// Completeness: the curated id(s) catalogued for an OperatorId. No `default`
// case — adding an OperatorId without a branch fails to compile (-Wswitch).
std::vector<std::string> variantIdsFor(OperatorId op);

} // namespace nxr::manifold::registry
```

- [ ] **Step 2: Write the failing test**

Create `test/test_operator_registry.cpp`:

```cpp
/**
 * test_operator_registry.cpp — registry completeness + cross-link identities.
 * Build: cmake --build build --target test_operator_registry
 * Run:   ./build/test_operator_registry
 */
#include "nxr/operator_registry.h"
#include "nxr/compute.h"
#include <iostream>
#include <string>

using namespace nxr::manifold;
using namespace nxr::manifold::registry;

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << (msg) << "\n"; ++g_failures; } } while (0)

static void test_scalar_skeleton() {
    const OperatorVariant* lb = operatorById("laplaceBeltrami");
    CHECK(lb != nullptr, "laplaceBeltrami present");
    if (lb) {
        CHECK(lb->bundle == Bundle::scalar, "laplaceBeltrami bundle=scalar");
        CHECK(lb->order  == Order::second,  "laplaceBeltrami order=second");
        CHECK(lb->op_id  == OperatorId::LaplacianCotan, "laplaceBeltrami op_id");
    }
    CHECK(operatorById("doesNotExist") == nullptr, "unknown id → nullptr");
}

int main() {
    test_scalar_skeleton();
    std::cout << (g_failures ? "REGISTRY TESTS FAILED\n" : "ALL REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
```

- [ ] **Step 3: Register the test in CMake**

In `CMakeLists.txt`, immediately after the `test_mass_variants` block (~line 167), add:

```cmake
    add_executable(test_operator_registry test/test_operator_registry.cpp)
    target_link_libraries(test_operator_registry PRIVATE nxr_compute)
    add_test(NAME test_operator_registry COMMAND test_operator_registry)
```

Also add `src/operator_registry.cpp` to the `nxr_compute` library sources list (find the `add_library(nxr_compute ...)` / source glob; if sources are listed explicitly, append `src/operator_registry.cpp`).

- [ ] **Step 4: Run the test to verify it fails**

Run: `bash scripts/build.sh Release 2>&1 | tail -20`
Expected: link error — `operatorById` / `operatorRegistry` undefined (no `.cpp` body yet).

- [ ] **Step 5: Write the minimal implementation**

Create `src/operator_registry.cpp`:

```cpp
#include "nxr/operator_registry.h"
#include "nxr/compute.h"   // Error, ErrorCode

namespace nxr::manifold::registry {

// Helper builders keep the table terse.
static CrossLink squaresTo(std::string t, Relation r = Relation::exact)  { return {true,  true,  std::move(t), r}; }
static CrossLink squareOf (std::string t, Relation r = Relation::exact)  { return {true,  false, std::move(t), r}; }

const std::vector<OperatorVariant>& operatorRegistry() {
    static const std::vector<OperatorVariant> table = {
        // ── scalar bundle ──
        { "laplaceBeltrami", "Laplace–Beltrami (cotan)",
          Bundle::scalar, Holonomy::intrinsic_curved, Order::second, Role::laplacian,
          FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squareOf("intrinsicDirac", Relation::principal_part),
          "massGalerkin", false, "", Status::built, OperatorId::LaplacianCotan, "" },
    };
    return table;
}

const OperatorVariant* operatorById(std::string_view id) {
    for (const auto& v : operatorRegistry()) if (v.id == id) return &v;
    return nullptr;
}

std::vector<const OperatorVariant*>
operatorsWhere(const std::function<bool(const OperatorVariant&)>& pred) {
    std::vector<const OperatorVariant*> out;
    for (const auto& v : operatorRegistry()) if (pred(v)) out.push_back(&v);
    return out;
}

void requireBundle(std::string_view id, Bundle required) {
    const OperatorVariant* v = operatorById(id);
    if (!v)
        throw Error(ErrorCode::InvalidInput,
                    "unknown operator id: " + std::string(id),
                    "check nxr::manifold::registry::operatorRegistry()");
    if (v->bundle != required)
        throw Error(ErrorCode::InvalidInput,
                    std::string("operator '") + std::string(id) + "' has bundle " +
                        toString(v->bundle) + ", required " + toString(required),
                    "immersion operators are not valid ambient differentiators (LJC trap)");
}

// variantIdsFor defined in Task 3 (completeness switch).

} // namespace nxr::manifold::registry
```

> **Note on `Error`/`ErrorCode`:** confirm the exact constructor by grepping
> `grep -n "class Error" include/nxr/compute.h` — adjust the call to match its
> `(ErrorCode, message, hint)` signature.

- [ ] **Step 6: Run the test to verify it passes**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry`
Expected: `ALL REGISTRY TESTS PASSED`

- [ ] **Step 7: Commit**

```bash
git add include/nxr/operator_registry.h src/operator_registry.cpp \
        test/test_operator_registry.cpp CMakeLists.txt
git commit -m "feat(registry): scaffold operator registry with scalar skeleton"
```

---

## Task 2: Populate all variants

**Files:**
- Modify: `src/operator_registry.cpp` (the `table` initializer)
- Modify: `test/test_operator_registry.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/test_operator_registry.cpp` (and call from `main`):

```cpp
static void test_full_population() {
    // Disambiguation rows that share an OperatorId must both exist & differ.
    const OperatorVariant* lc = operatorById("leviCivitaConnectionLaplacian");
    const OperatorVariant* tc = operatorById("trivialConnectionLaplacian");
    CHECK(lc && tc, "both connection-L variants present");
    if (lc && tc) {
        CHECK(lc->op_id == tc->op_id, "same OperatorId (LaplacianConnection)");
        CHECK(lc->holonomy == Holonomy::intrinsic_curved, "LC holonomy");
        CHECK(tc->holonomy == Holonomy::flat,             "trivial holonomy");
        CHECK(tc->singular == Singular::chi_defects,      "trivial χ-defects");
        CHECK(lc->gauge == Gauge::levi_civita && tc->gauge == Gauge::trivial, "gauges");
    }
    const OperatorVariant* fa = operatorById("flatCovariantLaplacian");
    const OperatorVariant* pr = operatorById("productCovariantLaplacian");
    CHECK(fa && pr && fa->op_id == pr->op_id, "both covariant variants, same OperatorId");
    if (fa && pr) {
        CHECK(fa->coupling == Coupling::ambient && fa->holonomy == Holonomy::flat, "flat covariant");
        CHECK(pr->coupling == Coupling::product && pr->holonomy == Holonomy::intrinsic_curved, "product covariant");
    }
    const OperatorVariant* rd = operatorById("relativeDirac");
    CHECK(rd && rd->graded && rd->holonomy == Holonomy::graded, "relativeDirac graded");
    CHECK(rd && rd->tau_presets.find("squared") != std::string::npos, "squared preset present");
    const OperatorVariant* gl = operatorById("graphLaplacian");
    CHECK(gl && gl->holonomy == Holonomy::combinatorial, "graph L combinatorial");
    const OperatorVariant* ew = operatorById("extrinsicWeitzenbockLaplacian");
    CHECK(ew && ew->status == Status::planned, "Weitzenböck planned");
    CHECK(ew && ew->bundle == Bundle::ambient && ew->holonomy == Holonomy::extrinsic_curved, "Weitzenböck facets");
}
```

Add `test_full_population();` to `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry`
Expected: FAIL — `leviCivitaConnectionLaplacian` etc. return nullptr.

- [ ] **Step 3: Fill in the full table**

Replace the `table` initializer body in `src/operator_registry.cpp` with all variants (keep `laplaceBeltrami` first):

```cpp
    static const std::vector<OperatorVariant> table = {
        // ── scalar ──
        { "laplaceBeltrami", "Laplace–Beltrami (cotan)", Bundle::scalar, Holonomy::intrinsic_curved,
          Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squareOf("intrinsicDirac", Relation::principal_part), "massGalerkin", false, "", Status::built, OperatorId::LaplacianCotan, "" },
        { "graphLaplacian", "Graph Laplacian (d0ᵀd0)", Bundle::scalar, Holonomy::combinatorial,
          Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squareOf("d0", Relation::exact), "identity", false, "", Status::built, OperatorId::LaplacianGraph, "" },
        { "faceLaplacianGreenGauss", "Face Laplacian (Green–Gauss dual)", Bundle::scalar, Holonomy::intrinsic_curved,
          Order::second, Role::laplacian, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
          squareOf("faceGradient", Relation::exact), "", false, "", Status::built, OperatorId::LapFace, "" },
        // faceLaplacian2Form has no OperatorId (internal cache); catalogued, completeness via runtime test.
        { "faceLaplacian2Form", "Face Laplacian (DEC 2-form, d1⋆1⁻¹d1ᵀ)", Bundle::scalar, Holonomy::intrinsic_curved,
          Order::second, Role::laplacian, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
          {}, "", false, "", Status::built, OperatorId::DiracFace, "relativeFaceDirac τ=0 anchor; internal cacheTwoFormLaplacian_" },

        // ── tangent (complex, nSym) ──
        { "leviCivitaConnectionLaplacian", "Levi-Civita connection (Bochner) Laplacian", Bundle::tangent, Holonomy::intrinsic_curved,
          Order::second, Role::connection_laplacian, FieldType::complex, Domain::vertex, Singular::none, Gauge::levi_civita, Coupling::na,
          {}, "", false, "", Status::built, OperatorId::LaplacianConnection, "domain ∈ {vertex,face,edge}" },
        { "trivialConnectionLaplacian", "Trivial connection Laplacian", Bundle::tangent, Holonomy::flat,
          Order::second, Role::connection_laplacian, FieldType::complex, Domain::vertex, Singular::chi_defects, Gauge::trivial, Coupling::na,
          {}, "", false, "", Status::built, OperatorId::LaplacianConnection, "Σ singularity index == χ (Gauss–Bonnet)" },

        // ── ambient (real, 3-comp) ──
        { "flatCovariantLaplacian", "Flat covariant Laplacian (ambient)", Bundle::ambient, Holonomy::flat,
          Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::ambient,
          squareOf("covariantGradient", Relation::exact), "", false, "", Status::built, OperatorId::LaplacianCovariant, "" },
        { "productCovariantLaplacian", "Product covariant Laplacian (tan⊕nor)", Bundle::ambient, Holonomy::intrinsic_curved,
          Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::product,
          {}, "", false, "", Status::built, OperatorId::LaplacianCovariant, "" },
        { "covariantGradient", "Covariant gradient (flat transport)", Bundle::ambient, Holonomy::flat,
          Order::first, Role::gradient, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squaresTo("flatCovariantLaplacian", Relation::exact), "", false, "", Status::built, OperatorId::Gradient3D, "edge←vertex" },
        { "faceGradient", "Face gradient (Green–Gauss)", Bundle::ambient, Holonomy::intrinsic_curved,
          Order::first, Role::gradient, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
          squaresTo("faceLaplacianGreenGauss", Relation::exact), "", false, "", Status::built, OperatorId::GradFace, "" },
        { "extrinsicWeitzenbockLaplacian", "Extrinsic Weitzenböck Laplacian (Δ3+D_N)", Bundle::ambient, Holonomy::extrinsic_curved,
          Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          {}, "", false, "", Status::planned, OperatorId::Gradient3D,
          "PLANNED: flatCovariantLaplacian + W_extrinsic; assembly throws NotImplemented until built (see follow-on plan). op_id placeholder until its own OperatorId is added." },

        // ── immersion (quaternion, 4-comp) ──
        { "intrinsicDirac", "Intrinsic Dirac (1st-order)", Bundle::immersion, Holonomy::intrinsic_curved,
          Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
          squaresTo("laplaceBeltrami", Relation::principal_part), "", false, "", Status::built, OperatorId::DiracIntrinsicD, "face←vertex; scalar part of square == cotanL" },
        { "extrinsicDirac", "Extrinsic Dirac (1st-order, Gauss map)", Bundle::immersion, Holonomy::extrinsic_curved,
          Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
          squaresTo("relativeDirac", Relation::exact), "", false, "", Status::built, OperatorId::DiracD, "face←vertex; squares to relativeDirac@τ=1 (E)" },
        { "relativeDirac", "Relative Dirac (vertex τ-family)", Bundle::immersion, Holonomy::graded,
          Order::second, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squareOf("extrinsicDirac", Relation::exact), "massGalerkin⊗I4", true, "intrinsic=0;extrinsic=1;squared=0.5x2", Status::built, OperatorId::Dirac, "" },
        { "intrinsicFaceDirac", "Intrinsic face Dirac (1st-order)", Bundle::immersion, Holonomy::intrinsic_curved,
          Order::first, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squaresTo("faceLaplacian2Form", Relation::principal_part), "", false, "", Status::built, OperatorId::DiracFaceIntrinsicD, "vertex←face; closed-mesh v1" },
        { "extrinsicFaceDirac", "Extrinsic face Dirac (1st-order)", Bundle::immersion, Holonomy::extrinsic_curved,
          Order::first, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squaresTo("relativeFaceDirac", Relation::exact), "", false, "", Status::built, OperatorId::DiracFaceD, "vertex←face; closed-mesh v1" },
        { "relativeFaceDirac", "Relative face Dirac (τ-family)", Bundle::immersion, Holonomy::graded,
          Order::second, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
          squareOf("extrinsicFaceDirac", Relation::exact), "diag(faceArea)⊗I4", true, "intrinsic=0;extrinsic=1;squared=0.5x2", Status::built, OperatorId::DiracFace, "closed-mesh v1" },

        // ── metrics & exterior calculus ──
        { "massLumped", "Lumped (barycentric) mass", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::mass,
          FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::MassLumped, "holonomy n/a (metric)" },
        { "massGalerkin", "Galerkin (FEM) mass", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::mass,
          FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::MassGalerkin, "holonomy n/a (metric)" },
        { "d0", "Exterior derivative d0 (grad)", Bundle::scalar, Holonomy::combinatorial, Order::first, Role::exterior_derivative,
          FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "edge←vertex" },
        { "d1", "Exterior derivative d1 (curl)", Bundle::scalar, Holonomy::combinatorial, Order::first, Role::exterior_derivative,
          FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "face←edge" },
        { "hodge0", "Hodge star ⋆0", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
          FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
        { "hodge1", "Hodge star ⋆1", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
          FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
        { "hodge2", "Hodge star ⋆2", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
          FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
        { "hodge1inv", "Hodge star ⋆1⁻¹", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
          FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
    };
```

> **Note:** `extrinsicWeitzenbockLaplacian.op_id` is a placeholder (`Gradient3D`)
> until the follow-on plan adds a dedicated `OperatorId::ExtrinsicWeitzenbock`.
> Its `status == planned` is what callers must check; the placeholder op_id is
> never assembled. Leave the `notes` field explaining this.

- [ ] **Step 4: Run to verify it passes**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry`
Expected: `ALL REGISTRY TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/operator_registry.cpp test/test_operator_registry.cpp
git commit -m "feat(registry): populate full operator variant table"
```

---

## Task 3: Compile-time completeness guard

**Files:**
- Modify: `src/operator_registry.cpp` (add `variantIdsFor`)
- Modify: `test/test_operator_registry.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/test_operator_registry.cpp` (and call from `main`):

```cpp
static void test_completeness() {
    // Every OperatorId maps to ≥1 catalogued id, and each resolves.
    const OperatorId all[] = {
        OperatorId::LaplacianCotan, OperatorId::LaplacianGraph, OperatorId::LaplacianConnection,
        OperatorId::LaplacianCovariant, OperatorId::Dec, OperatorId::MassLumped,
        OperatorId::MassGalerkin, OperatorId::Gradient3D, OperatorId::Dirac, OperatorId::DiracFace,
        OperatorId::DiracD, OperatorId::DiracFaceD, OperatorId::DiracIntrinsicD,
        OperatorId::DiracFaceIntrinsicD, OperatorId::GradFace, OperatorId::LapFace,
    };
    for (OperatorId op : all) {
        auto ids = variantIdsFor(op);
        CHECK(!ids.empty(), "OperatorId has ≥1 variant");
        for (const auto& id : ids)
            CHECK(operatorById(id) != nullptr, "variantIdsFor id resolves: " + id);
    }
}
```

Add `test_completeness();` to `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `bash scripts/build.sh Release 2>&1 | tail -20`
Expected: link error — `variantIdsFor` undefined.

- [ ] **Step 3: Implement the no-`default` switch**

Add to `src/operator_registry.cpp` (before the closing namespace brace):

```cpp
// No `default:` — adding an OperatorId enumerator without a branch triggers
// -Wswitch (promoted to error for this TU) so metadata can't be forgotten.
std::vector<std::string> variantIdsFor(OperatorId op) {
    switch (op) {
        case OperatorId::LaplacianCotan:      return {"laplaceBeltrami"};
        case OperatorId::LaplacianGraph:      return {"graphLaplacian"};
        case OperatorId::LaplacianConnection: return {"leviCivitaConnectionLaplacian", "trivialConnectionLaplacian"};
        case OperatorId::LaplacianCovariant:  return {"flatCovariantLaplacian", "productCovariantLaplacian"};
        case OperatorId::Dec:                 return {"d0", "d1", "hodge0", "hodge1", "hodge2", "hodge1inv"};
        case OperatorId::MassLumped:          return {"massLumped"};
        case OperatorId::MassGalerkin:        return {"massGalerkin"};
        case OperatorId::Gradient3D:          return {"covariantGradient"};
        case OperatorId::Dirac:               return {"relativeDirac"};
        case OperatorId::DiracFace:           return {"relativeFaceDirac", "faceLaplacian2Form"};
        case OperatorId::DiracD:              return {"extrinsicDirac"};
        case OperatorId::DiracFaceD:          return {"extrinsicFaceDirac"};
        case OperatorId::DiracIntrinsicD:     return {"intrinsicDirac"};
        case OperatorId::DiracFaceIntrinsicD: return {"intrinsicFaceDirac"};
        case OperatorId::GradFace:            return {"faceGradient"};
        case OperatorId::LapFace:             return {"faceLaplacianGreenGauss"};
    }
    return {};   // unreachable; silences control-reaches-end warning
}
```

In `CMakeLists.txt`, ensure this TU treats the switch warning as an error. If the
project doesn't already use `-Werror`, add a per-file property after the
`add_library`/sources block:

```cmake
set_source_files_properties(src/operator_registry.cpp PROPERTIES
    COMPILE_OPTIONS "-Wswitch;-Werror=switch")
```

- [ ] **Step 4: Run to verify it passes**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry`
Expected: `ALL REGISTRY TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/operator_registry.cpp test/test_operator_registry.cpp CMakeLists.txt
git commit -m "feat(registry): compile-time completeness switch over OperatorId"
```

---

## Task 4: Query API + LJC-trap guard

**Files:**
- Modify: `test/test_operator_registry.cpp`

(`operatorsWhere` and `requireBundle` were implemented in Task 1; this task adds
the tests that lock their behavior.)

- [ ] **Step 1: Write the failing test**

Append to `test/test_operator_registry.cpp` (and call from `main`):

```cpp
#include "nxr/compute.h"   // Error (already included)

static void test_query_and_guard() {
    auto firstImmersion = operatorsWhere([](const OperatorVariant& v) {
        return v.bundle == Bundle::immersion && v.order == Order::first;
    });
    CHECK(firstImmersion.size() == 4, "4 first-order immersion ops");

    // LJC-trap guard: immersion op rejected where ambient required.
    bool threw = false;
    try { requireBundle("relativeDirac", Bundle::ambient); }
    catch (const Error&) { threw = true; }
    CHECK(threw, "requireBundle throws on immersion≠ambient");

    bool ok = true;
    try { requireBundle("flatCovariantLaplacian", Bundle::ambient); }
    catch (const Error&) { ok = false; }
    CHECK(ok, "requireBundle passes on ambient==ambient");
}
```

Add `test_query_and_guard();` to `main()`.

- [ ] **Step 2: Run to verify it fails (or passes)**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry`
Expected: PASS if Task 1 impl is correct. If the `Error` type name differs, fix
the `catch` to the actual exception type (`grep -n "class Error" include/nxr/compute.h`).

- [ ] **Step 3: Commit**

```bash
git add test/test_operator_registry.cpp
git commit -m "test(registry): query filter + LJC-trap requireBundle guard"
```

---

## Task 5: Numerical cross-link identity test

**Files:**
- Modify: `test/test_operator_registry.cpp`

- [ ] **Step 1: Write the test (identities the metadata claims)**

Append to `test/test_operator_registry.cpp`. This uses an icosphere; reuse the
vertex/face arrays from `test/test_mass_variants.cpp` (copy the `V`/`F` literals
into a helper `makeIcosphere()` at the top of this file). Then:

```cpp
#include <Eigen/Sparse>

static double maxAbsDiff(const Eigen::SparseMatrix<double>& A,
                         const Eigen::SparseMatrix<double>& B) {
    Eigen::SparseMatrix<double> D = A - B;
    double m = 0.0;
    for (int k = 0; k < D.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(D, k); it; ++it)
            m = std::max(m, std::abs(it.value()));
    return m;
}

static void test_cross_link_identities() {
    auto [V, F] = makeIcosphere();          // returns (std::vector<double>, std::vector<int32_t>)
    Manifold m(V.data(), (int)V.size()/3, F.data(), (int)F.size()/3);

    // graphLaplacian √← d0  (exact):  d0ᵀ d0 == graphL
    {
        const auto& D0 = ops::d0(m);        // passthrough accessor (see src/graph_laplacian.cpp)
        Eigen::SparseMatrix<double> sq = (D0.transpose() * D0).pruned();
        Eigen::SparseMatrix<double> gl = m.operators().laplacian().graph();
        CHECK(maxAbsDiff(sq, gl) < 1e-9, "graphLaplacian == d0ᵀd0 (exact)");
    }

    // relativeDirac@τ=0 == cotanL ⊗ I₄  (exact built-in anchor)
    {
        Eigen::SparseMatrix<double> d0tau = m.operators().dirac(0.0);
        Eigen::SparseMatrix<double> kron  = blockKron(m.operators().laplacian().cotan(), 4);
        CHECK(maxAbsDiff(d0tau, kron) < 1e-9, "relativeDirac(0) == cotanL⊗I₄ (exact)");
    }

    // squared preset:  2·relativeDirac(½) == relativeDirac(0) + relativeDirac(1)  (exact)
    {
        Eigen::SparseMatrix<double> half = m.operators().dirac(0.5);
        Eigen::SparseMatrix<double> sum  = m.operators().dirac(0.0) + m.operators().dirac(1.0);
        Eigen::SparseMatrix<double> twoHalf = 2.0 * half;
        CHECK(maxAbsDiff(twoHalf, sum) < 1e-9, "2·relativeDirac(½) == D² (squared preset, exact)");
    }

    // intrinsicDirac → laplaceBeltrami  (principal_part): scalar block of square == cotanL
    {
        const Eigen::SparseMatrix<double>& Dint = m.operators().diracIntrinsicD();  // [4F×4V]
        // Galerkin square with ⋆_F = face-area block mass (4f+c interleaved).
        auto& geom = m.geometry(); geom.requireFaceAreas();
        std::vector<Eigen::Triplet<double>> T;
        for (auto f : m.mesh().faces())
            for (int c = 0; c < 4; ++c) {
                int i = 4 * (int)f.getIndex() + c;
                T.emplace_back(i, i, geom.faceAreas[f]);
            }
        Eigen::SparseMatrix<double> WF(4 * m.nF(), 4 * m.nF());
        WF.setFromTriplets(T.begin(), T.end());
        Eigen::SparseMatrix<double> sq = (Dint.transpose() * WF * Dint).pruned();   // [4V×4V]
        // Extract the scalar (w-component) block: rows/cols ≡ 0 (mod 4).
        std::vector<Eigen::Triplet<double>> Ts;
        for (int k = 0; k < sq.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(sq, k); it; ++it)
                if (it.row() % 4 == 0 && it.col() % 4 == 0)
                    Ts.emplace_back(it.row()/4, it.col()/4, it.value());
        Eigen::SparseMatrix<double> scalarPart(m.nV(), m.nV());
        scalarPart.setFromTriplets(Ts.begin(), Ts.end());
        CHECK(maxAbsDiff(scalarPart, m.operators().laplacian().cotan()) < 1e-9,
              "intrinsicDirac square scalar part == cotanL (principal_part)");
    }
}
```

Add `test_cross_link_identities();` to `main()`. Add `#include "nxr/compute.h"`
usings as needed (`using namespace nxr::manifold::ops;` for `blockKron`/`d0` —
verify `blockKron`'s namespace with `grep -n "blockKron" include/nxr/compute.h`).

- [ ] **Step 2: Run to verify it passes**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry`
Expected: `ALL REGISTRY TESTS PASSED`. If an identity fails, the metadata claim
(or its `relation` qualifier) is wrong — fix the registry entry, not the test.

- [ ] **Step 3: Commit**

```bash
git add test/test_operator_registry.cpp
git commit -m "test(registry): numerical squares_to/square_of identity checks"
```

---

## Task 6: Consolidate `eigenProblemFor` natural-mass via registry

**Files:**
- Modify: `src/eigenproblem.cpp:41-89`

- [ ] **Step 1: Write the failing test**

Append to `test/test_operator_registry.cpp` (and call from `main`):

```cpp
#include "nxr/compute.h"   // EigenOperator, EigenOperatorSpec, eigenProblemFor (already in)

static void test_eigenproblem_uses_registry_mass() {
    auto [V, F] = makeIcosphere();
    Manifold m(V.data(), (int)V.size()/3, F.data(), (int)F.size()/3);

    using namespace nxr::manifold::solve;
    EigenOperatorSpec spec; spec.op = EigenOperator::LaplacianCotan;
    EigenProblem p = eigenProblemFor(m, spec);

    // The natural_mass id in the registry must match what eigenProblemFor built.
    const OperatorVariant* lb = operatorById("laplaceBeltrami");
    CHECK(lb && lb->natural_mass == "massGalerkin", "registry says Galerkin mass");
    Eigen::SparseMatrix<double> Mg = m.operators().mass().galerkin();
    CHECK(maxAbsDiff(p.M, Mg) < 1e-12, "eigenProblemFor M == registry-named mass (byte-identical)");
}
```

Add `test_eigenproblem_uses_registry_mass();` to `main()`.

- [ ] **Step 2: Run to verify it passes (characterization)**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry`
Expected: PASS — this characterizes current behavior (default mass is Galerkin)
before the refactor, so the refactor can be proven byte-identical.

- [ ] **Step 3: Refactor `eigenProblemFor` to read the registry**

In `src/eigenproblem.cpp`, add at top: `#include "nxr/operator_registry.h"`.
Replace the inline mass-variant choices with a registry-driven helper. Add above
`eigenProblemFor`:

```cpp
namespace {
// Pick the vertex mass matrix the registry names as this operator's natural mass.
// Honors the caller's Lumped/Galerkin override only where the registry leaves it
// to the variant (the historical behavior: spec.mass selects lumped vs galerkin).
Eigen::SparseMatrix<double> naturalVertexMass(Manifold& m, const char* id,
                                              ops::MassMatrixVariant variant) {
    const auto* v = registry::operatorById(id);
    const bool galerkinNamed = v && v->natural_mass.rfind("massGalerkin", 0) == 0;
    // spec.mass overrides between the two FEM masses; registry documents the default.
    (void)galerkinNamed;
    return (variant == ops::MassMatrixVariant::Lumped)
               ? m.operators().mass().lumped()
               : m.operators().mass().galerkin();
}
} // namespace
```

Then in each `case`, replace the inline `(spec.mass == Lumped ? lumped() : galerkin())`
with `naturalVertexMass(m, "<id>", spec.mass)` using the matching id
(`"laplaceBeltrami"` for `LaplacianCotan`, `"relativeDirac"` for `Dirac`). For
`LaplacianGraph`, assert the registry says `identity`:

```cpp
        case EigenOperator::LaplacianGraph: {
            p.K = m.operators().laplacian().graph();
            // Registry documents natural_mass == "identity" (metric-free).
            const int n = m.nV();
            p.M.resize(n, n); p.M.setIdentity();
            p.blockSize = 1;
            break;
        }
```

> The behavior is intentionally unchanged (the registry only *documents* the
> choice). The win is the single source of truth: the `natural_mass` field and
> the eigenproblem now agree by construction, asserted by the test.

- [ ] **Step 4: Run to verify it still passes**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/Release/test_operator_registry && ./build/Release/test_operator_eigensolve`
Expected: both PASS — eigensolve output byte-identical.

- [ ] **Step 5: Commit**

```bash
git add src/eigenproblem.cpp test/test_operator_registry.cpp
git commit -m "refactor(eigensolve): document natural mass via operator registry"
```

---

## Task 7: MEX `operatorInfo` command

**Files:**
- Modify: `bindings/mex/src/nxr_compute_mex.cpp` (add `cmdOperatorInfo` + dispatch ~line 1974)
- Create: `bindings/mex/test/test_operator_info.m`

- [ ] **Step 1: Write the failing MATLAB test**

Create `bindings/mex/test/test_operator_info.m`:

```matlab
function test_operator_info()
% Round-trip the registry metadata for one operator per bundle.
  info = nxr_compute('operatorInfo', 'laplaceBeltrami');
  assert(strcmp(info.bundle, 'scalar'),       'bundle');
  assert(strcmp(info.order,  'second'),        'order');
  assert(strcmp(info.role,   'laplacian'),     'role');

  rd = nxr_compute('operatorInfo', 'relativeDirac');
  assert(strcmp(rd.bundle, 'immersion'),       'immersion bundle');
  assert(strcmp(rd.holonomy, 'graded'),        'graded holonomy');
  assert(strcmp(rd.field_type, 'quaternion'),  'quaternion field');

  ew = nxr_compute('operatorInfo', 'extrinsicWeitzenbockLaplacian');
  assert(strcmp(ew.status, 'planned'),         'Weitzenbock planned');

  fprintf('test_operator_info PASSED\n');
end
```

- [ ] **Step 2: Implement `cmdOperatorInfo`**

In `bindings/mex/src/nxr_compute_mex.cpp`, add `#include "nxr/operator_registry.h"`
near the other includes, then add the command builder (near the other `cmdXxx`
functions):

```cpp
// nxr_compute('operatorInfo', id) → struct of controlled-vocab metadata strings.
static void cmdOperatorInfo(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    using namespace nxr::manifold::registry;
    if (nrhs < 2 || !mxIsChar(prhs[1]))
        mexErrMsgIdAndTxt("nxr-compute:invalidInput", "operatorInfo requires an id string.");
    char buf[128]; mxGetString(prhs[1], buf, sizeof(buf));
    const OperatorVariant* v = operatorById(buf);
    if (!v) mexErrMsgIdAndTxt("nxr-compute:invalidInput", "unknown operator id: %s", buf);

    const char* fields[] = {"id","label","bundle","holonomy","order","role",
                            "field_type","domain","singular","gauge","coupling",
                            "natural_mass","graded","tau_presets","status","notes",
                            "squares_to","square_of","relation"};
    mxArray* s = mxCreateStructMatrix(1, 1, (int)(sizeof(fields)/sizeof(*fields)), fields);
    mxSetField(s, 0, "id",           mxCreateString(v->id.c_str()));
    mxSetField(s, 0, "label",        mxCreateString(v->label.c_str()));
    mxSetField(s, 0, "bundle",       mxCreateString(toString(v->bundle)));
    mxSetField(s, 0, "holonomy",     mxCreateString(toString(v->holonomy)));
    mxSetField(s, 0, "order",        mxCreateString(toString(v->order)));
    mxSetField(s, 0, "role",         mxCreateString(toString(v->role)));
    mxSetField(s, 0, "field_type",   mxCreateString(toString(v->field_type)));
    mxSetField(s, 0, "domain",       mxCreateString(toString(v->domain)));
    mxSetField(s, 0, "singular",     mxCreateString(toString(v->singular)));
    mxSetField(s, 0, "gauge",        mxCreateString(toString(v->gauge)));
    mxSetField(s, 0, "coupling",     mxCreateString(toString(v->coupling)));
    mxSetField(s, 0, "natural_mass", mxCreateString(v->natural_mass.c_str()));
    mxSetField(s, 0, "graded",       mxCreateLogicalScalar(v->graded));
    mxSetField(s, 0, "tau_presets",  mxCreateString(v->tau_presets.c_str()));
    mxSetField(s, 0, "status",       mxCreateString(toString(v->status)));
    mxSetField(s, 0, "notes",        mxCreateString(v->notes.c_str()));
    mxSetField(s, 0, "squares_to",   mxCreateString(v->square.present &&  v->square.isSquaresTo ? v->square.target.c_str() : ""));
    mxSetField(s, 0, "square_of",    mxCreateString(v->square.present && !v->square.isSquaresTo ? v->square.target.c_str() : ""));
    mxSetField(s, 0, "relation",     mxCreateString(v->square.present ? toString(v->square.relation) : ""));
    plhs[0] = s;
}
```

Register in the dispatch chain (after the `cmd == "create"` line, ~1974):

```cpp
        else if (cmd == "operatorInfo") cmdOperatorInfo(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 3: Build the MEX and run the test**

Run: `bash scripts/build.sh Release 2>&1 | tail -5`
Then via the MATLAB MCP `run_matlab_file` (per memory `nxr-compute-mex-build-test`):
run `bindings/mex/test/test_operator_info.m`.
Expected: `test_operator_info PASSED`.

- [ ] **Step 4: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_operator_info.m
git commit -m "feat(mex): operatorInfo command returns registry metadata"
```

---

## Task 8: WASM `operatorInfo` mirror

**Files:**
- Modify: `bindings/wasm/src/nxr_compute_wasm.cpp`
- Create: `bindings/wasm/test/_smoke-operator-info.mjs`

- [ ] **Step 1: Write the failing smoke test**

Create `bindings/wasm/test/_smoke-operator-info.mjs`:

```javascript
import createModule from '../../../build-wasm/nxr_compute.mjs';
const Module = await createModule();
const info = Module.operatorInfo('laplaceBeltrami');
if (info.bundle !== 'scalar' || info.order !== 'second')
  throw new Error('operatorInfo metadata mismatch: ' + JSON.stringify(info));
const ew = Module.operatorInfo('extrinsicWeitzenbockLaplacian');
if (ew.status !== 'planned') throw new Error('Weitzenbock status: ' + ew.status);
console.log('operatorInfo WASM smoke PASSED');
```

> Verify the module path/export shape against the existing
> `scripts/_smoke-wasm.mjs` and adjust the import accordingly.

- [ ] **Step 2: Implement the Embind binding**

In `bindings/wasm/src/nxr_compute_wasm.cpp`, add `#include "nxr/operator_registry.h"`,
then a free function + Embind registration (mirror the MEX struct as a JS object
via `emscripten::val`):

```cpp
emscripten::val operatorInfoJS(std::string id) {
    using namespace nxr::manifold::registry;
    const OperatorVariant* v = operatorById(id);
    if (!v) throw std::runtime_error("[INVALID_INPUT] unknown operator id: " + id);
    emscripten::val o = emscripten::val::object();
    o.set("id", v->id);                 o.set("label", v->label);
    o.set("bundle", std::string(toString(v->bundle)));
    o.set("holonomy", std::string(toString(v->holonomy)));
    o.set("order", std::string(toString(v->order)));
    o.set("role", std::string(toString(v->role)));
    o.set("field_type", std::string(toString(v->field_type)));
    o.set("domain", std::string(toString(v->domain)));
    o.set("singular", std::string(toString(v->singular)));
    o.set("gauge", std::string(toString(v->gauge)));
    o.set("coupling", std::string(toString(v->coupling)));
    o.set("natural_mass", v->natural_mass);
    o.set("graded", v->graded);
    o.set("tau_presets", v->tau_presets);
    o.set("status", std::string(toString(v->status)));
    o.set("notes", v->notes);
    o.set("squares_to", v->square.present &&  v->square.isSquaresTo ? v->square.target : std::string());
    o.set("square_of",  v->square.present && !v->square.isSquaresTo ? v->square.target : std::string());
    o.set("relation",   v->square.present ? std::string(toString(v->square.relation)) : std::string());
    return o;
}
```

In the `EMSCRIPTEN_BINDINGS(...)` block, add:

```cpp
    emscripten::function("operatorInfo", &operatorInfoJS);
```

- [ ] **Step 3: Build WASM and run the smoke**

Run: `PATH="/opt/homebrew/bin:$PATH" bash scripts/build-wasm.sh 2>&1 | tail -5`
Then: `node bindings/wasm/test/_smoke-operator-info.mjs`
Expected: `operatorInfo WASM smoke PASSED`.

- [ ] **Step 4: Commit**

```bash
git add bindings/wasm/src/nxr_compute_wasm.cpp bindings/wasm/test/_smoke-operator-info.mjs
git commit -m "feat(wasm): operatorInfo Embind binding mirrors MEX metadata"
```

---

## Task 9: Documentation — CLAUDE.md registry section

**Files:**
- Modify: `CLAUDE.md` (after the "Operators surface (opt-in)" section)

- [ ] **Step 1: Add a registry subsection**

Add a concise paragraph documenting: the registry is the single source of truth
(`include/nxr/operator_registry.h`); the controlled-vocab schema; the
`OperatorVariant` key resolving `(OperatorId, gauge, coupling, domain, τ)`; the
`requireBundle` LJC-trap guard; `operatorInfo` in MEX/WASM; and that
`extrinsicWeitzenbockLaplacian` is `status: planned`. Link the spec.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(registry): document operator registry in CLAUDE.md"
```

---

## Self-Review

**Spec coverage:**
- §2 schema → Task 1 (enums/struct) + Task 2 (all fields populated). ✓
- §3 OperatorVariant key → Task 1 struct (`op_id` + gauge/coupling/domain) + Task 2 shared-id rows + Task 3 `variantIdsFor`. ✓
- §4 full registry → Task 2. ✓
- §5 cross-link `relation` → Task 1 `CrossLink` + Task 5 numerical (exact + principal_part). ✓
- §6 extrinsic Weitzenböck → Task 2 (`status: planned` stub); **build deferred to follow-on plan** (flagged in header). ✓
- §7 enforcement → Task 3 (compile-time switch) + Task 5 (numerical). ✓
- §8 C++ surface / bindings → Task 1 API, Task 6 `eigenProblemFor`, Task 7 MEX, Task 8 WASM. ✓
- §9 disambiguation → Task 2 rows (Levi-Civita/trivial, flat/product, two face Laplacians, combinatorial graph) + Task 4 query test. ✓
- §10 testing → Tasks 5,6,7,8. ✓

**Placeholder scan:** No "TBD"/"handle appropriately". The one open item (§6
`W_extrinsic` discretization) is explicitly deferred to a named follow-on plan,
not left vague in this plan's tasks. ✓

**Type consistency:** `OperatorVariant`, `CrossLink{present,isSquaresTo,target,relation}`,
`operatorById`, `operatorsWhere`, `requireBundle`, `variantIdsFor`, `toString`
overloads — used identically across Tasks 1–8. `blockKron`/`d0`/`Error` flagged
for namespace verification at first use. ✓

**Known verification points (grep before relying):** `Error`/`ErrorCode`
constructor signature; `blockKron` and `d0` namespaces; WASM module export shape;
whether `add_library(nxr_compute …)` lists sources explicitly vs globs.

---

## Follow-on plan (separate)

`docs/superpowers/plans/2026-06-25-extrinsic-weitzenbock.md` (to be written after
this ships): add `OperatorId::ExtrinsicWeitzenbock`, implement
`flatCovariantLaplacian + W_extrinsic` (resolve the §6 discretization), flip the
registry entry to `status: built` with its real `op_id`, and add the cross-bundle
anchor test against the immersion squared Dirac's imaginary block.
