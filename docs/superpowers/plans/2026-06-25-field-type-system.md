# Field Type System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a machine-checkable Field type system — the dual of the operator registry — that types fields (`FieldDescriptor` + a catalogue of canonical Field variants), couples them to operators (`input_field`/`output_field`, `requireField`, `operatorsAccepting`), and declares a conversion graph wired to existing implementations.

**Architecture:** A new `field_registry.{h,cpp}` in the `nxr::manifold::registry` namespace (reusing the operator registry's `Domain`/`Bundle`/`FieldType`/`Gauge` enums) holds the `FieldDescriptor` schema, the Field-variant catalogue, the routing API, and the conversion graph. The `OperatorVariant` struct gains `input_field`/`output_field`, populated by an adjacent id-keyed table patched in at static-init (existing entries untouched). Additive, zero breakage; `fieldInfo` mirrors `operatorInfo` in both bindings.

**Tech Stack:** C++17, Eigen, geometry-central, CMake/CTest, MATLAB MEX, Emscripten/Embind WASM.

**Spec:** `docs/superpowers/specs/2026-06-25-field-type-system-design.md`

**Build/test notes for this machine:**
- Build: `bash scripts/build.sh Release 2>&1 | tail -30`. Native test binaries land in `./build/` (NOT `build/Release/`).
- `Error`/`ErrorCode` live in `nxr::core` (header `include/nxr/errors.h`); throw `nxr::core::Error(nxr::core::ErrorCode::InvalidInput, msg, hint)`.
- WASM: `PATH="/opt/homebrew/bin:$PATH" bash scripts/build-wasm.sh`; smokes run with `node <file>.mjs`.
- MEX: built by `build.sh` → `build/Release/nxr_compute.mexmaca64`; run `.m` tests via the MATLAB MCP (`ToolSearch` for `mcp__plugin_brainstorm-dev_MATLAB__run_matlab_file`).

---

## File Structure

| File | Responsibility |
|---|---|
| `include/nxr/field_registry.h` (create) | `NForm`/`Representation` enums + `toString`; `FieldDescriptor`; `FieldVariant`; `ConversionEdge`; API (`fieldRegistry`/`fieldById`/`fieldsWhere`, `fieldMatches`/`requireField`/`operatorsAccepting`/`validateFieldShape`, `conversionGraph`). |
| `src/field_registry.cpp` (create) | The variant catalogue, the conversion table, and all API implementations. |
| `test/test_field_registry.cpp` (create) | Catalogue completeness, operator I/O cross-reference integrity, conversion-edge integrity, routing accept/reject, shape validation. |
| `include/nxr/operator_registry.h` (modify) | `OperatorVariant` += `input_field`, `output_field`. |
| `src/operator_registry.cpp` (modify) | Populate field I/O via an adjacent id-keyed table patched in at init. |
| `bindings/mex/src/nxr_compute_mex.cpp` (modify) | `fieldInfo` command; surface `input_field`/`output_field` in `operatorInfo`. |
| `bindings/mex/test/test_field_info.m` (create) | MEX `fieldInfo` round-trip. |
| `bindings/wasm/src/nxr_compute_wasm.cpp` (modify) | `fieldInfo` Embind function; add op I/O to `operatorInfoJS`. |
| `bindings/wasm/test/_smoke-field-info.mjs` (create) | WASM `fieldInfo` smoke. |
| `CMakeLists.txt` (modify) | Add `src/field_registry.cpp` to the lib; register `test_field_registry`. |
| `CLAUDE.md` (modify) | Document the field registry. |

---

## Task 1: Field registry header + scalar skeleton

**Files:**
- Create: `include/nxr/field_registry.h`
- Create: `src/field_registry.cpp`
- Create: `test/test_field_registry.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the header**

Create `include/nxr/field_registry.h`:

```cpp
#pragma once
#include "nxr/operator_registry.h"   // reuse Domain, Bundle, FieldType, Gauge, OperatorVariant API
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace nxr::manifold::registry {

// New axes the operator registry didn't need (operators don't carry n-form degree
// or a component-layout representation; fields do).
enum class NForm          { na, zero, one, two };
enum class Representation  { na, world, local_frame, intrinsic_complex, quaternion_interleaved };

inline const char* toString(NForm v)         { static const char* s[]{"na","zero","one","two"}; return s[(int)v]; }
inline const char* toString(Representation v) { static const char* s[]{"na","world","local_frame","intrinsic_complex","quaternion_interleaved"}; return s[(int)v]; }

// The spatial type of a field. No time axis by design: a Field is "a typed payload
// over a domain", never "a vector" — a future temporal subsystem composes OVER this.
struct FieldDescriptor {
    Domain         domain;
    Bundle         bundle;
    FieldType      field_type;
    NForm          n_form         = NForm::na;
    Representation  representation = Representation::na;
    Gauge          gauge          = Gauge::na;   // which coordinate system local_frame/intrinsic_complex references
    int            nSym           = 1;           // n-RoSy order (tangent); advisory parameter, not a match key
};

struct FieldVariant {
    std::string     id;
    std::string     label;
    FieldDescriptor descriptor;
    std::string     notes;
};

// A declared representation conversion, wired to an existing implementation.
struct ConversionEdge {
    std::string from;          // Field-variant id
    std::string to;            // Field-variant id
    std::string impl;          // existing function name ("" if declared-only)
    bool        implemented;   // false ⇒ declared, unimplemented
};

// ── Catalogue ──
const std::vector<FieldVariant>& fieldRegistry();
const FieldVariant*  fieldById(std::string_view id);
std::vector<const FieldVariant*>
                     fieldsWhere(const std::function<bool(const FieldVariant&)>& pred);

// ── Routing (Task 4) ──
// Structural match: domain + bundle + field_type + n_form + representation.
// gauge and nSym are advisory parameters, NOT match keys.
bool fieldMatches(const FieldDescriptor& f, const FieldDescriptor& expected);
// Inverse of requireBundle: throw if f is not admissible as operatorId's input_field.
void requireField(const FieldDescriptor& f, std::string_view operatorId);
// Which operators accept a field of descriptor f as input.
std::vector<std::string> operatorsAccepting(const FieldDescriptor& f);
// Components per element, derived from bundle+field_type (scalar=1, tangent complex=1,
// ambient=3, immersion=4). Used by validateFieldShape.
int componentsPerElement(const FieldDescriptor& f);
// rows must equal nElements(domain) * componentsPerElement(f). Throws otherwise.
void validateFieldShape(const FieldDescriptor& f, int rows, int nV, int nE, int nF);

// ── Conversion graph (Task 5) ──
const std::vector<ConversionEdge>& conversionGraph();

} // namespace nxr::manifold::registry
```

- [ ] **Step 2: Write the failing test**

Create `test/test_field_registry.cpp`:

```cpp
/**
 * test_field_registry.cpp — Field type system: catalogue, operator coupling,
 * conversion graph, routing.
 * Build: cmake --build build --target test_field_registry
 * Run:   ./build/test_field_registry
 */
#include "nxr/field_registry.h"
#include "nxr/operator_registry.h"
#include "nxr/errors.h"
#include <iostream>
#include <string>

using namespace nxr::manifold;
using namespace nxr::manifold::registry;

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << (msg) << "\n"; ++g_failures; } } while (0)

static void test_scalar_skeleton() {
    const FieldVariant* sv = fieldById("scalarVertex");
    CHECK(sv != nullptr, "scalarVertex present");
    if (sv) {
        CHECK(sv->descriptor.domain == Domain::vertex, "scalarVertex domain=vertex");
        CHECK(sv->descriptor.bundle == Bundle::scalar, "scalarVertex bundle=scalar");
        CHECK(sv->descriptor.n_form == NForm::zero,    "scalarVertex n_form=zero");
    }
    CHECK(fieldById("doesNotExist") == nullptr, "unknown id → nullptr");
}

int main() {
    test_scalar_skeleton();
    std::cout << (g_failures ? "FIELD REGISTRY TESTS FAILED\n" : "ALL FIELD REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
```

- [ ] **Step 3: Register in CMake**

In `CMakeLists.txt`: append `src/field_registry.cpp` to the `nxr_compute` library source list (next to `src/operator_registry.cpp`). Then, after the `test_operator_registry` block, add:

```cmake
    add_executable(test_field_registry test/test_field_registry.cpp)
    target_link_libraries(test_field_registry PRIVATE nxr_compute)
    add_test(NAME test_field_registry COMMAND test_field_registry)
```

- [ ] **Step 4: Write the minimal implementation**

Create `src/field_registry.cpp`:

```cpp
#include "nxr/field_registry.h"
#include "nxr/errors.h"

namespace nxr::manifold::registry {

using nxr::core::Error;
using nxr::core::ErrorCode;

static FieldDescriptor desc(Domain d, Bundle b, FieldType ft, NForm nf,
                            Representation rep, Gauge g = Gauge::na, int nSym = 1) {
    return FieldDescriptor{ d, b, ft, nf, rep, g, nSym };
}

const std::vector<FieldVariant>& fieldRegistry() {
    static const std::vector<FieldVariant> table = {
        { "scalarVertex", "Scalar (0-form, vertex)",
          desc(Domain::vertex, Bundle::scalar, FieldType::real, NForm::zero, Representation::na),
          "vertex function / 0-form" },
    };
    return table;
}

const FieldVariant* fieldById(std::string_view id) {
    for (const auto& v : fieldRegistry()) if (v.id == id) return &v;
    return nullptr;
}

std::vector<const FieldVariant*>
fieldsWhere(const std::function<bool(const FieldVariant&)>& pred) {
    std::vector<const FieldVariant*> out;
    for (const auto& v : fieldRegistry()) if (pred(v)) out.push_back(&v);
    return out;
}

// fieldMatches / requireField / operatorsAccepting / componentsPerElement /
// validateFieldShape / conversionGraph are implemented in Tasks 4 and 5.

} // namespace nxr::manifold::registry
```

> Note: the routing + conversion functions are declared but defined in later tasks.
> Nothing references them yet, so the Task-1 build links. If a stray
> undefined-reference appears, confirm the Task-1 test only calls `fieldById`.

- [ ] **Step 5: Build & run**

Run: `bash scripts/build.sh Release 2>&1 | tail -20 && ./build/test_field_registry`
Expected: `ALL FIELD REGISTRY TESTS PASSED`

- [ ] **Step 6: Commit**

```bash
cd /Users/diellorbasha/workspace/research/code/nxr-compute
git add include/nxr/field_registry.h src/field_registry.cpp test/test_field_registry.cpp CMakeLists.txt
git commit -m "feat(field): scaffold field registry with scalar skeleton

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Populate the Field variant catalogue

**Files:** Modify `src/field_registry.cpp`, `test/test_field_registry.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/test_field_registry.cpp` and call `test_full_catalogue();` in `main()`:

```cpp
static void test_full_catalogue() {
    const char* ids[] = {"scalarVertex","oneFormEdge","twoFormFace","tangentVertex",
                         "tangentFace","ambientVertexWorld","ambientVertexLocal",
                         "ambientEdge","ambientFaceWorld","immersionVertex","immersionFace"};
    for (const char* id : ids) CHECK(fieldById(id) != nullptr, std::string("variant present: ") + id);

    const FieldVariant* tv = fieldById("tangentVertex");
    CHECK(tv && tv->descriptor.bundle == Bundle::tangent && tv->descriptor.field_type == FieldType::complex,
          "tangentVertex tangent+complex");
    CHECK(tv && tv->descriptor.representation == Representation::intrinsic_complex, "tangentVertex repr");
    const FieldVariant* aw = fieldById("ambientVertexWorld");
    CHECK(aw && aw->descriptor.representation == Representation::world, "ambientVertexWorld repr=world");
    const FieldVariant* al = fieldById("ambientVertexLocal");
    CHECK(al && al->descriptor.representation == Representation::local_frame, "ambientVertexLocal repr=local_frame");
    const FieldVariant* iv = fieldById("immersionVertex");
    CHECK(iv && iv->descriptor.bundle == Bundle::immersion && iv->descriptor.field_type == FieldType::quaternion,
          "immersionVertex immersion+quaternion");
    const FieldVariant* of = fieldById("oneFormEdge");
    CHECK(of && of->descriptor.domain == Domain::edge && of->descriptor.n_form == NForm::one, "oneFormEdge edge 1-form");
}
```

Build & run — confirm it FAILS (only `scalarVertex` exists).

- [ ] **Step 2: Fill the catalogue**

Replace the `table` initializer body in `fieldRegistry()` with the full set (keep `scalarVertex` first):

```cpp
    static const std::vector<FieldVariant> table = {
        // scalar / DEC forms
        { "scalarVertex", "Scalar (0-form, vertex)",
          desc(Domain::vertex, Bundle::scalar, FieldType::real, NForm::zero, Representation::na), "vertex function / 0-form" },
        { "oneFormEdge", "1-form (edge)",
          desc(Domain::edge, Bundle::scalar, FieldType::real, NForm::one, Representation::na), "DEC 1-form" },
        { "twoFormFace", "2-form (face)",
          desc(Domain::face, Bundle::scalar, FieldType::real, NForm::two, Representation::na), "per-face scalar / 2-form" },
        // tangent
        { "tangentVertex", "Tangent field (vertex)",
          desc(Domain::vertex, Bundle::tangent, FieldType::complex, NForm::na, Representation::intrinsic_complex, Gauge::levi_civita),
          "n-RoSy via nSym" },
        { "tangentFace", "Tangent field (face)",
          desc(Domain::face, Bundle::tangent, FieldType::complex, NForm::na, Representation::intrinsic_complex, Gauge::levi_civita),
          "n-RoSy via nSym" },
        // ambient (ℝ³)
        { "ambientVertexWorld", "Ambient vector (vertex, world)",
          desc(Domain::vertex, Bundle::ambient, FieldType::real, NForm::na, Representation::world),
          "R^3 global Cartesian (e.g. leadfield)" },
        { "ambientVertexLocal", "Ambient vector (vertex, local frame)",
          desc(Domain::vertex, Bundle::ambient, FieldType::real, NForm::na, Representation::local_frame, Gauge::levi_civita),
          "R^3 in per-vertex frame [a;b;c]" },
        { "ambientEdge", "Ambient vector (edge, local frame)",
          desc(Domain::edge, Bundle::ambient, FieldType::real, NForm::na, Representation::local_frame, Gauge::levi_civita),
          "covariant-difference output (3E)" },
        { "ambientFaceWorld", "Ambient vector (face, world)",
          desc(Domain::face, Bundle::ambient, FieldType::real, NForm::na, Representation::world),
          "R^3 per face (gradient/whitney output)" },
        // immersion (quaternion)
        { "immersionVertex", "Immersion spinor (vertex)",
          desc(Domain::vertex, Bundle::immersion, FieldType::quaternion, NForm::na, Representation::quaternion_interleaved),
          "4v+c shape-spinor" },
        { "immersionFace", "Immersion spinor (face)",
          desc(Domain::face, Bundle::immersion, FieldType::quaternion, NForm::na, Representation::quaternion_interleaved),
          "4f+c shape-spinor" },
    };
```

- [ ] **Step 3: Build & run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_field_registry`
Expected: `ALL FIELD REGISTRY TESTS PASSED`

- [ ] **Step 4: Commit**

```bash
git add src/field_registry.cpp test/test_field_registry.cpp
git commit -m "feat(field): populate the Field variant catalogue

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Extend OperatorVariant with input_field / output_field

**Files:** Modify `include/nxr/operator_registry.h`, `src/operator_registry.cpp`, `test/test_field_registry.cpp`

- [ ] **Step 1: Add the struct fields**

In `include/nxr/operator_registry.h`, add two trailing members to `OperatorVariant` (after `std::string notes;`), with defaults so the existing 25 brace-initializers stay valid unchanged:

```cpp
    std::string notes;
    std::string input_field;    // Field-variant id consumed (populated post-init)
    std::string output_field;   // Field-variant id produced
```

- [ ] **Step 2: Populate via an adjacent id-keyed table (existing entries untouched)**

In `src/operator_registry.cpp`, change `operatorRegistry()` so the static table is built by a lambda that patches in the field I/O by id. Replace:

```cpp
const std::vector<OperatorVariant>& operatorRegistry() {
    static const std::vector<OperatorVariant> table = {
        // ... 25 entries ...
    };
    return table;
}
```

with (keep the 25 entries EXACTLY as they are inside the initializer list):

```cpp
const std::vector<OperatorVariant>& operatorRegistry() {
    static const std::vector<OperatorVariant> table = [] {
        std::vector<OperatorVariant> t = {
            // ... the existing 25 entries, UNCHANGED ...
        };
        // Field I/O declared adjacent to the table, keyed by curated id (NOT OperatorId:
        // DiracFace backs both faceLaplacian2Form and relativeFaceDirac with different I/O).
        struct IO { const char* id; const char* in; const char* out; };
        static const IO io[] = {
            { "laplaceBeltrami",                "scalarVertex",      "scalarVertex" },
            { "graphLaplacian",                 "scalarVertex",      "scalarVertex" },
            { "faceLaplacianGreenGauss",        "twoFormFace",       "twoFormFace" },
            { "faceLaplacian2Form",             "twoFormFace",       "twoFormFace" },
            { "leviCivitaConnectionLaplacian",  "tangentVertex",     "tangentVertex" },
            { "trivialConnectionLaplacian",     "tangentVertex",     "tangentVertex" },
            { "flatCovariantLaplacian",         "ambientVertexLocal","ambientVertexLocal" },
            { "productCovariantLaplacian",      "ambientVertexLocal","ambientVertexLocal" },
            { "covariantGradient",              "ambientVertexLocal","ambientEdge" },
            { "faceGradient",                   "twoFormFace",       "ambientFaceWorld" },
            { "extrinsicWeitzenbockLaplacian",  "ambientVertexLocal","ambientVertexLocal" },
            { "intrinsicDirac",                 "immersionVertex",   "immersionFace" },
            { "extrinsicDirac",                 "immersionVertex",   "immersionFace" },
            { "relativeDirac",                  "immersionVertex",   "immersionVertex" },
            { "intrinsicFaceDirac",             "immersionFace",     "immersionVertex" },
            { "extrinsicFaceDirac",             "immersionFace",     "immersionVertex" },
            { "relativeFaceDirac",              "immersionFace",     "immersionFace" },
            { "massLumped",                     "scalarVertex",      "scalarVertex" },
            { "massGalerkin",                   "scalarVertex",      "scalarVertex" },
            { "d0",                             "scalarVertex",      "oneFormEdge" },
            { "d1",                             "oneFormEdge",       "twoFormFace" },
            { "hodge0",                         "scalarVertex",      "scalarVertex" },
            { "hodge1",                         "oneFormEdge",       "oneFormEdge" },
            { "hodge2",                         "twoFormFace",       "twoFormFace" },
            { "hodge1inv",                      "oneFormEdge",       "oneFormEdge" },
        };
        for (auto& e : t)
            for (const auto& x : io)
                if (e.id == x.id) { e.input_field = x.in; e.output_field = x.out; break; }
        return t;
    }();
    return table;
}
```

> This leaves the 25 entries verbatim (no positional rewrite) and keeps the field
> I/O readable and adjacent. Every operator id must appear in `io` — the Step-3 test
> enforces completeness.

- [ ] **Step 3: Cross-reference integrity test**

Append to `test/test_field_registry.cpp` and call `test_operator_io_integrity();` in `main()`:

```cpp
static void test_operator_io_integrity() {
    for (const auto& op : operatorRegistry()) {
        CHECK(!op.input_field.empty(),  std::string("operator has input_field: ")  + op.id);
        CHECK(!op.output_field.empty(), std::string("operator has output_field: ") + op.id);
        CHECK(fieldById(op.input_field)  != nullptr, std::string("input_field resolves: ")  + op.id + " -> " + op.input_field);
        CHECK(fieldById(op.output_field) != nullptr, std::string("output_field resolves: ") + op.id + " -> " + op.output_field);
    }
    // spot-check the DiracFace-shared pair has DISTINCT I/O
    const OperatorVariant* f2 = operatorById("faceLaplacian2Form");
    const OperatorVariant* rf = operatorById("relativeFaceDirac");
    CHECK(f2 && f2->input_field == "twoFormFace",   "faceLaplacian2Form input twoFormFace");
    CHECK(rf && rf->input_field == "immersionFace", "relativeFaceDirac input immersionFace");
}
```

- [ ] **Step 4: Build & run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_field_registry && ./build/test_operator_registry`
Expected: both pass. (`test_operator_registry` confirms the OperatorVariant change didn't break existing operator tests.)

- [ ] **Step 5: Commit**

```bash
git add include/nxr/operator_registry.h src/operator_registry.cpp test/test_field_registry.cpp
git commit -m "feat(field): operators declare input_field/output_field

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Routing API (fieldMatches / requireField / operatorsAccepting / shape)

**Files:** Modify `src/field_registry.cpp`, `test/test_field_registry.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/test_field_registry.cpp` and call `test_routing();` in `main()`:

```cpp
static void test_routing() {
    const FieldDescriptor& scalarV = fieldById("scalarVertex")->descriptor;
    const FieldDescriptor& tangV   = fieldById("tangentVertex")->descriptor;
    const FieldDescriptor& immV    = fieldById("immersionVertex")->descriptor;

    // requireField accepts the right field, rejects the wrong one.
    bool ok = true;
    try { requireField(scalarV, "laplaceBeltrami"); } catch (const nxr::core::Error&) { ok = false; }
    CHECK(ok, "scalarVertex accepted by laplaceBeltrami");

    bool threw = false;
    try { requireField(tangV, "laplaceBeltrami"); } catch (const nxr::core::Error&) { threw = true; }
    CHECK(threw, "tangentVertex rejected by laplaceBeltrami");

    threw = false;
    try { requireField(immV, "flatCovariantLaplacian"); } catch (const nxr::core::Error&) { threw = true; }
    CHECK(threw, "immersionVertex rejected by ambient operator");

    // operatorsAccepting(scalarVertex) includes laplaceBeltrami and graphLaplacian.
    auto ops = operatorsAccepting(scalarV);
    bool hasLB = false, hasGL = false;
    for (auto& id : ops) { if (id == "laplaceBeltrami") hasLB = true; if (id == "graphLaplacian") hasGL = true; }
    CHECK(hasLB && hasGL, "operatorsAccepting(scalarVertex) ⊇ {laplaceBeltrami, graphLaplacian}");

    // componentsPerElement
    CHECK(componentsPerElement(scalarV) == 1, "scalar components=1");
    CHECK(componentsPerElement(fieldById("ambientVertexWorld")->descriptor) == 3, "ambient components=3");
    CHECK(componentsPerElement(immV) == 4, "immersion components=4");

    // validateFieldShape: scalarVertex on a 12-vertex mesh wants 12 rows.
    bool shapeOk = true;
    try { validateFieldShape(scalarV, 12, /*nV*/12, /*nE*/30, /*nF*/20); } catch (const nxr::core::Error&) { shapeOk = false; }
    CHECK(shapeOk, "scalarVertex 12 rows valid on 12-vertex mesh");
    bool shapeThrew = false;
    try { validateFieldShape(scalarV, 11, 12, 30, 20); } catch (const nxr::core::Error&) { shapeThrew = true; }
    CHECK(shapeThrew, "scalarVertex wrong row count rejected");
}
```

Build & run — confirm link failure (routing functions undefined).

- [ ] **Step 2: Implement the routing functions**

Add to `src/field_registry.cpp` before the closing namespace brace:

```cpp
bool fieldMatches(const FieldDescriptor& f, const FieldDescriptor& expected) {
    // gauge and nSym are advisory parameters, not match keys.
    return f.domain == expected.domain
        && f.bundle == expected.bundle
        && f.field_type == expected.field_type
        && f.n_form == expected.n_form
        && f.representation == expected.representation;
}

void requireField(const FieldDescriptor& f, std::string_view operatorId) {
    const OperatorVariant* op = operatorById(operatorId);
    if (!op)
        throw Error(ErrorCode::InvalidInput, "unknown operator id: " + std::string(operatorId),
                    "check operatorRegistry()");
    const FieldVariant* expected = fieldById(op->input_field);
    if (!expected)
        throw Error(ErrorCode::InvalidInput,
                    "operator '" + std::string(operatorId) + "' names unknown input_field '" + op->input_field + "'",
                    "field registry integrity error");
    if (!fieldMatches(f, expected->descriptor))
        throw Error(ErrorCode::InvalidInput,
                    "field not admissible as input of '" + std::string(operatorId) + "' (expected " + op->input_field + ")",
                    "field bundle/domain/degree/representation mismatch");
}

std::vector<std::string> operatorsAccepting(const FieldDescriptor& f) {
    std::vector<std::string> out;
    for (const auto& op : operatorRegistry()) {
        const FieldVariant* in = fieldById(op.input_field);
        if (in && fieldMatches(f, in->descriptor)) out.push_back(op.id);
    }
    return out;
}

int componentsPerElement(const FieldDescriptor& f) {
    switch (f.bundle) {
        case Bundle::scalar:    return 1;
        case Bundle::tangent:   return 1;   // complex coordinate per element (VectorXcd)
        case Bundle::ambient:   return 3;
        case Bundle::immersion: return 4;
    }
    return 1;
}

void validateFieldShape(const FieldDescriptor& f, int rows, int nV, int nE, int nF) {
    int n = (f.domain == Domain::vertex) ? nV : (f.domain == Domain::edge) ? nE : nF;
    int expected = n * componentsPerElement(f);
    if (rows != expected)
        throw Error(ErrorCode::InvalidInput,
                    "field row count " + std::to_string(rows) + " != expected " + std::to_string(expected),
                    "rows must equal nElements(domain) * componentsPerElement");
}
```

> `componentsPerElement` for a real ambient field counts the 3 `[a;b;c]` components,
> so a `[3N]` component-major flattening validates; a complex tangent field is one
> `VectorXcd` entry per element (components=1). Document this convention in the header
> comment if not already clear.

- [ ] **Step 3: Build & run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_field_registry`
Expected: `ALL FIELD REGISTRY TESTS PASSED`

- [ ] **Step 4: Commit**

```bash
git add src/field_registry.cpp test/test_field_registry.cpp
git commit -m "feat(field): routing — requireField, operatorsAccepting, shape validation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Conversion graph

**Files:** Modify `src/field_registry.cpp`, `test/test_field_registry.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/test_field_registry.cpp` and call `test_conversion_graph();` in `main()`:

```cpp
static void test_conversion_graph() {
    const auto& edges = conversionGraph();
    CHECK(!edges.empty(), "conversion graph non-empty");
    for (const auto& e : edges) {
        CHECK(fieldById(e.from) != nullptr, std::string("conversion from resolves: ") + e.from);
        CHECK(fieldById(e.to)   != nullptr, std::string("conversion to resolves: ")   + e.to);
        if (e.implemented) CHECK(!e.impl.empty(), std::string("implemented edge names impl: ") + e.from + "->" + e.to);
    }
    // a known edge: ambientVertexWorld <-> ambientVertexLocal via lift
    bool hasLift = false;
    for (const auto& e : edges)
        if (e.from == "ambientVertexWorld" && e.to == "ambientVertexLocal" && e.implemented) hasLift = true;
    CHECK(hasLift, "world->local lift edge present and implemented");
}
```

Build & run — confirm link failure (`conversionGraph` undefined).

- [ ] **Step 2: Implement the conversion table**

Add to `src/field_registry.cpp` before the closing namespace brace:

```cpp
const std::vector<ConversionEdge>& conversionGraph() {
    static const std::vector<ConversionEdge> edges = {
        { "ambientVertexWorld", "ambientVertexLocal", "differential::liftToFrame", true },
        { "ambientVertexLocal", "ambientVertexWorld", "differential::liftToWorld", true },
        { "ambientVertexWorld", "ambientVertexLocal", "G·cᵀ (covariantGradient correspondence)", true },
        { "oneFormEdge",        "ambientFaceWorld",   "field::interp::whitney",      true },
        { "scalarVertex",       "ambientFaceWorld",   "field::op::gradient",         true },
        { "tangentVertex",      "tangentVertex",      "lowerToReal2N (complex<->real2N)", true },
    };
    return edges;
}
```

> v1 declares edges and names the existing impl; call-through (invoking a conversion
> via the registry) is deferred (spec §10). The duplicate world↔local pair documents
> both the frame-lift route and the `G·cᵀ` leadfield correspondence.

- [ ] **Step 3: Build & run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5 && ./build/test_field_registry`
Expected: `ALL FIELD REGISTRY TESTS PASSED`

- [ ] **Step 4: Commit**

```bash
git add src/field_registry.cpp test/test_field_registry.cpp
git commit -m "feat(field): declared conversion graph wired to existing impls

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: MEX `fieldInfo` + operator I/O in `operatorInfo`

**Files:** Modify `bindings/mex/src/nxr_compute_mex.cpp`; create `bindings/mex/test/test_field_info.m`

- [ ] **Step 1: Write the failing MATLAB test**

Create `bindings/mex/test/test_field_info.m` (copy the addpath/setup preamble from `bindings/mex/test/test_operator_info.m`):

```matlab
function test_field_info()
  % <copy addpath/setup preamble from test_operator_info.m>

  fi = nxr_compute('fieldInfo', 'tangentVertex');
  assert(strcmp(fi.bundle, 'tangent'),            'bundle');
  assert(strcmp(fi.field_type, 'complex'),         'field_type');
  assert(strcmp(fi.representation, 'intrinsic_complex'), 'representation');

  av = nxr_compute('fieldInfo', 'ambientVertexWorld');
  assert(strcmp(av.bundle, 'ambient'),            'ambient bundle');
  assert(strcmp(av.representation, 'world'),        'world repr');

  % operatorInfo now surfaces input/output field
  oi = nxr_compute('operatorInfo', 'laplaceBeltrami');
  assert(strcmp(oi.input_field,  'scalarVertex'),  'op input_field');
  assert(strcmp(oi.output_field, 'scalarVertex'),  'op output_field');

  fprintf('test_field_info PASSED\n');
end
```

- [ ] **Step 2: Implement `cmdFieldInfo` and extend `cmdOperatorInfo`**

In `bindings/mex/src/nxr_compute_mex.cpp`: add `#include "nxr/field_registry.h"` near the other includes. Add a builder next to `cmdOperatorInfo`:

```cpp
static void cmdFieldInfo(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    using namespace nxr::manifold::registry;
    if (nrhs < 2 || !mxIsChar(prhs[1]))
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "fieldInfo requires an id string.");
    char buf[128]; mxGetString(prhs[1], buf, sizeof(buf));
    const FieldVariant* v = fieldById(buf);
    if (!v) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                                   std::string("fieldInfo: unknown field id: ") + buf);
    const char* fields[] = {"id","label","domain","bundle","field_type","n_form",
                            "representation","gauge","nSym","notes"};
    mxArray* s = mxCreateStructMatrix(1, 1, (int)(sizeof(fields)/sizeof(*fields)), fields);
    const FieldDescriptor& d = v->descriptor;
    mxSetField(s, 0, "id",             mxCreateString(v->id.c_str()));
    mxSetField(s, 0, "label",          mxCreateString(v->label.c_str()));
    mxSetField(s, 0, "domain",         mxCreateString(toString(d.domain)));
    mxSetField(s, 0, "bundle",         mxCreateString(toString(d.bundle)));
    mxSetField(s, 0, "field_type",     mxCreateString(toString(d.field_type)));
    mxSetField(s, 0, "n_form",         mxCreateString(toString(d.n_form)));
    mxSetField(s, 0, "representation", mxCreateString(toString(d.representation)));
    mxSetField(s, 0, "gauge",          mxCreateString(toString(d.gauge)));
    mxSetField(s, 0, "nSym",           mxCreateDoubleScalar((double)d.nSym));
    mxSetField(s, 0, "notes",          mxCreateString(v->notes.c_str()));
    plhs[0] = s;
}
```

In `cmdOperatorInfo`, add two fields to its struct (extend the `fields[]` array with `"input_field","output_field"` and set them):

```cpp
    mxSetField(s, 0, "input_field",  mxCreateString(v->input_field.c_str()));
    mxSetField(s, 0, "output_field", mxCreateString(v->output_field.c_str()));
```

Register in the dispatch chain (next to `operatorInfo`):

```cpp
        else if (cmd == "fieldInfo")    cmdFieldInfo(nlhs, plhs, nrhs, prhs);
```

- [ ] **Step 3: Build + run**

Run: `bash scripts/build.sh Release 2>&1 | tail -5`
Then via the MATLAB MCP `run_matlab_file`, run `bindings/mex/test/test_field_info.m`.
Expected: `test_field_info PASSED`.

- [ ] **Step 4: Commit**

```bash
git add bindings/mex/src/nxr_compute_mex.cpp bindings/mex/test/test_field_info.m
git commit -m "feat(mex): fieldInfo command + input/output_field in operatorInfo

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: WASM `fieldInfo`

**Files:** Modify `bindings/wasm/src/nxr_compute_wasm.cpp`; create `bindings/wasm/test/_smoke-field-info.mjs`

- [ ] **Step 1: Write the failing smoke**

Create `bindings/wasm/test/_smoke-field-info.mjs` (copy the module-load preamble from `bindings/wasm/test/_smoke-operator-info.mjs`):

```javascript
// <module load preamble copied from _smoke-operator-info.mjs, yielding `Module`>
const fi = Module.fieldInfo('tangentVertex');
if (fi.bundle !== 'tangent' || fi.field_type !== 'complex' || fi.representation !== 'intrinsic_complex')
  throw new Error('fieldInfo mismatch: ' + JSON.stringify(fi));
const oi = Module.operatorInfo('laplaceBeltrami');
if (oi.input_field !== 'scalarVertex' || oi.output_field !== 'scalarVertex')
  throw new Error('operatorInfo missing field I/O: ' + JSON.stringify(oi));
console.log('fieldInfo WASM smoke PASSED');
```

- [ ] **Step 2: Implement the Embind function + extend operatorInfoJS**

In `bindings/wasm/src/nxr_compute_wasm.cpp`: add `#include "nxr/field_registry.h"`. Add:

```cpp
emscripten::val fieldInfoJS(std::string id) {
    using namespace nxr::manifold::registry;
    const FieldVariant* v = fieldById(id);
    if (!v) {
        try { throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "unknown field id: " + id); }
        catch (const nxr::core::Error& e) { rethrowAsJsError(e); }
    }
    const FieldDescriptor& d = v->descriptor;
    emscripten::val o = emscripten::val::object();
    o.set("id", v->id);                 o.set("label", v->label);
    o.set("domain", std::string(toString(d.domain)));
    o.set("bundle", std::string(toString(d.bundle)));
    o.set("field_type", std::string(toString(d.field_type)));
    o.set("n_form", std::string(toString(d.n_form)));
    o.set("representation", std::string(toString(d.representation)));
    o.set("gauge", std::string(toString(d.gauge)));
    o.set("nSym", d.nSym);
    o.set("notes", v->notes);
    return o;
}
```

In `operatorInfoJS`, add: `o.set("input_field", v->input_field); o.set("output_field", v->output_field);`.

In the `EMSCRIPTEN_BINDINGS(...)` block add: `emscripten::function("fieldInfo", &fieldInfoJS);`.

- [ ] **Step 3: Build + run**

Run: `PATH="/opt/homebrew/bin:$PATH" bash scripts/build-wasm.sh 2>&1 | tail -6`
Then: `node bindings/wasm/test/_smoke-field-info.mjs`
Expected: `fieldInfo WASM smoke PASSED`.

- [ ] **Step 4: Commit**

```bash
git add bindings/wasm/src/nxr_compute_wasm.cpp bindings/wasm/test/_smoke-field-info.mjs
git commit -m "feat(wasm): fieldInfo Embind binding + op field I/O in operatorInfo

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Document the field registry in CLAUDE.md

**Files:** Modify `CLAUDE.md`

- [ ] **Step 1: Add a field-registry subsection**

After the operator-registry paragraph in `CLAUDE.md`, add a concise paragraph covering: `field_registry.{h,cpp}` as the dual of the operator registry; the `FieldDescriptor` axes (domain/bundle/field_type/n_form/representation/gauge/nSym); that frames are coordinate systems (not fields) and time is a deferred orthogonal subsystem (a Field is "payload over a domain", never a vector); the catalogue + `fieldById`/`fieldsWhere`; the operator coupling (`input_field`/`output_field`, `requireField` as the inverse of `requireBundle`, `operatorsAccepting`); the declared conversion graph; and `fieldInfo` in MEX/WASM. Link the spec.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(field): document the field type system in CLAUDE.md

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §3 time guardrail → `FieldDescriptor` has no time axis; header comment states "payload over a domain" (Task 1). ✓
- §4 schema (incl. frames-excluded, cleaned representation) → Task 1 enums/struct. ✓
- §5 variant catalogue → Task 2. ✓
- §6 operator coupling + routing API → Task 3 (`input_field`/`output_field` + integrity) + Task 4 (`requireField`/`operatorsAccepting`/`fieldMatches`/`validateFieldShape`). ✓
- §7 conversion graph → Task 5. ✓
- §8 bindings → Task 6 (MEX) + Task 7 (WASM). ✓
- §9 files/enforcement → all tasks; completeness/integrity/routing/conversion tests in `test_field_registry`. ✓
- §10 deferrals → respected (no time, no halfedge/corner, no call-through, no API rewiring). ✓

**Placeholder scan:** No "TBD"/"handle appropriately". The conversion `implemented=true` edges name real functions; call-through is explicitly deferred, not vaguely left. ✓

**Type consistency:** `FieldDescriptor`, `FieldVariant`, `ConversionEdge`, `fieldById`, `fieldMatches`, `requireField`, `operatorsAccepting`, `componentsPerElement`, `validateFieldShape`, `conversionGraph`, the `input_field`/`output_field` members, and the variant ids (`scalarVertex`…`immersionFace`) are used identically across Tasks 1–8. Field ids referenced in Task 3's operator I/O all exist in Task 2's catalogue. ✓

**Known verification points (grep before relying):** the exact `add_library(nxr_compute …)` source-list form in `CMakeLists.txt`; the MEX `cmdOperatorInfo` `fields[]` array (to extend it); the WASM module-load preamble shape in the existing smokes; that `operatorById`/`operatorRegistry` are reachable from `field_registry.cpp` via the `operator_registry.h` include.
