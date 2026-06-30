# Node addon → WASM parity (registry/operator surface) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the operator-registry / named-operator surface (`operators`, `eigs`, `frames`, `operatorInfo`, `fieldInfo`) to the N-API addon so it reaches parity with the WASM `Manifold`.

**Architecture:** Purely additive to `bindings/node/src/addon.cpp` (five new entry points + one `EigsWorker` AsyncWorker) and `bindings/node/index.mjs` (wiring). Every method calls the same `nxr::manifold` accessors the WASM `ContextWrapper` already uses — single source of truth, no library changes. Marshaling reuses the addon's existing `sparseToCOO` / `sparseComplexToCOO` / `matrixToFloat64Array` / `toFloat64Array` helpers.

**Tech Stack:** C++17, node-addon-api (N-API), Eigen, geometry-central, cmake-js, Node ESM.

## Global Constraints

- **No library changes.** Only `bindings/node/` and `CLAUDE.md` are touched. All operator math comes from `m.operators().*`, `solve::eigenOperator`, `geometry::frames`, and `nxr::manifold::registry::{operatorById,fieldById}` — already in the static lib.
- **§11 storage convention:** sparse → COO `{row,col,data,rows,cols,nnz}` (complex: `realData`/`imagData`); dense `[N×3]`/`[V×K]` → row-major `Float64Array`. Reuse existing helpers.
- **Error contract:** synchronous methods throw a JS `Error` with `.code` (enumerator name, e.g. `"INVALID_INPUT"`) and `.hint` via `nxrSyncCall`; `eigs` rejects its Promise with the same fields.
- **Parity rule:** dispatch strings, arg rules, and error messages must match WASM (`bindings/wasm/src/nxr_compute_wasm.cpp`) verbatim so outputs are byte-identical across bindings.
- **Build command (run from repo root):** `bash scripts/build.sh Release`. It is incremental (ninja); the addon is rebuilt only when `addon.cpp` changes. The built artifact is copied to repo-root `nxr_compute_addon.node`, which `bindings/node/index.mjs` loads.
- **`eigs` is async** (Promise) on the addon — intentional difference from WASM's sync `eigs`, matching the addon's own `solve()`/`hodge()` (CLAUDE.md rule 7).

---

### Task 1: `operators(handle, family, arg)` + `nnz` parity field

**Files:**
- Modify: `bindings/node/src/addon.cpp` — add `nnz` to `sparseToCOO` (≈ line 147-153) and `sparseComplexToCOO` (≈ line 186-193); add `Operators` entry point near the other `Napi::Value` functions; register in `Init()` (≈ line 1120).
- Modify: `bindings/node/index.mjs` — add `operators` to the structured context (≈ line 263, in the object returned by `makeManifoldContext`).
- Test: `scripts/_smoke-addon-operators.mjs` (create)

**Interfaces:**
- Consumes: existing `getContext(info, 0)`, `nxrSyncCall(env, fn)`, `sparseToCOO(env, M)`, `sparseComplexToCOO(env, M)`, `holder->manifold` (a `std::shared_ptr<Manifold>`), `CovariantCoupling::Ambient` (already in scope via `using namespace nxr::manifold::ops::laplacian::connection;`).
- Produces: addon export `operators(handle, family, arg)` → COO object, complex-COO object (`connection`/`connectionGradient`), or `{d0,d1}` (`dec`). Structured `mctx.operators(family, arg)`.

- [ ] **Step 1: Add `nnz` to the two COO helpers**

In `sparseToCOO` (after `result.Set("cols", …)`, before `return result;`):

```cpp
    result.Set("nnz", Napi::Number::New(env, nnz));
```

In `sparseComplexToCOO` (after its `result.Set("cols", …)`, before `return result;`):

```cpp
    result.Set("nnz", Napi::Number::New(env, nnz));
```

- [ ] **Step 2: Add the `Operators` entry point**

Insert this function in `addon.cpp` near the other entry points (e.g. just after `AssembleConnectionLaplacian`):

```cpp
// ─── operators(handle, family, arg) → COO / complex-COO / {d0,d1} ───
// Mirrors ContextWrapper::operators (bindings/wasm/src/nxr_compute_wasm.cpp).
// Same family/subtype dispatch, same arg rules (string subtype | numeric
// tau/nSym | undefined), same error messages — single source of truth is the
// Manifold operators facet, so outputs are byte-identical to the WASM binding.
Napi::Value Operators(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);            // info[0] = External handle
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 2 || !info[1].IsString())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators(handle, family, arg): family must be a string.");
        const std::string family = info[1].As<Napi::String>().Utf8Value();
        const Napi::Value arg = info.Length() > 2 ? info[2] : env.Undefined();
        const bool        hasNum = arg.IsNumber();
        const double      num    = hasNum ? arg.As<Napi::Number>().DoubleValue() : 0.0;
        const std::string sub    = arg.IsString() ? arg.As<Napi::String>().Utf8Value() : "";
        Manifold& m = *holder->manifold;

        if (family == "laplacian") {
            if (sub == "cotan")      return sparseToCOO(env, m.operators().laplacian().cotan());
            if (sub == "graph")      return sparseToCOO(env, m.operators().laplacian().graph());
            if (sub == "connection") return sparseComplexToCOO(env, m.operators().laplacian().connection());
            if (sub == "covariant")  return sparseToCOO(env, m.operators().laplacian().covariant(CovariantCoupling::Ambient));
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators laplacian: subtype must be cotan|graph|connection|covariant.");
        } else if (family == "mass") {
            if (sub == "lumped")   return sparseToCOO(env, m.operators().mass().lumped());
            if (sub == "galerkin") return sparseToCOO(env, m.operators().mass().galerkin());
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators mass: subtype must be lumped|galerkin.");
        } else if (family == "hodge") {
            if (sub == "h0")    return sparseToCOO(env, m.operators().hodge().h0());
            if (sub == "h1")    return sparseToCOO(env, m.operators().hodge().h1());
            if (sub == "h2")    return sparseToCOO(env, m.operators().hodge().h2());
            if (sub == "h1inv") return sparseToCOO(env, m.operators().hodge().h1inv());
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators hodge: subtype must be h0|h1|h2|h1inv.");
        } else if (family == "dec") {
            const auto& dec = m.operators().dec();
            auto obj = Napi::Object::New(env);
            obj.Set("d0", sparseToCOO(env, dec.d0));
            obj.Set("d1", sparseToCOO(env, dec.d1));
            return obj;
        } else if (family == "gradient3D") {
            return sparseToCOO(env, m.operators().gradient3D());
        } else if (family == "dirac") {
            if (!hasNum) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators dirac: expected a numeric tau, operators('dirac', tau).");
            return sparseToCOO(env, m.operators().dirac(num));
        } else if (family == "diracFace") {
            if (!hasNum) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators diracFace: expected a numeric tau, operators('diracFace', tau).");
            return sparseToCOO(env, m.operators().diracFace(num));
        } else if (family == "diracD") {
            return sparseToCOO(env, m.operators().diracD());
        } else if (family == "diracFaceD") {
            return sparseToCOO(env, m.operators().diracFaceD());
        } else if (family == "diracIntrinsicD") {
            return sparseToCOO(env, m.operators().diracIntrinsicD());
        } else if (family == "diracFaceIntrinsicD") {
            return sparseToCOO(env, m.operators().diracFaceIntrinsicD());
        } else if (family == "gradFace") {
            return sparseToCOO(env, m.operators().gradFace());
        } else if (family == "lapFace") {
            return sparseToCOO(env, m.operators().lapFace());
        } else if (family == "connectionGradient") {
            int nSym = hasNum ? static_cast<int>(num) : 1;
            return sparseComplexToCOO(env, m.operators().connectionGradient(nSym));
        } else if (family == "extrinsicWeitzenbock") {
            return sparseToCOO(env, m.operators().extrinsicWeitzenbock());
        }
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operators: family must be "
            "laplacian|mass|hodge|dec|gradient3D|dirac|diracFace|diracD|diracFaceD|"
            "diracIntrinsicD|diracFaceIntrinsicD|gradFace|lapFace|connectionGradient|extrinsicWeitzenbock.");
    });
}
```

Register it in `Init()` alongside the other `exports.Set(...)` lines:

```cpp
    exports.Set("operators", Napi::Function::New(env, Operators));
```

- [ ] **Step 3: Wire the structured surface**

In `bindings/node/index.mjs`, in the object returned by `makeManifoldContext` (the `return { … }` near line 263), add after the `nV/nE/nF` accessors:

```js
    /** Named native-sparse operator (parity with WASM `manifold.operators`).
     *  family ∈ laplacian|mass|hodge|dec|gradient3D|dirac|diracFace|diracD|
     *  diracFaceD|diracIntrinsicD|diracFaceIntrinsicD|gradFace|lapFace|
     *  connectionGradient|extrinsicWeitzenbock. `arg` is a subtype string,
     *  a numeric tau/nSym, or omitted. */
    operators(family, arg) { return addon.operators(rawCtx, family, arg) },
```

- [ ] **Step 4: Write the failing smoke test**

Create `scripts/_smoke-addon-operators.mjs`:

```js
// Smoke test: addon operators() surface. Asserts shapes + the documented
// τ=0 Dirac anchor (dirac(0) == kron(cotanLaplacian, I4)) WITHOUT any WASM
// dependency, so it has signal even if the WASM build is stale.
//
// Usage (from repo root): node scripts/_smoke-addon-operators.mjs
import { initNxrCompute } from '../bindings/node/index.mjs'

const nxr = await initNxrCompute()

const t = (1 + Math.sqrt(5)) / 2
const raw = [
  -1, t, 0,   1, t, 0,  -1, -t, 0,   1, -t, 0,
   0, -1, t,  0, 1, t,   0, -1, -t,  0, 1, -t,
   t, 0, -1,  t, 0, 1,  -t, 0, -1,  -t, 0, 1,
]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) {
  const len = Math.hypot(raw[i], raw[i + 1], raw[i + 2])
  verts[i] = raw[i] / len; verts[i + 1] = raw[i + 1] / len; verts[i + 2] = raw[i + 2] / len
}
const faces = new Int32Array([
  0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
  3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
])

const m = nxr.createManifoldContext(verts, faces)
let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

// Shape + structural checks
const cotan = m.operators('laplacian', 'cotan')
check(cotan.rows === 12 && cotan.cols === 12, 'cotan laplacian is 12×12')
check(cotan.nnz === cotan.data.length, 'cotan COO carries nnz')
// cotan Laplacian rows sum to ~0
const rowsum = new Float64Array(12)
for (let i = 0; i < cotan.nnz; i++) rowsum[cotan.row[i]] += cotan.data[i]
check(Math.max(...rowsum.map(Math.abs)) < 1e-9, 'cotan laplacian rows sum to 0')

const dec = m.operators('dec')
check(dec.d0.rows === 30 && dec.d0.cols === 12, 'd0 is E×V (30×12)')
check(dec.d1.rows === 20 && dec.d1.cols === 30, 'd1 is F×E (20×30)')

const conn = m.operators('laplacian', 'connection')
check(conn.realData.length === conn.imagData.length, 'connection L is complex COO')

// τ=0 Dirac anchor: dirac(0) == kron(cotanLaplacian, I4)
const dirac0 = m.operators('dirac', 0)
check(dirac0.rows === 48 && dirac0.cols === 48, 'dirac(0) is 4V×4V (48×48)')
const key = (r, c) => r * 1000 + c
const dmap = new Map()
for (let i = 0; i < dirac0.nnz; i++) dmap.set(key(dirac0.row[i], dirac0.col[i]), dirac0.data[i])
let maxErr = 0
for (let i = 0; i < cotan.nnz; i++) {
  const r = cotan.row[i], c = cotan.col[i], v = cotan.data[i]
  for (let b = 0; b < 4; b++) {
    const got = dmap.get(key(4 * r + b, 4 * c + b)) ?? 0
    maxErr = Math.max(maxErr, Math.abs(got - v))
  }
}
check(maxErr < 1e-12, `dirac(0) == kron(cotanL, I4) (maxErr ${maxErr.toExponential(2)})`)

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
```

- [ ] **Step 5: Build, then run to verify it fails**

```bash
bash scripts/build.sh Release && node scripts/_smoke-addon-operators.mjs
```

Expected (BEFORE implementing Steps 1-3 — if you run the test against an un-rebuilt addon): `TypeError: addon.operators is not a function` or `m.operators is not a function`. After building with Steps 1-3 applied, it should PASS — so this step is the proof the steps were applied.

- [ ] **Step 6: Run to verify it passes**

```bash
node scripts/_smoke-addon-operators.mjs
```

Expected: all `✓` lines, ending `PASS`, exit 0.

- [ ] **Step 7: Commit**

```bash
git add bindings/node/src/addon.cpp bindings/node/index.mjs scripts/_smoke-addon-operators.mjs nxr_compute_addon.node
git commit -m "feat(node): operators() named-operator surface + nnz COO parity field

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `frames(handle)` + un-stub `measure.frame`

**Files:**
- Modify: `bindings/node/src/addon.cpp` — add `Frames` entry point; register in `Init()`.
- Modify: `bindings/node/index.mjs` — replace `frame: notWired('measure.frame')` (line 225) with a real call.
- Test: `scripts/_smoke-addon-frames.mjs` (create)

**Interfaces:**
- Consumes: `getContext`, `nxrSyncCall`, `matrixToFloat64Array`, `geometry::frames(Manifold&)` → `FaceFrames{e1,e2,normals}` (each `Eigen::MatrixXd` `[nF×3]`; `frames` is in scope via `using namespace nxr::manifold::geometry;`).
- Produces: addon export `frames(handle)` → `{e1, e2, normals}`; structured `mctx.measure.frame()`.

- [ ] **Step 1: Add the `Frames` entry point**

```cpp
// ─── frames(handle) → { e1, e2, normals } (per-face tangent frames) ───
// Mirrors ContextWrapper::frames. Each field is a row-major [nF×3] Float64Array.
Napi::Value Frames(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        FaceFrames f = frames(*holder->manifold);
        auto obj = Napi::Object::New(env);
        obj.Set("e1",      matrixToFloat64Array(env, f.e1));
        obj.Set("e2",      matrixToFloat64Array(env, f.e2));
        obj.Set("normals", matrixToFloat64Array(env, f.normals));
        return obj;
    });
}
```

Register in `Init()`:

```cpp
    exports.Set("frames", Napi::Function::New(env, Frames));
```

- [ ] **Step 2: Un-stub the structured surface**

In `bindings/node/index.mjs`, replace this line (≈ 223-225):

```js
    /** frames is exposed in the WASM binding but not in
     *  the addon yet — see surface-delta note above. */
    frame:                  notWired('measure.frame'),
```

with:

```js
    /** Per-face tangent frames { e1, e2, normals }, each [nF×3] row-major
     *  Float64Array. Parity with WASM `manifold.frames()`. */
    frame() { return addon.frames(rawCtx) },
```

- [ ] **Step 3: Write the failing smoke test**

Create `scripts/_smoke-addon-frames.mjs` (reuse the icosahedron fixture inline — repeat it; the engineer may run tasks out of order):

```js
// Smoke test: addon frames() / measure.frame(). Asserts shape + orthonormality
// of the per-face frame (e1 ⟂ e2, e1 × e2 == normal).
// Usage (from repo root): node scripts/_smoke-addon-frames.mjs
import { initNxrCompute } from '../bindings/node/index.mjs'

const nxr = await initNxrCompute()
const t = (1 + Math.sqrt(5)) / 2
const raw = [
  -1, t, 0,   1, t, 0,  -1, -t, 0,   1, -t, 0,
   0, -1, t,  0, 1, t,   0, -1, -t,  0, 1, -t,
   t, 0, -1,  t, 0, 1,  -t, 0, -1,  -t, 0, 1,
]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) {
  const len = Math.hypot(raw[i], raw[i + 1], raw[i + 2])
  verts[i] = raw[i] / len; verts[i + 1] = raw[i + 1] / len; verts[i + 2] = raw[i + 2] / len
}
const faces = new Int32Array([
  0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
  3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
])

const m = nxr.createManifoldContext(verts, faces)
let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

const fr = m.measure.frame()
check(fr.e1.length === 20 * 3, 'e1 is [20×3]')
check(fr.e2.length === 20 * 3 && fr.normals.length === 20 * 3, 'e2 and normals are [20×3]')

let maxDot = 0, maxCross = 0
const dot = (a, b, o) => a[o] * b[o] + a[o + 1] * b[o + 1] + a[o + 2] * b[o + 2]
for (let f = 0; f < 20; f++) {
  const o = f * 3
  maxDot = Math.max(maxDot, Math.abs(dot(fr.e1, fr.e2, o)))
  // e1 × e2 should equal normals
  const cx = fr.e1[o+1]*fr.e2[o+2] - fr.e1[o+2]*fr.e2[o+1]
  const cy = fr.e1[o+2]*fr.e2[o]   - fr.e1[o]  *fr.e2[o+2]
  const cz = fr.e1[o]  *fr.e2[o+1] - fr.e1[o+1]*fr.e2[o]
  maxCross = Math.max(maxCross, Math.abs(cx-fr.normals[o]), Math.abs(cy-fr.normals[o+1]), Math.abs(cz-fr.normals[o+2]))
}
check(maxDot < 1e-9, `e1 ⟂ e2 (max |dot| ${maxDot.toExponential(2)})`)
check(maxCross < 1e-9, `e1 × e2 == normal (max err ${maxCross.toExponential(2)})`)

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
```

- [ ] **Step 4: Build, then run to verify it passes**

```bash
bash scripts/build.sh Release && node scripts/_smoke-addon-frames.mjs
```

Expected: all `✓`, ending `PASS`. (If you run the test before applying Steps 1-2, `m.measure.frame()` throws `[NOT_WIRED_IN_ADDON]` — that is the failing state.)

- [ ] **Step 5: Commit**

```bash
git add bindings/node/src/addon.cpp bindings/node/index.mjs scripts/_smoke-addon-frames.mjs nxr_compute_addon.node
git commit -m "feat(node): frames() per-face tangent frames; un-stub measure.frame

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `operatorInfo(id)` / `fieldInfo(id)` registry metadata

**Files:**
- Modify: `bindings/node/src/addon.cpp` — add the two registry includes; add `OperatorInfo` and `FieldInfo` entry points; register in `Init()`.
- Test: `scripts/_smoke-addon-registry.mjs` (create)

**Interfaces:**
- Consumes: `nxrSyncCall`; `nxr::manifold::registry::operatorById(id)` → `const OperatorVariant*` (null if unknown); `fieldById(id)` → `const FieldVariant*`; `registry::toString(enum)` overloads. `OperatorVariant` fields: `id,label` (string), `bundle,holonomy,order,role,field_type,domain,singular,gauge,coupling,status` (enums), `square` (`CrossLink{present,isSquaresTo,target,relation}`), `natural_mass,tau_presets,notes,input_field,output_field` (string), `graded` (bool). `FieldVariant`: `id,label,notes` (string), `descriptor` (`FieldDescriptor{domain,bundle,field_type,n_form,representation,gauge}` enums + `nSym` int).
- Produces: handle-free addon exports `operatorInfo(id)` and `fieldInfo(id)` → plain objects (field names match WASM/MEX exactly). These flow into `index.mjs`'s public surface automatically via `...addon`.

- [ ] **Step 1: Add registry includes**

At the top of `addon.cpp`, after `#include "nxr/compute.h"`:

```cpp
#include "nxr/operator_registry.h"
#include "nxr/field_registry.h"
```

- [ ] **Step 2: Add the two entry points**

```cpp
// ─── operatorInfo(id) → operator-registry metadata (handle-free) ───
// Mirrors operatorInfoJS (wasm). Field names match the MEX/WASM struct exactly.
Napi::Value OperatorInfo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 1 || !info[0].IsString())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operatorInfo(id): id must be a string.");
        const std::string id = info[0].As<Napi::String>().Utf8Value();
        using namespace nxr::manifold::registry;
        const OperatorVariant* v = operatorById(id);
        if (!v) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "unknown operator id: " + id);
        auto o = Napi::Object::New(env);
        auto S = [&](const char* k, const std::string& val) { o.Set(k, Napi::String::New(env, val)); };
        S("id",           v->id);
        S("label",        v->label);
        S("bundle",       toString(v->bundle));
        S("holonomy",     toString(v->holonomy));
        S("order",        toString(v->order));
        S("role",         toString(v->role));
        S("field_type",   toString(v->field_type));
        S("domain",       toString(v->domain));
        S("singular",     toString(v->singular));
        S("gauge",        toString(v->gauge));
        S("coupling",     toString(v->coupling));
        S("natural_mass", v->natural_mass);
        o.Set("graded",   Napi::Boolean::New(env, v->graded));
        S("tau_presets",  v->tau_presets);
        S("status",       toString(v->status));
        S("notes",        v->notes);
        S("squares_to",   v->square.present &&  v->square.isSquaresTo ? v->square.target : std::string());
        S("square_of",    v->square.present && !v->square.isSquaresTo ? v->square.target : std::string());
        S("relation",     v->square.present ? std::string(toString(v->square.relation)) : std::string());
        S("input_field",  v->input_field);
        S("output_field", v->output_field);
        return o;
    });
}

// ─── fieldInfo(id) → field-registry metadata (handle-free) ───
Napi::Value FieldInfo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 1 || !info[0].IsString())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "fieldInfo(id): id must be a string.");
        const std::string id = info[0].As<Napi::String>().Utf8Value();
        using namespace nxr::manifold::registry;
        const FieldVariant* v = fieldById(id);
        if (!v) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "unknown field id: " + id);
        const FieldDescriptor& d = v->descriptor;
        auto o = Napi::Object::New(env);
        auto S = [&](const char* k, const std::string& val) { o.Set(k, Napi::String::New(env, val)); };
        S("id",             v->id);
        S("label",          v->label);
        S("domain",         toString(d.domain));
        S("bundle",         toString(d.bundle));
        S("field_type",     toString(d.field_type));
        S("n_form",         toString(d.n_form));
        S("representation", toString(d.representation));
        S("gauge",          toString(d.gauge));
        o.Set("nSym",       Napi::Number::New(env, d.nSym));
        S("notes",          v->notes);
        return o;
    });
}
```

Register in `Init()`:

```cpp
    exports.Set("operatorInfo", Napi::Function::New(env, OperatorInfo));
    exports.Set("fieldInfo",    Napi::Function::New(env, FieldInfo));
```

> Note: `...addon` in `index.mjs` (line 379) spreads these into the public surface automatically, so `nxr.operatorInfo(id)` / `nxr.fieldInfo(id)` work with no index.mjs change.

- [ ] **Step 3: Write the failing smoke test**

Create `scripts/_smoke-addon-registry.mjs`:

```js
// Smoke test: addon operatorInfo()/fieldInfo() handle-free registry lookups.
// Usage (from repo root): node scripts/_smoke-addon-registry.mjs
import { initNxrCompute } from '../bindings/node/index.mjs'

const nxr = await initNxrCompute()
let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

const op = nxr.operatorInfo('laplaceBeltrami')
check(op.id === 'laplaceBeltrami', 'operatorInfo id round-trips')
check(typeof op.bundle === 'string' && op.bundle.length > 0, 'operator bundle is a non-empty string')
check(typeof op.graded === 'boolean', 'operator graded is a boolean')
check('input_field' in op && 'output_field' in op, 'operator carries input/output_field')

const fld = nxr.fieldInfo('scalarVertex')
check(fld.id === 'scalarVertex', 'fieldInfo id round-trips')
check(typeof fld.nSym === 'number', 'field nSym is a number')
check(typeof fld.representation === 'string', 'field representation is a string')

let threw = false
try { nxr.operatorInfo('nope_not_a_real_id') } catch (e) { threw = true; check(e.code === 'INVALID_INPUT', 'unknown operator id throws INVALID_INPUT') }
check(threw, 'unknown operator id throws')

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
```

- [ ] **Step 4: Build, then run to verify it passes**

```bash
bash scripts/build.sh Release && node scripts/_smoke-addon-registry.mjs
```

Expected: all `✓`, ending `PASS`. (Before applying Steps 1-2: `nxr.operatorInfo is not a function`.)

> If `operatorById('laplaceBeltrami')` or `fieldById('scalarVertex')` returns null at runtime, the curated id is wrong — grep `src/operator_registry.cpp` / `src/field_registry.cpp` for the actual ids and update the test strings. The ids in CLAUDE.md are the source of truth.

- [ ] **Step 5: Commit**

```bash
git add bindings/node/src/addon.cpp scripts/_smoke-addon-registry.mjs nxr_compute_addon.node
git commit -m "feat(node): operatorInfo()/fieldInfo() registry metadata lookups

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `eigs(handle, opts)` async named-operator eigensolve

**Files:**
- Modify: `bindings/node/src/addon.cpp` — add `EigsWorker` class (near `EigenSolveWorker`, ≈ line 400) and the `Eigs` entry point; register in `Init()`.
- Modify: `bindings/node/index.mjs` — add `async eigs(opts)` to the structured context.
- Test: `scripts/_smoke-addon-eigs.mjs` (create)

**Interfaces:**
- Consumes: `getContext`, `matrixToFloat64Array`, `toFloat64Array`, `solve::eigenOperator(Manifold&, const EigenOperatorSpec&, int k, double sigma, bool normalize, bool reconstructMultiplets, bool dense)` → `EigenResult{eigenvectors,eigenvalues,k,nConverged}`. `EigenOperatorSpec{op,tau,mass}` with `EigenOperator{LaplacianCotan,LaplacianGraph,Dirac,DiracFace}`. `parseMassMatrixVariant(std::string)` (in scope via `using namespace nxr::manifold::ops;`). `nxr::core::errorCodeName(code)`.
- Produces: addon export `eigs(handle, opts)` → `Promise<{eigenvectors, eigenvalues, k, nConverged, blockSize}>`; structured `mctx.eigs(opts)`.

- [ ] **Step 1: Add the `EigsWorker` class**

Insert near `EigenSolveWorker` in `addon.cpp`:

```cpp
// AsyncWorker for the named-operator eigensolve. opts are parsed into an
// EigenOperatorSpec on the JS thread (in Eigs); this worker only runs the
// off-thread solve and marshals the result. Result shape matches WASM's
// eigs ({eigenvectors, eigenvalues, k, nConverged} + blockSize).
class EigsWorker : public Napi::AsyncWorker {
public:
    EigsWorker(Napi::Env env, ContextHolder* holder, solve_ns::EigenOperatorSpec spec,
               int k, double sigma, bool normalize, bool multiplets, bool dense,
               int blockSize, Napi::Value ctxVal)
        : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)),
          holder_(holder), spec_(spec), k_(k), sigma_(sigma),
          normalize_(normalize), multiplets_(multiplets), dense_(dense),
          blockSize_(blockSize) {
        ctxRef_ = Napi::Persistent(ctxVal);   // pin handle for worker lifetime
    }

    void Execute() override {
        try {
            result_ = solve_ns::eigenOperator(*holder_->manifold, spec_, k_, sigma_,
                                              normalize_, multiplets_, dense_);
        } catch (const nxr::core::Error& e) {
            errorCode_ = std::string(nxr::core::errorCodeName(e.code()));
            errorHint_ = std::string(e.hint());
            SetError(e.what());
        } catch (const std::exception& e) {
            errorCode_ = "INTERNAL_ERROR";
            SetError(e.what());
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        auto obj = Napi::Object::New(env);
        obj.Set("eigenvectors", matrixToFloat64Array(env, result_.eigenvectors));
        obj.Set("eigenvalues",  toFloat64Array(env, result_.eigenvalues));
        obj.Set("k",            Napi::Number::New(env, result_.k));
        obj.Set("nConverged",   Napi::Number::New(env, result_.nConverged));
        obj.Set("blockSize",    Napi::Number::New(env, blockSize_));
        deferred.Resolve(obj);
    }

    void OnError(const Napi::Error& e) override {
        Napi::Env env = Env();
        Napi::Error err = Napi::Error::New(env, e.Message());
        if (!errorCode_.empty()) err.Set("code", Napi::String::New(env, errorCode_));
        if (!errorHint_.empty()) err.Set("hint", Napi::String::New(env, errorHint_));
        deferred.Reject(err.Value());
    }

    Napi::Promise GetPromise() { return deferred.Promise(); }

private:
    Napi::Promise::Deferred deferred;
    Napi::Reference<Napi::Value> ctxRef_;
    ContextHolder* holder_;
    solve_ns::EigenOperatorSpec spec_;
    int k_; double sigma_; bool normalize_, multiplets_, dense_;
    int blockSize_;
    EigenResult result_;
    std::string errorCode_, errorHint_;
};
```

- [ ] **Step 2: Add the `Eigs` entry point**

```cpp
// ─── eigs(handle, opts) → Promise<{eigenvectors,eigenvalues,k,nConverged,blockSize}> ───
// Async (matches addon solve()/hodge()). Mirrors ContextWrapper::eigs opts
// parsing; the operator AND its natural generalized mass are assembled C++-side.
Napi::Value Eigs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto holder = getContext(info);
    if (!holder) return env.Null();
    // Reject (not throw) on validation error so the caller's await/.catch is uniform.
    auto fail = [&](const std::string& codeName, const std::string& msg, const std::string& hint) -> Napi::Value {
        auto d = Napi::Promise::Deferred::New(env);
        Napi::Error err = Napi::Error::New(env, "[" + codeName + "] " + msg);
        err.Set("code", Napi::String::New(env, codeName));
        if (!hint.empty()) err.Set("hint", Napi::String::New(env, hint));
        d.Reject(err.Value());
        return d.Promise();
    };
    try {
        if (info.Length() < 2 || !info[1].IsObject())
            return fail("INVALID_INPUT", "eigs(handle, opts): opts must be an object.", "");
        Napi::Object opts = info[1].As<Napi::Object>();
        if (!opts.Has("operator"))
            return fail("INVALID_INPUT", "eigs: expected { operator, k, ... }.", "");

        solve_ns::EigenOperatorSpec spec;
        int blockSize = 1;
        const std::string op = opts.Get("operator").As<Napi::String>().Utf8Value();
        if (op == "laplacian") {
            const std::string sub = opts.Has("subtype") ? opts.Get("subtype").As<Napi::String>().Utf8Value() : "";
            if      (sub == "cotan") spec.op = solve_ns::EigenOperator::LaplacianCotan;
            else if (sub == "graph") spec.op = solve_ns::EigenOperator::LaplacianGraph;
            else return fail("INVALID_INPUT", "eigs laplacian: subtype must be cotan|graph.", "");
        } else if (op == "cotan") {
            spec.op = solve_ns::EigenOperator::LaplacianCotan;
        } else if (op == "graph") {
            spec.op = solve_ns::EigenOperator::LaplacianGraph;
        } else if (op == "dirac") {
            spec.op = solve_ns::EigenOperator::Dirac; blockSize = 4;
        } else if (op == "diracFace") {
            spec.op = solve_ns::EigenOperator::DiracFace; blockSize = 4;
        } else {
            return fail("INVALID_INPUT", "eigs: operator must be laplacian|cotan|graph|dirac|diracFace.", "");
        }

        if (opts.Has("tau"))  spec.tau  = opts.Get("tau").As<Napi::Number>().DoubleValue();
        if (opts.Has("mass")) spec.mass = parseMassMatrixVariant(opts.Get("mass").As<Napi::String>().Utf8Value());
        if (!opts.Has("k")) return fail("INVALID_INPUT", "eigs: k is required.", "");

        const int    k          = opts.Get("k").As<Napi::Number>().Int32Value();
        const double sigma      = opts.Has("sigma")      ? opts.Get("sigma").As<Napi::Number>().DoubleValue() : -1e-8;
        const bool   normalize  = opts.Has("normalize")  ? opts.Get("normalize").As<Napi::Boolean>().Value()  : true;
        const bool   multiplets = opts.Has("multiplets") ? opts.Get("multiplets").As<Napi::Boolean>().Value() : false;
        const bool   dense      = opts.Has("dense")      ? opts.Get("dense").As<Napi::Boolean>().Value()      : false;

        auto* worker = new EigsWorker(env, holder, spec, k, sigma, normalize,
                                      multiplets, dense, blockSize, info[0]);
        worker->Queue();
        return worker->GetPromise();
    } catch (const nxr::core::Error& e) {
        return fail(std::string(nxr::core::errorCodeName(e.code())), e.what(), std::string(e.hint()));
    } catch (const std::exception& e) {
        return fail("INTERNAL_ERROR", e.what(), "");
    }
}
```

Register in `Init()`:

```cpp
    exports.Set("eigs", Napi::Function::New(env, Eigs));
```

- [ ] **Step 3: Wire the structured surface**

In `makeManifoldContext`'s returned object (right after the `operators` method added in Task 1):

```js
    /** Named-operator eigensolve (async — runs in a libuv worker, parity with
     *  WASM `manifold.eigs` but Promise-returning like addon `solve.eigen`).
     *  opts: { operator, subtype?, tau?, mass?, k, sigma?, normalize?,
     *  multiplets?, dense? } → { eigenvectors, eigenvalues, k, nConverged,
     *  blockSize }. */
    async eigs(opts) { return addon.eigs(rawCtx, opts) },
```

- [ ] **Step 4: Write the failing smoke test**

Create `scripts/_smoke-addon-eigs.mjs`:

```js
// Smoke test: addon eigs() async named-operator eigensolve.
// Usage (from repo root): node scripts/_smoke-addon-eigs.mjs
import { initNxrCompute } from '../bindings/node/index.mjs'

const nxr = await initNxrCompute()
const t = (1 + Math.sqrt(5)) / 2
const raw = [
  -1, t, 0,   1, t, 0,  -1, -t, 0,   1, -t, 0,
   0, -1, t,  0, 1, t,   0, -1, -t,  0, 1, -t,
   t, 0, -1,  t, 0, 1,  -t, 0, -1,  -t, 0, 1,
]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) {
  const len = Math.hypot(raw[i], raw[i + 1], raw[i + 2])
  verts[i] = raw[i] / len; verts[i + 1] = raw[i + 1] / len; verts[i + 2] = raw[i + 2] / len
}
const faces = new Int32Array([
  0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
  3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
])

const m = nxr.createManifoldContext(verts, faces)
let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

// cotan Laplacian: smallest eigenvalue ~0 (constant mode), blockSize 1
const lap = await m.eigs({ operator: 'cotan', k: 6 })
check(lap.eigenvalues.length === 6, 'cotan eigs returns 6 eigenvalues')
check(lap.blockSize === 1, 'cotan blockSize is 1')
check(Math.abs(lap.eigenvalues[0]) < 1e-6, 'cotan smallest eigenvalue ~0')
check(lap.eigenvectors.length === 12 * lap.k, 'cotan eigenvectors are vMajor [12×k]')

// Dirac: blockSize 4, constant mode is a 4-fold zero (with multiplets)
const dir = await m.eigs({ operator: 'dirac', tau: 0.5, k: 8, multiplets: true })
check(dir.blockSize === 4, 'dirac blockSize is 4')
check(dir.eigenvalues.length === 8, 'dirac eigs returns 8 eigenvalues')

// Validation error rejects with INVALID_INPUT (no k)
let rejected = false
try { await m.eigs({ operator: 'cotan' }) } catch (e) { rejected = true; check(e.code === 'INVALID_INPUT', 'missing k rejects INVALID_INPUT') }
check(rejected, 'missing k rejects the promise')

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
```

- [ ] **Step 5: Build, then run to verify it passes**

```bash
bash scripts/build.sh Release && node scripts/_smoke-addon-eigs.mjs
```

Expected: all `✓`, ending `PASS`. (Before Steps 1-3: `m.eigs is not a function`.)

- [ ] **Step 6: Commit**

```bash
git add bindings/node/src/addon.cpp bindings/node/index.mjs scripts/_smoke-addon-eigs.mjs nxr_compute_addon.node
git commit -m "feat(node): async eigs() named-operator eigensolve (EigsWorker)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Cross-binding byte-parity test

**Files:**
- Test: `bindings/node/test/test_parity_wasm.mjs` (create)

**Interfaces:**
- Consumes: the addon public surface (`nxr.createManifoldContext`, `nxr.operatorInfo`, `nxr.fieldInfo`) from `bindings/node/index.mjs`; the WASM module at `build_wasm/nxr_compute.js` (constructed `new wasm.Manifold(verts, faces)` with methods `operators`, `eigs`, `frames`, and module-level `operatorInfo`/`fieldInfo`).
- Produces: a Node script asserting addon == WASM (max abs err < 1e-12 for sparse data, < 1e-9 for eigenvalues). Skips the cross-binding comparisons (running only a standalone guard) if the WASM build is absent.

- [ ] **Step 1: Write the parity test**

Create `bindings/node/test/test_parity_wasm.mjs`:

```js
// Cross-binding byte-parity: the addon must match the WASM Manifold on the
// registry/operator surface. Run from repo root:
//   bash scripts/build.sh Release && bash scripts/build-wasm.sh
//   node bindings/node/test/test_parity_wasm.mjs
import path from 'node:path'
import fs from 'node:fs'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { initNxrCompute } from '../index.mjs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot  = path.resolve(__dirname, '..', '..', '..')

let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

// ── Shared icosahedron fixture ──
const t = (1 + Math.sqrt(5)) / 2
const rawV = [
  -1, t, 0,   1, t, 0,  -1, -t, 0,   1, -t, 0,
   0, -1, t,  0, 1, t,   0, -1, -t,  0, 1, -t,
   t, 0, -1,  t, 0, 1,  -t, 0, -1,  -t, 0, 1,
]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) {
  const len = Math.hypot(rawV[i], rawV[i + 1], rawV[i + 2])
  verts[i] = rawV[i] / len; verts[i + 1] = rawV[i + 1] / len; verts[i + 2] = rawV[i + 2] / len
}
const faces = new Int32Array([
  0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
  3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
])

// Sort a COO's entries by (row,col) into a canonical key→value map.
const cooMap = (coo, field) => {
  const map = new Map()
  for (let i = 0; i < coo[field].length; i++) map.set(coo.row[i] * 100000 + coo.col[i], coo[field][i])
  return map
}
const cooEqual = (a, b, field, tol, label) => {
  if (a.rows !== b.rows || a.cols !== b.cols || a[field].length !== b[field].length) {
    check(false, `${label}: shape/nnz mismatch (addon ${a.rows}×${a.cols} nnz ${a[field].length} vs wasm ${b.rows}×${b.cols} nnz ${b[field].length})`)
    return
  }
  const ma = cooMap(a, field), mb = cooMap(b, field)
  let maxErr = 0
  for (const [k, va] of ma) maxErr = Math.max(maxErr, Math.abs(va - (mb.get(k) ?? Infinity)))
  check(maxErr < tol, `${label}: addon == wasm (maxErr ${maxErr.toExponential(2)})`)
}

const nxr = await initNxrCompute()
const A = nxr.createManifoldContext(verts, faces)

// ── WASM module (skip cross-binding if absent) ──
const wasmJs = path.join(repoRoot, 'build_wasm', 'nxr_compute.js')
if (!fs.existsSync(wasmJs)) {
  console.warn(`[parity] WASM build not found at ${wasmJs} — run scripts/build-wasm.sh.`)
  console.warn('[parity] Skipping cross-binding comparison (build the WASM target to enable it).')
  // Standalone guard so the test still has signal: dirac(0) == kron(cotanL, I4).
  const cotan = A.operators('laplacian', 'cotan')
  const dirac0 = A.operators('dirac', 0)
  const dmap = new Map()
  for (let i = 0; i < dirac0.nnz; i++) dmap.set(dirac0.row[i] * 100000 + dirac0.col[i], dirac0.data[i])
  let maxErr = 0
  for (let i = 0; i < cotan.nnz; i++)
    for (let b = 0; b < 4; b++)
      maxErr = Math.max(maxErr, Math.abs((dmap.get((4*cotan.row[i]+b)*100000 + (4*cotan.col[i]+b)) ?? 0) - cotan.data[i]))
  check(maxErr < 1e-12, `standalone: dirac(0) == kron(cotanL, I4) (maxErr ${maxErr.toExponential(2)})`)
  console.log(failures === 0 ? '\nPASS (standalone only)' : `\nFAIL (${failures})`)
  process.exit(failures === 0 ? 0 : 1)
}

const { default: createWasm } = await import(pathToFileURL(wasmJs).href)
const wasm = await createWasm()
const B = new wasm.Manifold(verts, faces)

// ── operators: real families ──
for (const [family, sub] of [
  ['laplacian', 'cotan'], ['laplacian', 'graph'], ['laplacian', 'covariant'],
  ['mass', 'lumped'], ['mass', 'galerkin'],
  ['hodge', 'h0'], ['hodge', 'h1'], ['hodge', 'h2'], ['hodge', 'h1inv'],
]) cooEqual(A.operators(family, sub), B.operators(family, sub), 'data', 1e-12, `operators ${family} ${sub}`)

for (const family of ['gradient3D', 'diracD', 'diracFaceD', 'gradFace', 'lapFace', 'extrinsicWeitzenbock'])
  cooEqual(A.operators(family), B.operators(family), 'data', 1e-12, `operators ${family}`)

for (const tau of [0, 0.5, 1]) {
  cooEqual(A.operators('dirac', tau), B.operators('dirac', tau), 'data', 1e-12, `operators dirac ${tau}`)
  cooEqual(A.operators('diracFace', tau), B.operators('diracFace', tau), 'data', 1e-12, `operators diracFace ${tau}`)
}

// ── operators: complex families ──
const cc = A.operators('laplacian', 'connection'), cw = B.operators('laplacian', 'connection')
cooEqual(cc, cw, 'realData', 1e-12, 'connection L (real part)')
cooEqual(cc, cw, 'imagData', 1e-12, 'connection L (imag part)')
for (const nSym of [1, 2]) {
  const ga = A.operators('connectionGradient', nSym), gw = B.operators('connectionGradient', nSym)
  cooEqual(ga, gw, 'realData', 1e-12, `connectionGradient ${nSym} (real)`)
  cooEqual(ga, gw, 'imagData', 1e-12, `connectionGradient ${nSym} (imag)`)
}

// ── dec ──
const da = A.operators('dec'), db = B.operators('dec')
cooEqual(da.d0, db.d0, 'data', 1e-12, 'dec d0')
cooEqual(da.d1, db.d1, 'data', 1e-12, 'dec d1')

// ── frames ──
const fa = A.measure.frame(), fb = B.frames()
for (const f of ['e1', 'e2', 'normals']) {
  let maxErr = 0
  for (let i = 0; i < fa[f].length; i++) maxErr = Math.max(maxErr, Math.abs(fa[f][i] - fb[f][i]))
  check(maxErr < 1e-12, `frames ${f}: addon == wasm (maxErr ${maxErr.toExponential(2)})`)
}

// ── eigs (spectrum only — eigenvectors are sign/gauge-ambiguous) ──
for (const spec of [
  { operator: 'cotan', k: 6 },
  { operator: 'graph', k: 6 },
  { operator: 'dirac', tau: 0.5, k: 8 },
  { operator: 'diracFace', tau: 0.5, k: 8 },
]) {
  const ea = await A.eigs(spec)
  const eb = B.eigs(spec)   // WASM eigs is synchronous
  check(ea.k === eb.k && ea.blockSize === eb.blockSize, `eigs ${spec.operator}: k/blockSize match`)
  let maxErr = 0
  for (let i = 0; i < ea.eigenvalues.length; i++) maxErr = Math.max(maxErr, Math.abs(ea.eigenvalues[i] - eb.eigenvalues[i]))
  check(maxErr < 1e-9, `eigs ${spec.operator}: eigenvalues match (maxErr ${maxErr.toExponential(2)})`)
}

// ── operatorInfo / fieldInfo string parity ──
for (const id of ['laplaceBeltrami', 'relativeDirac']) {
  const oa = nxr.operatorInfo(id), ob = wasm.operatorInfo(id)
  check(JSON.stringify(oa) === JSON.stringify(ob), `operatorInfo('${id}') string-identical`)
}
const fa2 = nxr.fieldInfo('scalarVertex'), fb2 = wasm.fieldInfo('scalarVertex')
check(JSON.stringify(fa2) === JSON.stringify(fb2), `fieldInfo('scalarVertex') string-identical`)

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
```

- [ ] **Step 2: Build both bindings and run**

```bash
bash scripts/build.sh Release && bash scripts/build-wasm.sh && node bindings/node/test/test_parity_wasm.mjs
```

Expected: all `✓`, ending `PASS`. If the WASM toolchain is unavailable, the test prints the skip message and runs the standalone `dirac(0)` guard only, still ending `PASS`.

> If any `operatorInfo` JSON comparison fails, diff the two objects — a type mismatch (e.g. addon emits `graded` as a string where WASM emits boolean) is the likely cause; fix the marshaling in Task 3's entry point to match WASM's JS types.

- [ ] **Step 3: Commit**

```bash
git add bindings/node/test/test_parity_wasm.mjs
git commit -m "test(node): cross-binding byte-parity test (addon vs WASM operators/eigs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `bindings/node/../../CLAUDE.md` (repo-root `CLAUDE.md`)

**Interfaces:**
- Consumes: nothing. Produces: corrected documentation.

- [ ] **Step 1: Update the `operators` table row**

In `CLAUDE.md`, find the sentence in the MEX `operators` command row:

```
The N-API addon does **not** expose this surface.
```

Replace with:

```
The N-API addon now exposes the same surface (`bindings/node/index.mjs` → `nxr.operators` / `manifoldContext.operators`), with one intentional difference: the addon's named-operator eigensolve `eigs` is **async** (returns a Promise, matching the addon's own `solve()`), whereas WASM's `eigs` is synchronous.
```

- [ ] **Step 2: Add a parity note for `operatorInfo` / `fieldInfo`**

Find the line documenting `operatorInfo` exposure:

```
Exposed to consumers as `nxr_compute('operatorInfo', id)` (MEX) and
`manifold.operatorInfo(id)` (WASM) → a struct/object of the metadata strings.
```

Replace with:

```
Exposed to consumers as `nxr_compute('operatorInfo', id)` (MEX),
`manifold.operatorInfo(id)` (WASM), and `nxr.operatorInfo(id)` (N-API addon)
→ a struct/object of the metadata strings.
```

Make the analogous edit to the `fieldInfo` exposure sentence (add `and `nxr.fieldInfo(id)` (N-API addon)`).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: N-API addon reaches WASM parity on operator/registry surface

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- `operators` → Task 1 ✓
- `eigs` (async) → Task 4 ✓
- `frames` → Task 2 ✓
- `operatorInfo`/`fieldInfo` → Task 3 ✓
- `nnz` parity field → Task 1 ✓
- index.mjs flat (`...addon`, automatic) + structured wiring → Tasks 1, 2, 4 ✓
- Cross-binding byte-parity test → Task 5 ✓
- Standalone `dirac(0)==cotanL⊗I₄` guard → Tasks 1 and 5 ✓
- CLAUDE.md update → Task 6 ✓
- Error contract (`.code`/`.hint`, sync throw vs promise reject) → `nxrSyncCall` reuse (Tasks 1-3), `fail`/`OnError` (Task 4) ✓

**Type consistency:** `EigenOperatorSpec`/`EigenOperator`/`parseMassMatrixVariant`/`eigenOperator` signatures match `include/nxr/compute.h`. `OperatorVariant`/`FieldVariant`/`FieldDescriptor` field names and types match `include/nxr/operator_registry.h` / `field_registry.h` (`graded` bool → `Napi::Boolean`; `nSym` int → `Napi::Number`; everything else string/enum→string). COO field names (`row/col/data/realData/imagData/rows/cols/nnz`) match the addon's helpers and WASM's `sparseToVal`. `eigs` result keys (`eigenvectors/eigenvalues/k/nConverged/blockSize`) match WASM's `eigenResultToVal` + `blockSize`. `solve_ns` alias is defined at addon.cpp top (`namespace solve_ns = nxr::manifold::solve;`).

**Placeholder scan:** none — every step has full code or an exact command.

**Note on registry ids:** `laplaceBeltrami`, `relativeDirac`, `scalarVertex` are used in test assertions; if any returns null at runtime, the curated id changed — grep the registry source and update. Flagged inline in Tasks 3 and 5.
