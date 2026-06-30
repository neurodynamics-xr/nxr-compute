# Node addon → WASM parity: registry / named-operator surface

**Date:** 2026-06-30
**Status:** Design approved, pending spec review
**Scope:** Bring the N-API addon (`bindings/node/`) to parity with the WASM
`Manifold` for the operator-registry / named-operator surface.

---

## Motivation

Since the May-2026 namespace refactor (`7bfbadc`), a large body of operator
work landed in the **WASM and MEX** bindings but never in the **N-API addon**:
the operator-registry surface, named native-sparse operators, the
named-operator eigensolve, and the registry/field metadata lookups. CLAUDE.md
records this twice as a deliberate gap ("The N-API addon does **not** expose
this surface."). This design closes that gap.

The addon already sits on the modern `Manifold` API and already carries every
marshaling primitive needed (`sparseToCOO`, `sparseComplexToCOO`,
`matrixToFloat64Array`, the `EigenResult` flatten in `EigenSolveWorker::OnOK`).
`index.mjs` already has `notWired()` placeholders anticipating this work. So the
change is **purely additive** — no library changes, no changes to existing
exports' behavior.

## Out of scope

The broader (non-registry) WASM-only methods — `normalize`, `removeDC`,
`precompute`, `tracePath`, `bff`, `randomDecomposed1Form` — are **not** part of
this work. They are general method parity, unrelated to the registry surface,
and can be a separate follow-on.

---

## The five methods

All five mirror the WASM `ContextWrapper` / free-function implementations
verbatim — same dispatch, same arg rules, same error messages, same output
shapes — so a consumer gets byte-identical results from either binding.

| New addon export | WASM source | Signature | Returns |
|---|---|---|---|
| `operators(ctx, family, arg)` | `ContextWrapper::operators` (wasm L304) | `family: string`, `arg: string \| number \| undefined` | real COO, complex COO, or `{d0,d1}` |
| `eigs(ctx, opts)` → **Promise** | `ContextWrapper::eigs` (wasm L387) | `opts: object` | `{eigenvectors, eigenvalues, k, nConverged, blockSize}` |
| `frames(ctx)` | `ContextWrapper::frames` (wasm L435) | — | `{e1, e2, normals}` |
| `operatorInfo(id)` | `operatorInfoJS` (wasm L1102) | `id: string` (**no handle**) | metadata object |
| `fieldInfo(id)` | `fieldInfoJS` (wasm L1142) | `id: string` (**no handle**) | metadata object |

### `operators(ctx, family, arg)`

Synchronous. Dispatches on `family` exactly as WASM does:

```
laplacian   → cotan | graph | connection(complex) | covariant(Ambient)
mass        → lumped | galerkin
hodge       → h0 | h1 | h2 | h1inv
dec         → { d0, d1 }
gradient3D  → (no arg)
dirac       → numeric τ (required)
diracFace   → numeric τ (required)
diracD | diracFaceD | diracIntrinsicD | diracFaceIntrinsicD | gradFace | lapFace → (no arg)
connectionGradient → numeric nSym (default 1), complex COO
extrinsicWeitzenbock → (no arg), real [3V×3V]
```

`arg` may be a string (subtype), a number (τ / nSym), or undefined.
Marshaling:
- real sparse → `sparseToCOO` → `{row, col, data, rows, cols, nnz}`
- complex sparse (`connection`, `connectionGradient`) → `sparseComplexToCOO`
  → `{row, col, realData, imagData, rows, cols, nnz}`
- `dec` → `{ d0: COO, d1: COO }`

Unknown family/subtype, or a missing required τ, throws the **same** error
message string WASM uses (`nxr::core::Error`, `InvalidInput`).

### `eigs(ctx, opts)` — async

Returns a **Promise** (decision: async, matching the addon's own
`solve()`/`hodge()` per CLAUDE.md rule 7, even though WASM's `eigs` is sync).
A new `EigsWorker` AsyncWorker, modeled on `EigenSolveWorker` (addon.cpp:430):

1. **On the JS thread** (in the `Eigs` entry point): parse `opts` into a
   `nxr::manifold::solve::EigenOperatorSpec` using the same logic as WASM
   L390–423 — `operator` (`laplacian|cotan|graph|dirac|diracFace`), optional
   `subtype`, `tau`, `mass` (`parseMassMatrixVariant`), required `k`, optional
   `sigma` (= −1e-8), `normalize` (= true), `multiplets` (= false),
   `dense` (= false). Compute `blockSize` (4 for dirac/diracFace, else 1).
   Pin the ctx with a persistent reference.
2. **In `Execute()`** (worker thread): call
   `solve::eigenOperator(*holder_->manifold, spec, k, sigma, normalize,
   multiplets, dense)` → `EigenResult`.
3. **In `OnOK()`:** resolve with
   `{ eigenvectors (vMajor row-major), eigenvalues, k, nConverged, blockSize }`
   — matching WASM's `eigenResultToVal` + the `blockSize` field.
4. **In `OnError()`:** reject with a JS `Error` carrying `.code` (enumerator
   name) and `.hint`, reusing the existing structured-error path.

Note: unlike the existing `solve()` result, `eigs` does **not** emit `nV`
(strict WASM parity — `nV` is trivially recoverable as
`eigenvectors.length / k` and is omitted to keep the two bindings' `eigs`
shapes identical).

`eigs` does not plumb cancel/progress in v1 (WASM's `eigs` doesn't either);
the named-operator solve completes in seconds on cortical meshes and the async
wrapping alone keeps the event loop responsive.

### `frames(ctx)`

Synchronous. `geometry::frames(*manifold)` → `{e1, e2, normals}`, each a
row-major `Float64Array` via `matrixToFloat64Array`.

### `operatorInfo(id)` / `fieldInfo(id)`

Synchronous, **handle-free** top-level exports (they touch only the static
registry, no `Manifold`). Each builds a plain JS object with the exact field
names WASM/MEX use, so downstream consumers work identically against any
binding. Unknown id throws `InvalidInput`. Field sets:

- `operatorInfo`: `id, label, bundle, holonomy, order, role, field_type,
  domain, singular, gauge, coupling, natural_mass, graded, tau_presets,
  status, notes, squares_to, square_of, relation, input_field, output_field`
- `fieldInfo`: `id, label, domain, bundle, field_type, n_form, representation,
  gauge, nSym, notes`

---

## Supporting change: `sparseToCOO` gains `nnz`

WASM's `sparseToVal` emits an `nnz` field; the addon's `sparseToCOO` /
`sparseComplexToCOO` currently do not. Add `nnz` to both for byte-parity.
Additive — existing consumers ignore the extra field.

---

## `index.mjs` wiring

- **Flat surface** (raw re-export): add `operators`, `eigs`, `frames`,
  `operatorInfo`, `fieldInfo`.
- **Structured surface:**
  - replace `notWired('measure.frame')` with a real `frames()` call;
  - add an `operators(family, arg)` accessor and an `eigs(opts)` method on the
    manifold context object;
  - expose `operatorInfo` / `fieldInfo` at module level (handle-free).
- Remove the now-satisfied `[NOT_WIRED_IN_ADDON]` throw for `measure.frame`.
  (The hodge `star0/star2/star1Inverse` and `uv.bff` stubs stay — out of
  scope.)

---

## Error handling

Unchanged idiom: catch `nxr::core::Error`, throw a JS `Error` with `.code`
(the enumerator name, e.g. `"INVALID_INPUT"`) and `.hint`. `operators`,
`frames`, `operatorInfo`, `fieldInfo` throw synchronously; `eigs` rejects its
Promise via `OnError`.

---

## Testing — cross-binding byte parity

New `bindings/node/test/test_parity_wasm.mjs`:

1. Load one mesh (icosphere fixture) into **both** the addon and the WASM
   module.
2. For each `operators` family/subtype (including complex `connection` /
   `connectionGradient` and the `{d0,d1}` dec case), assert addon output ==
   WASM output (sorted-COO compare, max abs err < 1e-12 on `data` /
   `realData` / `imagData`, exact match on `row`/`col`/`rows`/`cols`/`nnz`).
3. For `eigs` on `cotan`, `graph`, `dirac`, `diracFace`: assert eigenvalues
   match (< 1e-9; eigenvectors are gauge/sign-ambiguous so compare the
   spectrum, not raw vectors) and `k`/`nConverged`/`blockSize` match.
4. `frames`: assert `e1`/`e2`/`normals` match (< 1e-12).
5. `operatorInfo` / `fieldInfo`: spot-check exact string equality on a few ids.

Plus a standalone invariant guard that does **not** depend on WASM, so the
test still has signal if the WASM build is stale: assert
`operators('dirac', 0)` equals `kron(cotanLaplacian, I₄)` (the documented τ=0
anchor).

The test runs under Node and requires the WASM module to be built
(`scripts/build-wasm.sh`); it skips the cross-binding comparisons with a clear
message (and runs only the standalone invariant) if the WASM artifact is
absent.

---

## CLAUDE.md update

Two passages currently state the addon does not expose this surface:

1. The `operators` row of the MEX command table: *"The N-API addon does **not**
   expose this surface."*
2. (Same sentiment in the WASM-parity note.)

Update both to state the addon now reaches parity for the registry/named-
operator surface, noting the one intentional difference: the addon's `eigs` is
**async** (returns a Promise) whereas WASM's is synchronous. Mention
`operatorInfo`/`fieldInfo` are now exposed by all three of MEX, WASM, and the
N-API addon.

---

## Files touched

- `bindings/node/src/addon.cpp` — 5 new entry points + `EigsWorker`; `nnz`
  added to the two COO helpers.
- `bindings/node/index.mjs` — flat + structured wiring; drop one `notWired`.
- `bindings/node/test/test_parity_wasm.mjs` — new cross-binding parity test.
- `CLAUDE.md` — parity note.
