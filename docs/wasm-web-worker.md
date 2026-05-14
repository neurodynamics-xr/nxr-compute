# Hosting nxr-compute WASM in a Web Worker

The WASM build of nxr-compute is single-threaded by design — every call
runs on whichever JS thread loaded the module. On the main thread that
means a 1–3 s `solveEigenmodes` or `hodgeDecompose` on a cortical mesh
will block the UI. The browser-side equivalent of the Node addon's
`Napi::AsyncWorker` is a **Web Worker**: instantiate the WASM module
inside the worker and dispatch heavy calls through `postMessage`.

nxr-compute deliberately does **not** ship a built-in worker sidecar.
Bundler URL resolution, transferable choices, and the progress-channel
shape are application concerns — what your app sends to the worker and
how it surfaces results is up to you. This page documents the pattern
and points at the copy-and-modify template the repo ships.

---

## Files

| Path | Purpose |
|---|---|
| `bindings/wasm/js/nxr_compute_wasm.worker.template.mjs` | Minimal worker script. Receives `{ id, method, args }`, dispatches into the same six-group `nxr.manifold.*` surface as the in-process API, returns `{ id, result }` or `{ id, error }`. Use as-is or modify. |
| `bindings/wasm/js/index.mjs` | The in-process loader the worker imports. Returns the same `nxr.createManifoldContext(...)` you'd use directly on the main thread. |

---

## Main-thread RPC wrapper (sketch)

```js
// Spawn the worker once per ComputeContext you want hosted off-thread.
const worker = new Worker(
  new URL('./nxr_compute_wasm.worker.template.mjs', import.meta.url),
  { type: 'module' },
)

let nextId = 0
const pending = new Map()

worker.onmessage = (e) => {
  const { id, ok, result, error } = e.data
  const slot = pending.get(id)
  if (!slot) return
  pending.delete(id)
  if (error) {
    const err = new Error(error.message)
    err.code = error.code
    err.hint = error.hint
    slot.reject(err)
  } else {
    slot.resolve(ok ? true : result)
  }
}

function rpc(method, args = [], extra = {}) {
  const id = ++nextId
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    worker.postMessage({ id, method, args, ...extra })
  })
}

// Initialise once per mesh.
await rpc('init', [], { verts, faces })

// Long-running calls — main thread stays responsive.
const eig    = await rpc('solve.eigen', [300])
const hodge  = await rpc('solve.hodge', [/* optional omega */])
const dist   = await rpc('measure.distance', [[0, 1, 2]])
```

---

## Method paths

The template walks a dot-separated `method` string into the
ManifoldContext object. Anything in the in-process six-group surface is
addressable:

```
solve.eigen(k)              → 'solve.eigen', args: [k]
solve.poisson(verts, vals)  → 'solve.poisson', args: [verts, vals]
solve.hodge(omega?)         → 'solve.hodge', args: [omega]

operator.d0()               → 'operator.d0', args: []
operator.cotanLaplacian()   → 'operator.cotanLaplacian', args: []
operator.connectionLaplacian(opts) → 'operator.connectionLaplacian', args: [opts]

measure.curvature()         → 'measure.curvature', args: []
measure.distance(verts)     → 'measure.distance', args: [[0, 1, 2]]
measure.distance.signed(curve, isLoop, level) → 'measure.distance.signed', args: [curve, isLoop, level]
measure.normal(type)        → 'measure.normal', args: [type]

interpolate.transport(verts, vecs)      → 'interpolate.transport', args: [verts, vecs]
interpolate.directionField(verts, vals) → 'interpolate.directionField', args: [verts, vals]
interpolate.smoothFaceField(nSym)       → 'interpolate.smoothFaceField', args: [nSym]
```

See `bindings/wasm/js/index.d.ts` for the complete surface.

---

## Transferables

The template auto-collects `ArrayBuffer` references from the result and
posts them as transferable. That means a `Float64Array` returned from
the worker is **moved** to the main thread, not copied — zero memcpy
on the largest payloads (eigenvectors at V·K doubles can be hundreds of
MB). The collection heuristic skips views into the wasm HEAP itself
(those would detach the entire wasm linear memory and break the
worker). For nxr-compute's typed-array outputs, which own their own
ArrayBuffers, this works automatically.

---

## Cancellation + progress

The WASM binding accepts cancel/progress as `SharedArrayBuffer`-backed
`Int32Array` heap pointers (CLAUDE.md §12). The same mechanism works
across worker boundaries — allocate a `SharedArrayBuffer` on the main
thread, pass its pointer (computed via the worker's `Module._malloc`)
both ways. The template doesn't bake this in because the worker would
need access to the same WASM module instance the main thread allocated
from, which is awkward. The simplest path:

1. Worker does its own `_malloc(4)` for a cancel cell during `init`,
   returns the address to the main thread along with a handle to its
   `HEAP32` view (via `postMessage` of a `SharedArrayBuffer` snapshot).
2. Main thread writes `1` to the cancel cell with `Atomics.store` from
   anywhere; the worker's WASM polls it via the existing C++ contract.

This adds ~30 lines to the template; tag a TODO in your copy if you
need it.

---

## When NOT to use a worker

- Calls that complete in <50 ms: the postMessage round-trip cost dominates.
  Examples: `operator.d0()` (returns a cached sparse COO), `query.*`,
  `measure.curvature()` (geometry-central caches once per geometry).
- One-shot batch pipelines where the main thread can `await` everything
  serially: the JS thread blocking is acceptable if no UI rendering or
  event handling needs to happen meanwhile.

Reserve the worker for **interactive UIs** where >100 ms of blocking
would drop frames. `solveEigenmodes`, `hodgeDecompose`,
`computeDirectionField`, and cold-cache `computeSmooth*Field` are the
clear candidates.
