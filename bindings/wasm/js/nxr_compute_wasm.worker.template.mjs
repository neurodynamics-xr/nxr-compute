// nxr_compute_wasm.worker.template.mjs
//
// Copy-and-modify starting point for hosting nxr-compute WASM inside a
// Web Worker. The WASM runtime is single-threaded by design; running it
// in a Worker is how you keep heavy ops (solveEigenmodes, hodgeDecompose,
// computeDirectionField) off the main JS thread.
//
// nxr-compute deliberately does NOT ship a built-in worker sidecar —
// bundler URL resolution, transferable choices, and progress-channel
// shape are application concerns, not engine concerns. The template
// below is intentionally minimal; copy it into your app's worker
// folder, point the import at your bundle's WASM URL, and add
// whatever message shape your app prefers.
//
// Protocol implemented here:
//
//   main → worker  { id, method: 'init', verts, faces }
//   worker → main  { id, ok: true }                                    // init complete
//
//   main → worker  { id, method: 'solve.eigen', args: [300] }
//   worker → main  { id, result: { eigenvalues, eigenvectors, k } }
//
//   main → worker  { id, method: 'measure.distance', args: [[0]] }
//   worker → main  { id, result: <Float64Array> }
//
//   any failure   { id, error: { message, code, hint } }
//
// The dispatcher walks dot-paths into the same six-group surface the
// in-process consumer would use (mctx.solve.eigen, mctx.operator.d0,
// mctx.measure.distance, etc.) — see `bindings/wasm/js/index.mjs`.
// Refer to `docs/wasm-web-worker.md` for the main-thread RPC wrapper
// pattern.

import { initNxrCompute } from './index.mjs'  // ← repoint at your bundle's nxr-compute import

let mctx = null   // ManifoldContext, populated on 'init'

self.onmessage = async (e) => {
  const { id, method, args = [], verts, faces } = e.data
  try {
    if (method === 'init') {
      const nxr = await initNxrCompute()
      mctx = nxr.createManifoldContext(verts, faces)
      self.postMessage({ id, ok: true })
      return
    }
    if (method === 'dispose') {
      // ManifoldContext doesn't expose an explicit dispose today; the
      // underlying Embind handle is released when GC collects mctx.
      // For tighter control, hold a ref to mctx._raw and call .delete().
      mctx = null
      self.postMessage({ id, ok: true })
      return
    }

    if (!mctx) {
      throw new Error('worker not initialised; send { method: "init", verts, faces } first')
    }

    // Walk a dot-path like 'solve.eigen' or 'measure.distance.signed'.
    // The final leaf is a function; everything before it is a group object.
    const parts = method.split('.')
    let target = mctx
    let host   = mctx
    for (let i = 0; i < parts.length; i++) {
      host = target
      target = target?.[parts[i]]
      if (target === undefined) {
        throw new Error(`unknown method: ${method}`)
      }
    }
    if (typeof target !== 'function') {
      throw new Error(`not a function: ${method}`)
    }

    // Await even sync returns — handles both Promise-returning (eigen,
    // hodge, directionField) and immediate-returning (operator.d0,
    // measure.curvature) leaves uniformly.
    const result = await target.apply(host, args)

    // Collect Transferable typed-array buffers from the result so we
    // hand JS ownership to the main thread without copying. Skips
    // ArrayBuffers that are aliased into the wasm heap (HEAP*) — those
    // would detach the wasm memory and break the worker.
    const transferables = collectTransferables(result)
    self.postMessage({ id, result }, transferables)
  } catch (err) {
    self.postMessage({
      id,
      error: {
        message: err?.message ?? String(err),
        code:    err?.code,
        hint:    err?.hint,
      },
    })
  }
}

// Best-effort transferable collection. Walks a plain-object/array
// tree and gathers .buffer references for any TypedArray it finds.
// Skips views into the wasm HEAP* (transferring those would detach
// the entire wasm linear memory).
function collectTransferables(value, out = []) {
  if (!value || typeof value !== 'object') return out
  if (ArrayBuffer.isView(value)) {
    const buf = value.buffer
    // Heuristic: HEAP* views are wide (the whole wasm heap, hundreds
    // of MB) and have byteOffset; the typed arrays nxr-compute returns
    // are bounded slices that own their own ArrayBuffer.
    if (buf && value.byteOffset === 0 && buf.byteLength === value.byteLength) {
      out.push(buf)
    }
    return out
  }
  if (Array.isArray(value)) {
    for (const v of value) collectTransferables(v, out)
    return out
  }
  for (const k in value) collectTransferables(value[k], out)
  return out
}
