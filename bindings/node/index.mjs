// nxr-compute — N-API addon entry point.
//
// Wraps the cmake-js-built addon (`nxr_compute_addon.node` at the
// project root) and exposes two surfaces in parallel:
//
//   1. Flat surface (legacy, mirrors the addon's exports verbatim):
//        const nxr = await initNxrCompute()
//        const ctx = nxr.createContext(verts, faces)
//        nxr.solveEigenmodes(ctx, 300)
//
//   2. Six-group nested namespace `nxr.manifold.*` (preferred for new code):
//        const mctx = nxr.createManifoldContext(verts, faces)
//        mctx.solve.eigen(300)
//        mctx.operator.connectionLaplacian({ domain: 'vertex', nSym: 4 })
//        mctx.measure.distance([0])
//        mctx.measure.distance.signed([0, 1, 2], true)
//
//      Or in functional form:
//        nxr.nxr.manifold.solve.eigen(mctx, 300)
//
// Same convention as the WASM binding's `bindings/wasm/js/index.mjs`.
// Both shells share the same conceptual surface; this shim makes the
// addon feel identical to the WASM module from a consumer's
// perspective. The underlying compute kernels are the same code — see
// `include/nxr/compute.h` and the per-binding glue layers.
//
// Surface delta vs WASM (the addon doesn't yet expose every embind
// method): `measure.frame` (computeFaceFrames), `uv.bff`
// (computeUVCoordinates), `operator.star0` / `star2` /
// `star1Inverse`, and a few normalize/removeDC helpers throw a stable
// `Error` with code `NOT_WIRED_IN_ADDON` so consumers get a clear
// signal rather than silent undefined behaviour. To unblock these,
// add the corresponding `exports.Set(...)` line in
// `bindings/node/src/addon.cpp` and the body delegation will start
// working without further shim changes.

import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Default search path for the addon binary. cmake-js places the
// .node file under build/Release/ (or build/Debug/); scripts/build.sh
// copies it to the project root for the smoke scripts. Try the root
// first, then the build directories. Consumers can override with
// `process.env.NXR_COMPUTE_ADDON_PATH` to load a specific binary.
const repoRoot = path.resolve(__dirname, '..', '..')
function locateAddon() {
  const explicit = process.env.NXR_COMPUTE_ADDON_PATH
  if (explicit) return explicit
  return path.join(repoRoot, 'nxr_compute_addon.node')
}

const addon = require(locateAddon())

/* ---------------------------------------------------------------------- */
/*                         Argument coercions                             */
/* ---------------------------------------------------------------------- */
//
// The addon only accepts strict TypedArray inputs (Int32Array,
// Float64Array). Plain number[] is a common JS shape, especially
// inline in tests, so coerce on the way in.

const asI32 = (arr) => arr instanceof Int32Array
  ? arr
  : Int32Array.from(arr)

const asF64 = (arr) => arr instanceof Float64Array
  ? arr
  : Float64Array.from(arr)

/* ---------------------------------------------------------------------- */
/*                      Stub helpers — uniform errors                     */
/* ---------------------------------------------------------------------- */
//
// Three classes of leaf:
//
//  • notWired(name)     — addon doesn't export this yet. Throws so
//                         downstream pattern-match on err.code works.
//  • stubWarn(name)     — placeholder leaf (matches the WASM shim's
//                         round-1 stubs). Returns a marker once and
//                         warns once per session.
//  • stubFn(name)       — same, used inside expressions.
//
// Reuses the same code names the C++ Error layer uses, so consumers
// can switch on err.code consistently across both bindings.

function notWired(name) {
  return () => {
    const err = new Error(`[NOT_WIRED_IN_ADDON] ${name} is exposed by the WASM binding but not yet by the N-API addon. Add it to bindings/node/src/addon.cpp's Init() to enable.`)
    err.code = 'NOT_WIRED_IN_ADDON'
    throw err
  }
}

const _warned = new Set()
function stubWarn(name) {
  if (!_warned.has(name)) {
    console.warn(`[nxr.manifold.${name}] is a round-1 stub — returns a TODO marker. Wire when needed.`)
    _warned.add(name)
  }
  return { method: 'todo', name }
}

/* ---------------------------------------------------------------------- */
/*                      Six-group nested context                          */
/* ---------------------------------------------------------------------- */
//
// Mirrors the structure of `bindings/wasm/js/index.mjs::makeManifoldContext`.
// Difference: the addon takes `ctx` as the first arg of every method,
// vs. embind which dispatches through a `ContextWrapper` instance.
// We hide that asymmetry inside the shim — leaves look identical
// from the caller's perspective.

function makeManifoldContext(rawCtx) {
  // Lazy caches for the bulk operator assemblies, identical to the
  // WASM shim. First access of any DEC piece triggers a single
  // assembleDECOperators(); same for mass / cotanLaplacian.
  let _dec  = null
  let _mesh = null
  const dec  = () => (_dec  ??= addon.assembleDECOperators(rawCtx))
  const mesh = () => (_mesh ??= addon.assembleMeshOperators(rawCtx))

  // ── solve ─────────────────────────────────────────────────────
  const solve = {
    poisson(sourceVerts, sourceValues) {
      return addon.solvePoisson(rawCtx, asI32(sourceVerts), asF64(sourceValues))
    },
    /** Heat diffusion via spectral evolution. Requires `solve.eigen()`
     *  to have run at least once on this context — the addon stores
     *  the modes in its ContextHolder for the duration. */
    heat(sources, sourceValues, timesteps, alpha = 1.0) {
      return addon.generateHeatDiffusion(
        rawCtx,
        asI32(sources), asF64(sourceValues), asF64(timesteps), alpha,
      )
    },
    /** Async — the addon runs the eigensolve in a libuv worker thread
     *  via Napi::AsyncWorker so the Node event loop stays free. The
     *  resolved value is normalized to `{ eigenvectors, eigenvalues,
     *  k, nConverged }` matching the WASM binding's shape — addon
     *  natively returns `{ eigenvectors, eigenvalues, nV }` which we
     *  translate. */
    async eigen(k, sigma = -1e-8) {
      const r = await addon.solveEigenmodes(rawCtx, k, sigma)
      return {
        eigenvectors: r.eigenvectors,
        eigenvalues:  r.eigenvalues,
        k:            r.eigenvalues.length,
        nConverged:   r.eigenvalues.length,
      }
    },
    /** Async — the addon runs the Hodge solve in a libuv worker
     *  thread, so the JS event loop stays free during the d0ᵀ★₁d0
     *  Cholesky back-sub and the d1★₁⁻¹d1ᵀ LU solve. Resolves to
     *  the same `{ alpha, beta, gamma, omega, dAlpha, deltaBeta,
     *  omegaVectors, ... }` shape as the previous sync version.
     *  Provisional placement — Hodge–Helmholtz decomposition may be
     *  reclassified to `operator.hodgeDecomp` in a future round. */
    async hodge(omega) {
      return await addon.hodgeDecompose(rawCtx, asF64(omega))
    },
  }

  // ── operator ──────────────────────────────────────────────────
  // The split operators are CACHED — first access of any DEC piece
  // triggers a single assembleDECOperators() and reuses the result.
  // Same pattern for mass / stiffness / laplacian.
  //
  // Surface delta: addon's assembleDECOperators returns only
  // { d0, d1, hodge1 } (no hodge0, hodge2, hodge1Inverse) — those
  // four leaves are NOT_WIRED_IN_ADDON until extended in addon.cpp.
  const operator_ = {
    d0:           () => dec().d0,
    d1:           () => dec().d1,
    star0:        notWired('operator.star0'),
    star1:        () => dec().hodge1,
    star2:        notWired('operator.star2'),
    star1Inverse: notWired('operator.star1Inverse'),
    /** Mass matrix (sparse COO), variant-aware: diagonal for
     *  Voronoi/Barycentric, sparse non-diagonal for ConsistentFEM.
     *  Mirrors the WASM and MEX bindings' uniform exposure. */
    mass:         () => mesh().mass,
    stiffness:    () => mesh().cotanLaplacian,
    /** Cotangent Laplacian — same matrix as `operator.stiffness()`. */
    laplacian:    () => mesh().cotanLaplacian,
    /** Connection Laplacian on the chosen domain. See
     *  bindings/wasm/js/index.d.ts ConnectionLaplacianOptions /
     *  ConnectionLaplacianResult for the option / return shapes. */
    connectionLaplacian: (options = {}) =>
      addon.assembleConnectionLaplacian(rawCtx, options),
    /** Force a fresh assembly on next access (e.g. after geometry edit). */
    invalidateCache() { _dec = null; _mesh = null },
  }

  // ── query ─────────────────────────────────────────────────────
  const query = {
    vertex(v) { return { vertexIndex: v } },
    line(v1, v2)  { return stubWarn('query.line') },
    circle(v, r)  { return stubWarn('query.circle') },
    region(v, r)  { return stubWarn('query.region') },
    isoline(field, level) {
      return addon.computeIsolines(rawCtx, asF64(field), 1, level, level)
    },
    center(sourceVerts, p = 2) {
      return addon.vectorHeatFindCenter(rawCtx, asI32(sourceVerts), p)
    },
  }

  // ── measure ───────────────────────────────────────────────────
  // `measure.distance` is callable AND has a `.signed` sub-property:
  //   mctx.measure.distance([0])
  //   mctx.measure.distance.signed([0,1,2], true)
  function distance(sourceVerts) {
    return addon.computeGeodesicDistance(rawCtx, asI32(sourceVerts))
  }
  distance.signed = function distanceSigned(curveVerts, isLoop = true, levelSet = 1) {
    return addon.signedHeatDistance(rawCtx, asI32(curveVerts), isLoop, levelSet)
  }
  const measure = {
    distance,
    area(region)            { return stubWarn('measure.area') },
    density(region, field)  { return stubWarn('measure.density') },
    curvature()             { return addon.computeCurvatures(rawCtx) },
    normal(type = 0)        { return addon.computeVertexNormals(rawCtx, type) },
    /** computeFaceFrames is exposed in the WASM binding but not in
     *  the addon yet — see surface-delta note above. */
    frame:                  notWired('measure.frame'),
  }

  // ── uv ────────────────────────────────────────────────────────
  const uv = {
    /** computeUVCoordinates (BFF) is exposed in the WASM binding but
     *  not in the addon yet — see surface-delta note above. */
    bff:                    notWired('uv.bff'),
    logMap(sourceVertex, strategy = 1) {
      return addon.vectorHeatLogMap(rawCtx, sourceVertex, strategy)
    },
    stripe(vertexFieldRaw, uniformFrequency, connectOnSingularities = true) {
      return addon.computeStripePattern(rawCtx, asF64(vertexFieldRaw), uniformFrequency, connectOnSingularities)
    },
    stripeFreq(vertexFieldRaw, frequencies, connectOnSingularities = true) {
      return addon.computeStripePatternFreq(rawCtx, asF64(vertexFieldRaw), asF64(frequencies), connectOnSingularities)
    },
  }

  // ── interpolate ───────────────────────────────────────────────
  const interpolate = {
    transport(sourceVerts, sourceVectors) {
      return addon.vectorHeatTransport(rawCtx, asI32(sourceVerts), asF64(sourceVectors))
    },
    extend(sourceVerts, sourceValues) {
      return addon.vectorHeatExtendScalar(rawCtx, asI32(sourceVerts), asF64(sourceValues))
    },
    /** Async — addon dispatches the d0ᵀ★₁d0 solve to a worker thread.
     *  Returns Promise<{ connections, directionVectors, orthogonalVectors,
     *  eulerCharacteristic, gaussBonnetSatisfied }>. */
    async directionField(singVerts, singValues) {
      return await addon.computeDirectionField(rawCtx, asI32(singVerts), asF64(singValues))
    },
    smoothFaceField(nSym = 4, alignToCurvature = false) {
      return addon.computeSmoothFaceField(rawCtx, nSym, alignToCurvature)
    },
    smoothVertexField(nSym = 2, alignToCurvature = false) {
      return addon.computeSmoothVertexField(rawCtx, nSym, alignToCurvature)
    },
  }

  return {
    // Mesh sizes are not yet exposed as accessors on the addon's
    // ContextHolder — derive from the assembled MeshOperators (which
    // is cached on first access anyway).
    nV: () => mesh().nV,
    nE: () => mesh().nE,
    nF: () => mesh().nF,

    // Six-group nested namespace
    solve,
    operator: operator_,
    query,
    measure,
    uv,
    interpolate,

    /** Underlying External handle. Exposed for parity with the WASM
     *  shim's `_flat`; pass directly to any flat addon method. */
    _raw: rawCtx,
  }
}

/* ---------------------------------------------------------------------- */
/*    Functional form: `nxr.manifold.{group}.{method}(mctx, ...args)`     */
/* ---------------------------------------------------------------------- */
//
// Identical to the WASM shim's functional namespace. Each leaf
// forwards into the matching per-context method on the supplied
// ManifoldContext — both APIs share the same caches and the same
// underlying addon.

function makeFunctionalNamespace() {
  const fwd = (group, method) => (mctx, ...args) => {
    const fn = mctx?.[group]?.[method]
    if (typeof fn !== 'function') {
      const err = new Error(`nxr.manifold.${group}.${method}: not callable on this context`)
      err.code = 'INVALID_INPUT'
      throw err
    }
    return fn.apply(mctx[group], args)
  }
  return {
    solve: {
      poisson: fwd('solve', 'poisson'),
      heat:    fwd('solve', 'heat'),
      eigen:   fwd('solve', 'eigen'),
      hodge:   fwd('solve', 'hodge'),
    },
    operator: {
      d0:                  fwd('operator', 'd0'),
      d1:                  fwd('operator', 'd1'),
      star0:               fwd('operator', 'star0'),
      star1:               fwd('operator', 'star1'),
      star2:               fwd('operator', 'star2'),
      star1Inverse:        fwd('operator', 'star1Inverse'),
      mass:                fwd('operator', 'mass'),
      stiffness:           fwd('operator', 'stiffness'),
      laplacian:           fwd('operator', 'laplacian'),
      connectionLaplacian: fwd('operator', 'connectionLaplacian'),
    },
    query: {
      vertex:  fwd('query', 'vertex'),
      line:    fwd('query', 'line'),
      circle:  fwd('query', 'circle'),
      region:  fwd('query', 'region'),
      isoline: fwd('query', 'isoline'),
      center:  fwd('query', 'center'),
    },
    measure: {
      distance: Object.assign(
        (mctx, ...args) => mctx.measure.distance(...args),
        { signed: (mctx, ...args) => mctx.measure.distance.signed(...args) },
      ),
      area:      fwd('measure', 'area'),
      density:   fwd('measure', 'density'),
      curvature: fwd('measure', 'curvature'),
      normal:    fwd('measure', 'normal'),
      frame:     fwd('measure', 'frame'),
    },
    uv: {
      bff:        fwd('uv', 'bff'),
      logMap:     fwd('uv', 'logMap'),
      stripe:     fwd('uv', 'stripe'),
      stripeFreq: fwd('uv', 'stripeFreq'),
    },
    interpolate: {
      transport:         fwd('interpolate', 'transport'),
      extend:            fwd('interpolate', 'extend'),
      directionField:    fwd('interpolate', 'directionField'),
      smoothFaceField:   fwd('interpolate', 'smoothFaceField'),
      smoothVertexField: fwd('interpolate', 'smoothVertexField'),
    },
  }
}

/* ---------------------------------------------------------------------- */
/*                            Public surface                              */
/* ---------------------------------------------------------------------- */

/** Synchronous initializer — returns immediately since the addon is a
 *  CommonJS native module loaded via `require()`. The async signature
 *  matches `initNxrCompute` in the WASM binding so client code can
 *  swap targets without changing call sites:
 *
 *    import { initNxrCompute } from '@neurodynamics-xr/nxr-compute/node'
 *    const nxr = await initNxrCompute()
 *
 *  Both `await nxr` and direct synchronous use are supported.
 */
export async function initNxrCompute() {
  return api
}

const api = {
  /** Flat surface: identical to the addon's raw exports.
   *  `nxr.solveEigenmodes(ctx, k, …)` etc. */
  ...addon,

  /** Six-group nested namespace over the same compute context. */
  createManifoldContext(vertices, faces) {
    const rawCtx = addon.createContext(asF64(vertices), asI32(faces))
    return makeManifoldContext(rawCtx)
  },

  /** Functional form of the six-group namespace. Each leaf takes a
   *  ManifoldContext as its first argument. */
  nxr: {
    manifold: makeFunctionalNamespace(),
  },
}

export default initNxrCompute
