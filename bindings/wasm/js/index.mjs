// nxr-compute — WebAssembly compute backend for cortical surface analysis.
//
// Public ES-module API. Wraps the Emscripten-generated factory
// (createNxrComputeModule) so consumers don't need to know about WASM
// initialisation. Re-exports a friendly `initNxrCompute` that returns
// the loaded module.
//
// Two surfaces share the same underlying ContextWrapper:
//
//   1. Flat surface (legacy, backward-compatible):
//        const nxr = await initNxrCompute()
//        const ctx = nxr.createContext(verts, faces)
//        ctx.solve(300)
//
//   2. Six-group nested namespace `nxr.manifold.*` (preferred for new code):
//        const mctx = nxr.createManifoldContext(verts, faces)
//        mctx.solve.eigen(300)
//        mctx.operator.d0()
//        mctx.measure.distance([0])
//        mctx.measure.distance.signed([0, 1, 2], true)
//
//      Or in functional form:
//        nxr.nxr.manifold.solve.eigen(mctx, 300)
//        nxr.nxr.manifold.operator.d0(mctx)
//
// Repo / package: `@neurodynamics-xr/nxr-compute` (the engine).
// JS API namespace: `nxr.manifold.*` (the manifold concepts inside the engine).
// WASM artifact: `nxr_compute.wasm` (unchanged).
// C++ embind class: `ContextWrapper` (unchanged, flat).

// dist/wasm/ holds the committed prebuilt artifacts (refreshed by
// scripts/build-wasm.sh from build_wasm/ after a successful build).
// index.mjs lives three levels deep at bindings/wasm/js/, so the
// artifact is `../../../dist/wasm/`.
import createNxrComputeModule from '../../../dist/wasm/nxr_compute.js'

let _modulePromise = null

/**
 * Load the nxr-compute WASM module. Idempotent — subsequent calls return the
 * same module instance.
 *
 * @param {object} [options]
 * @param {(filename: string) => string} [options.locateFile] —
 *   Custom resolver for nxr_compute.wasm. By default the loader looks for
 *   nxr_compute.wasm alongside nxr_compute.js. Use this to override the location
 *   (e.g. when serving WASM from a CDN).
 * @returns {Promise<NxrCompute>}
 */
export async function initNxrCompute(options = {}) {
  if (_modulePromise) return _modulePromise
  _modulePromise = createNxrComputeModule(options).then(wrap)
  return _modulePromise
}

/** Builds the public surface from the raw Embind module. */
function wrap(mod) {
  const api = {
    /** @returns {string} the nxr-compute version string, e.g. "nxr-compute 0.1.0" */
    version: () => mod.version(),

    /**
     * Generalized eigensolve K φ = λ M φ from caller-supplied COO triplets.
     * Bypasses the halfedge Manifold path — use this when K and M
     * are assembled outside of geometry-central (graph Laplacians, FEM
     * assemblies on non-manifold meshes, custom regularized matrices).
     *
     * @param {object} args
     * @param {Int32Array | number[]} args.K_rows  COO row indices of K
     * @param {Int32Array | number[]} args.K_cols  COO col indices of K
     * @param {Float64Array | number[]} args.K_vals COO values of K
     * @param {Int32Array | number[]} args.M_rows  COO row indices of M
     * @param {Int32Array | number[]} args.M_cols  COO col indices of M
     * @param {Float64Array | number[]} args.M_vals COO values of M
     * @param {number} args.n        common matrix dimension (square)
     * @param {number} args.k        number of eigenmodes to compute
     * @param {number} [args.sigma]  shift-invert center (default -1e-8)
     * @param {number} [args.cancelAddr]   wasm heap pointer to cancel int32 (0 = none)
     * @param {number} [args.progressAddr] wasm heap pointer to 3×int32 progress region
     * @param {number} [args.progressLen]  progress region length, default 0
     * @returns {{ eigenvectors: Float64Array, eigenvalues: Float64Array, k: number, nConverged: number }}
     *          eigenvectors are vMajor row-major (Φ[v, k] at offset v*K + k).
     */
    solveEigenmodesFromTriplets: ({
        K_rows, K_cols, K_vals,
        M_rows, M_cols, M_vals,
        n, k, sigma = -1e-8,
        cancelAddr = 0, progressAddr = 0, progressLen = 0,
    }) => mod.solveEigenmodesFromTriplets(
        K_rows, K_cols, K_vals,
        M_rows, M_cols, M_vals,
        n, k, sigma,
        cancelAddr, progressAddr, progressLen,
    ),

    /**
     * Create a Manifold for the given mesh (flat surface).
     *
     * @param {Float64Array | number[]} vertices — `V × 3` row-major
     *   xyz triples. Float64 is the canonical type; arrays are
     *   accepted for convenience.
     * @param {Int32Array | number[]} faces — `F × 3` row-major
     *   vertex indices, **0-based**.
     * @returns {Manifold}
     */
    createContext(vertices, faces) {
      const raw = makeRaw(mod, vertices, faces)
      return makeContextWrapper(raw)
    },

    /**
     * Create a ManifoldContext for the given mesh — the same compute
     * context exposed under the six-group `nxr.manifold.*` namespace.
     * Internally constructs a single ContextWrapper; both
     * `mctx.solve.eigen(...)` and `mctx._flat.solve(...)`
     * call into the same C++ object.
     *
     * @param {Float64Array | number[]} vertices
     * @param {Int32Array | number[]} faces
     * @returns {ManifoldContext}
     */
    createManifoldContext(vertices, faces) {
      const raw = makeRaw(mod, vertices, faces)
      return makeManifoldContext(raw)
    },

    /**
     * Free-function form of the six-group namespace. Each leaf takes a
     * ManifoldContext as its first argument:
     *   nxr.nxr.manifold.solve.eigen(mctx, 300)
     * Backed by the same per-context methods, so behaviour is identical
     * to `mctx.solve.eigen(300)`.
     */
    nxr: {
      manifold: makeFunctionalNamespace(),
    },

    /** Underlying Embind module — escape hatch for advanced use. */
    _raw: mod,
  }
  return api
}

/* ---------------------------------------------------------------------- */
/*                        Raw context construction                        */
/* ---------------------------------------------------------------------- */

function makeRaw(mod, vertices, faces) {
  const v = vertices instanceof Float64Array ? vertices : new Float64Array(vertices)
  const f = faces    instanceof Int32Array   ? faces    : new Int32Array(faces)
  if (v.length % 3 !== 0) throw new Error('nxr: vertices length must be a multiple of 3')
  if (f.length % 3 !== 0) throw new Error('nxr: faces length must be a multiple of 3')
  // The Embind constructor accepts JS arrays via val; pass typed arrays directly.
  return new mod.Manifold(v, f)
}

/* ---------------------------------------------------------------------- */
/*                Flat surface (legacy, backward-compatible)              */
/* ---------------------------------------------------------------------- */

/** Adds JS-side ergonomics on top of the raw Embind Manifold. */
function makeContextWrapper(raw) {
  return {
    nV: () => raw.nV(),
    nE: () => raw.nE(),
    nF: () => raw.nF(),

    // Operators / geometry
    // `variant` ∈ "voronoi" (default) | "barycentric" | "full".
    assembleManifoldOperators: (variant = "") => raw.assembleManifoldOperators(variant),
    assembleDECOperators:  () => raw.assembleDECOperators(),
    frames:     () => raw.frames(),
    normals:  (type = 0) => raw.normals(type),

    // Spectral
    solve:       (k, sigma = -1e-8) => raw.solve(k, sigma),
    normalize:   (U, rows, cols)    => raw.normalize(U, rows, cols),
    removeDC:              (eig)              => raw.removeDC(eig),

    /**
     * One-shot precompute: assemble + DEC + eigensolve + normalize +
     * removeDC + face frames. Returns the visualization-defaults pack.
     *
     * @param {object} options
     * @param {number} [options.k=300]   — eigenmodes to solve
     * @param {number} [options.sigma=-1e-8] — shift for shift-invert
     */
    precompute({ k = 300, sigma = -1e-8 } = {}) {
      return raw.precompute(k, sigma)
    },

    // Solvers
    poisson(sourceVerts, sourceValues) {
      const sv = sourceVerts  instanceof Int32Array   ? sourceVerts  : new Int32Array(sourceVerts)
      const sa = sourceValues instanceof Float64Array ? sourceValues : new Float64Array(sourceValues)
      return raw.poisson(sv, sa)
    },
    heat(sourceVerts) {
      const sv = sourceVerts instanceof Int32Array ? sourceVerts : new Int32Array(sourceVerts)
      return raw.heat(sv)
    },
    tracePath:        (vStart, vEnd) => raw.tracePath(vStart, vEnd),
    hodge:   (omega)        => raw.hodge(omega),

    // Geometric
    curvatures:    () => raw.curvatures(),
    bff: () => raw.bff(),
    isoline(scalars, numLevels, minVal = 0, maxVal = 0) {
      return raw.isoline(scalars, numLevels, minVal, maxVal)
    },
    trivial(singVerts, singValues) {
      const sv = singVerts  instanceof Int32Array   ? singVerts  : new Int32Array(singVerts)
      const sa = singValues instanceof Float64Array ? singValues : new Float64Array(singValues)
      return raw.trivial(sv, sa)
    },
    streamline(faceField, numSeeds = 15, stepCoef = 0.15, maxSteps = 1000) {
      return raw.streamline(faceField, numSeeds, stepCoef, maxSteps)
    },

    // Vector field
    whitney: (oneForm) => raw.whitney(oneForm),
    gradient:     (scalar)  => raw.gradient(scalar),

    // Time-varying generators
    heatDiffusion(sources, sourceValues, timesteps, alpha = 1.0) {
      const sv = sources       instanceof Int32Array   ? sources       : new Int32Array(sources)
      const sa = sourceValues  instanceof Float64Array ? sourceValues  : new Float64Array(sourceValues)
      const ts = timesteps     instanceof Float64Array ? timesteps     : new Float64Array(timesteps)
      return raw.heatDiffusion(sv, sa, ts, alpha)
    },
    dampedWave(modeIndices, amplitudes, dampings, phases, timesteps) {
      const mi = modeIndices instanceof Int32Array   ? modeIndices : new Int32Array(modeIndices)
      const am = amplitudes  instanceof Float64Array ? amplitudes  : new Float64Array(amplitudes)
      const da = dampings    instanceof Float64Array ? dampings    : new Float64Array(dampings)
      const ph = phases      instanceof Float64Array ? phases      : new Float64Array(phases)
      const ts = timesteps   instanceof Float64Array ? timesteps   : new Float64Array(timesteps)
      return raw.dampedWave(mi, am, da, ph, ts)
    },
    randomDecomposed1Form(alphaStrength, betaStrength, gammaStrength, seed = 42) {
      return raw.randomDecomposed1Form(alphaStrength, betaStrength, gammaStrength, seed)
    },

    // Vector heat method (Sharp, Soliman, Crane 2019)
    parallel(sourceVerts, sourceVectors) {
      const sv = sourceVerts    instanceof Int32Array   ? sourceVerts    : new Int32Array(sourceVerts)
      const vv = sourceVectors  instanceof Float64Array ? sourceVectors  : new Float64Array(sourceVectors)
      return raw.parallel(sv, vv)
    },
    extendScalar(sourceVerts, sourceValues) {
      const sv = sourceVerts   instanceof Int32Array   ? sourceVerts   : new Int32Array(sourceVerts)
      const sa = sourceValues  instanceof Float64Array ? sourceValues  : new Float64Array(sourceValues)
      return raw.extendScalar(sv, sa)
    },
    logMap(sourceVertex, strategy = 1 /* AffineLocal */) {
      return raw.logMap(sourceVertex, strategy)
    },
    findCenter(sourceVerts, p = 2) {
      const sv = sourceVerts instanceof Int32Array ? sourceVerts : new Int32Array(sourceVerts)
      return raw.findCenter(sv, p)
    },

    // Signed heat method (Feng & Crane 2024)
    signedHeat(curveVerts, isLoop = true, levelSet = 1 /* ZeroSet */) {
      const cv = curveVerts instanceof Int32Array ? curveVerts : new Int32Array(curveVerts)
      return raw.signedHeat(cv, isLoop, levelSet)
    },

    // Smooth direction fields (Knöppel-Crane)
    smoothFace(nSym = 4, alignToCurvature = false) {
      return raw.smoothFace(nSym, alignToCurvature)
    },
    smoothVertex(nSym = 2, alignToCurvature = false) {
      return raw.smoothVertex(nSym, alignToCurvature)
    },

    // Stripe patterns (Knöppel-Crane SIGGRAPH 2015)
    compute(vertexFieldRaw, uniformFrequency, connectOnSingularities = true) {
      const vf = vertexFieldRaw instanceof Float64Array ? vertexFieldRaw : new Float64Array(vertexFieldRaw)
      return raw.compute(vf, uniformFrequency, connectOnSingularities)
    },
    computeFreq(vertexFieldRaw, frequencies, connectOnSingularities = true) {
      const vf = vertexFieldRaw instanceof Float64Array ? vertexFieldRaw : new Float64Array(vertexFieldRaw)
      const fr = frequencies    instanceof Float64Array ? frequencies    : new Float64Array(frequencies)
      return raw.computeFreq(vf, fr, connectOnSingularities)
    },

    /** Release WASM heap memory for this context. After delete(),
     *  the wrapper is unusable; create a fresh context for new work. */
    delete: () => raw.delete(),

    /** Underlying Embind handle — escape hatch. */
    _raw: raw,
  }
}

/* ---------------------------------------------------------------------- */
/*           Six-group nested namespace `nxr.manifold.*` shim             */
/* ---------------------------------------------------------------------- */

// Module-level dedup set so `console.warn` per stub fires once across
// the whole process, not once per ManifoldContext instance.
const _stubWarned = new Set()
function stubWarn(name) {
  if (!_stubWarned.has(name)) {
    _stubWarned.add(name)
    // eslint-disable-next-line no-console
    console.warn(
      `[nxr.manifold] ${name}: round-1 placeholder — full implementation lands in a follow-up round.`,
    )
  }
  return { method: 'todo', name }
}

function asI32(a) { return a instanceof Int32Array   ? a : new Int32Array(a) }
function asF64(a) { return a instanceof Float64Array ? a : new Float64Array(a) }

/**
 * Build a ManifoldContext exposing the six-group nested namespace
 * (solve / operator / query / measure / uv / interpolate) on top of the
 * underlying ContextWrapper. Each leaf delegates to a single C++ method;
 * no math is reimplemented here. The split DEC and mesh operators (e.g.
 * `operator.d0`, `operator.mass`) are extracted from the cached results
 * of the bulk `assembleDECOperators` / `assembleManifoldOperators` calls.
 */
function makeManifoldContext(raw) {
  // Lazy caches for the bulk operator assemblies. `operator.d0()` and
  // `operator.d1()` should not call assembleDECOperators twice.
  let _dec  = null
  let _mesh = null
  const dec  = () => (_dec  ??= raw.assembleDECOperators())
  const mesh = () => (_mesh ??= raw.assembleManifoldOperators(""))

  // ----- solve --------------------------------------------------------
  const solve = {
    poisson(sourceVerts, sourceValues) {
      return raw.poisson(asI32(sourceVerts), asF64(sourceValues))
    },
    /**
     * Heat diffusion on the manifold over the given timesteps.
     * Consolidates the legacy `heatDiffusion` entry point —
     * one call returns a TimeSeriesField.
     */
    heat(sources, sourceValues, timesteps, alpha = 1.0) {
      return raw.heatDiffusion(
        asI32(sources), asF64(sourceValues), asF64(timesteps), alpha,
      )
    },
    eigen(k, sigma = -1e-8) {
      return raw.solve(k, sigma)
    },
    /** Provisional — flag for future re-classification (likely
     *  `operator.hodgeDecomp` or `interpolate.hodge`). */
    hodge(omega) {
      return raw.hodge(omega)
    },
  }

  // ----- operator -----------------------------------------------------
  // The split operators are CACHED — the first access of any DEC piece
  // triggers a single assembleDECOperators() and reuses the result for
  // every subsequent .d0() / .d1() / .star0() / .star1() / .star2() /
  // .star1Inverse() call. Same pattern for mass / stiffness / laplacian.
  const operator = {
    d0:           () => dec().d0,
    d1:           () => dec().d1,
    star0:        () => dec().hodge0,
    star1:        () => dec().hodge1,
    star2:        () => dec().hodge2,
    star1Inverse: () => dec().hodge1Inverse,
    mass:         () => mesh().mass,
    stiffness:    () => mesh().stiffness,
    /** Cotangent Laplacian — same matrix as `operator.stiffness()`. */
    laplacian:    () => mesh().stiffness,
    /**
     * Connection Laplacian on the chosen domain. Drives smoothest
     * n-RoSy direction fields, parallel-transport energies, and
     * tangent-bundle eigendecompositions.
     *
     * Default `format: 'real2N'` returns a 2N×2N symmetric real
     * sparse COO matrix that drops directly into
     * `solveEigenmodesFromTriplets` together with a block-diagonal
     * real mass matrix `blkdiag(M, M)`. The smallest eigenpair
     * reproduces the smoothest n-direction field that
     * `interpolate.smoothFace` / `smoothVertex` return
     * internally.
     *
     * The `format: 'complex'` form returns a complex Hermitian COO
     * with parallel `realData` / `imagData` value arrays — useful
     * when callers want to apply a complex-Hermitian eigensolver
     * themselves.
     *
     * Result-level cache: identical options round-trip in O(1).
     *
     * @param {Object}   [options]
     * @param {'vertex'|'face'|'edge'} [options.domain='vertex']
     * @param {number}   [options.nSym=1]               1=vector, 2=line, 4=cross
     * @param {number}   [options.regularization=1e-8]  additive ε·I shift
     * @param {'real2N'|'complex'} [options.format='real2N']
     * @returns {{
     *   K: SparseMatrixCOO | SparseMatrixCOOComplex,
     *   baseDim: number, outputDim: number,
     *   domain: 'vertex'|'face'|'edge',
     *   nSym: number, regularization: number,
     *   format: 'real2N'|'complex'
     * }}
     */
    connectionLaplacian: (options = {}) =>
      raw.assembleConnectionLaplacian(options),
    /** Force a fresh assembly on next access (e.g. after geometry edit). */
    invalidateCache() { _dec = null; _mesh = null },
  }

  // ----- query --------------------------------------------------------
  // Most query primitives are round-1 stubs. Where a clean composition
  // from existing primitives is available (e.g. `query.isoline` from
  // isoline), we wire it; otherwise emit a one-shot warning and
  // return a TODO marker.
  const query = {
    /** Identity wrapper — useful for the functional form
     *  `nxr.manifold.query.vertex(mctx, v)` where you want a structured
     *  representation of a single-vertex selection. */
    vertex(v) { return { vertexIndex: v } },
    line(v1, v2)        { return stubWarn('query.line'); /* TODO: trace flip-out geodesic between v1 and v2 */ },
    circle(v, r)        { return stubWarn('query.circle'); /* TODO: isoline of geodesic distance from v at radius r */ },
    region(v, r)        { return stubWarn('query.region'); /* TODO: vertex set within geodesic radius r of v */ },
    /** Single-level isoline of a per-vertex scalar field. */
    isoline(field, level) {
      const f = asF64(field)
      return raw.isoline(f, 1, level, level)
    },
    /** Karcher mean of source vertices (vector-heat findCenter, p-th power). */
    center(sourceVerts, p = 2) {
      return raw.findCenter(asI32(sourceVerts), p)
    },
  }

  // ----- measure ------------------------------------------------------
  // `measure.distance` is callable AND has a `.signed` sub-property:
  //   mctx.measure.distance([0])                  // unsigned heat-method geodesic
  //   mctx.measure.distance.signed([0,1,2], true) // signed heat distance from a curve
  function distance(sourceVerts) {
    return raw.heat(asI32(sourceVerts))
  }
  distance.signed = function distanceSigned(curveVerts, isLoop = true, levelSet = 1 /* ZeroSet */) {
    return raw.signedHeat(asI32(curveVerts), isLoop, levelSet)
  }
  const measure = {
    distance,
    area(region)            { return stubWarn('measure.area'); /* TODO: integrate vertex/face areas over a region selection */ },
    density(region, field)  { return stubWarn('measure.density'); /* TODO: average a scalar field over a region */ },
    curvature()             { return raw.curvatures() },
    normal(type = 0)        { return raw.normals(type) },
    frame()                 { return raw.frames() },
  }

  // ----- uv -----------------------------------------------------------
  const uv = {
    /** Boundary First Flattening (open meshes only — throws otherwise). */
    bff() { return raw.bff() },
    /** Logarithmic map at one source vertex. */
    logMap(sourceVertex, strategy = 1 /* AffineLocal */) {
      return raw.logMap(sourceVertex, strategy)
    },
    /** Sinusoidal stripes aligned to a 2-RoSy field. */
    stripe(vertexFieldRaw, uniformFrequency, connectOnSingularities = true) {
      return raw.compute(asF64(vertexFieldRaw), uniformFrequency, connectOnSingularities)
    },
    /** Stripe pattern with a per-vertex frequency (length V). */
    stripeFreq(vertexFieldRaw, frequencies, connectOnSingularities = true) {
      return raw.computeFreq(asF64(vertexFieldRaw), asF64(frequencies), connectOnSingularities)
    },
  }

  // ----- interpolate --------------------------------------------------
  const interpolate = {
    /** Parallel-transport tangent vectors via the vector heat method. */
    transport(sourceVerts, sourceVectors) {
      return raw.parallel(asI32(sourceVerts), asF64(sourceVectors))
    },
    /** Smoothly extend a sparse scalar from the source vertices to all V. */
    extend(sourceVerts, sourceValues) {
      return raw.extendScalar(asI32(sourceVerts), asF64(sourceValues))
    },
    /** Trivial-connection direction field driven by prescribed singularities. */
    trivial(singVerts, singValues) {
      return raw.trivial(asI32(singVerts), asF64(singValues))
    },
    /** Smoothest face-based direction field (Knöppel-Crane). */
    smoothFace(nSym = 4, alignToCurvature = false) {
      return raw.smoothFace(nSym, alignToCurvature)
    },
    /** Smoothest vertex-based direction field — the `vertexFieldRaw` field
     *  in the result is what `uv.stripe` consumes. */
    smoothVertex(nSym = 2, alignToCurvature = false) {
      return raw.smoothVertex(nSym, alignToCurvature)
    },
  }

  return {
    // Mesh sizes
    nV: () => raw.nV(),
    nE: () => raw.nE(),
    nF: () => raw.nF(),

    // Six-group nested namespace
    solve,
    operator,
    query,
    measure,
    uv,
    interpolate,

    /** Release WASM heap memory; the context is invalid after dispose(). */
    dispose: () => raw.delete(),

    /**
     * Flat compatibility surface — same methods as `createContext()`'s
     * return value, sharing the same underlying ContextWrapper. Useful
     * for code that wants to mix flat and namespaced calls during the
     * migration.
     */
    _flat: makeContextWrapper(raw),

    /** Underlying Embind handle — escape hatch. */
    _raw: raw,
  }
}

/* ---------------------------------------------------------------------- */
/*    Functional form: `nxr.manifold.{group}.{method}(mctx, ...args)`     */
/* ---------------------------------------------------------------------- */

/**
 * Build the functional namespace tree once; every leaf forwards into
 * the corresponding per-context method on the supplied ManifoldContext.
 * Both APIs share the same caches and underlying ContextWrapper, so
 *   nxr.nxr.manifold.operator.d0(mctx)
 * is byte-equivalent to
 *   mctx.operator.d0()
 */
function makeFunctionalNamespace() {
  const fwd = (group, method) => (mctx, ...args) => {
    const fn = mctx?.[group]?.[method]
    if (typeof fn !== 'function') {
      throw new Error(`nxr.manifold.${group}.${method}: not callable on this context`)
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
      d0:                   fwd('operator', 'd0'),
      d1:                   fwd('operator', 'd1'),
      star0:                fwd('operator', 'star0'),
      star1:                fwd('operator', 'star1'),
      star2:                fwd('operator', 'star2'),
      star1Inverse:         fwd('operator', 'star1Inverse'),
      mass:                 fwd('operator', 'mass'),
      stiffness:            fwd('operator', 'stiffness'),
      laplacian:            fwd('operator', 'laplacian'),
      connectionLaplacian:  fwd('operator', 'connectionLaplacian'),
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
      // distance is special: callable AND has a .signed sub-leaf.
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
      trivial:    fwd('interpolate', 'trivial'),
      smoothFace:   fwd('interpolate', 'smoothFace'),
      smoothVertex: fwd('interpolate', 'smoothVertex'),
    },
  }
}

export default initNxrCompute
