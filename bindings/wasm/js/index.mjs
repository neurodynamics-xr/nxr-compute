// cxf — WebAssembly compute backend for cortical surface analysis.
//
// Public ES-module API. Wraps the Emscripten-generated factory
// (createCxfModule) so consumers don't need to know about WASM
// initialisation. Re-exports a friendly `initCxf` that returns
// the loaded module.
//
// Usage:
//   import { initCxf } from '@cxf/wasm'
//   const cxf = await initCxf()
//   const ctx = cxf.createContext(verticesFloat64, facesInt32)
//   const data = ctx.precompute({ k: 300 })
//   // … use data.operators, data.dec, data.eigenmodes, data.faceFrames
//   ctx.delete()  // when done with this mesh

import createCxfModule from '../../build_wasm/cxf.js'

let _modulePromise = null

/**
 * Load the cxf WASM module. Idempotent — subsequent calls return the
 * same module instance.
 *
 * @param {object} [options]
 * @param {(filename: string) => string} [options.locateFile] —
 *   Custom resolver for cxf.wasm. By default the loader looks for
 *   cxf.wasm alongside cxf.js. Use this to override the location
 *   (e.g. when serving WASM from a CDN).
 * @returns {Promise<Cxf>}
 */
export async function initCxf(options = {}) {
  if (_modulePromise) return _modulePromise
  _modulePromise = createCxfModule(options).then(wrap)
  return _modulePromise
}

/** Builds the public surface from the raw Embind module. */
function wrap(mod) {
  return {
    /** @returns {string} the cxf version string, e.g. "cxf 0.1.0" */
    version: () => mod.version(),

    /**
     * Create a ComputeContext for the given mesh.
     *
     * @param {Float64Array | number[]} vertices — `V × 3` row-major
     *   xyz triples. Float64 is the canonical type; arrays are
     *   accepted for convenience.
     * @param {Int32Array | number[]} faces — `F × 3` row-major
     *   vertex indices, **0-based**.
     * @returns {ComputeContext}
     */
    createContext(vertices, faces) {
      const v = vertices instanceof Float64Array ? vertices : new Float64Array(vertices)
      const f = faces    instanceof Int32Array   ? faces    : new Int32Array(faces)
      if (v.length % 3 !== 0) throw new Error('cxf: vertices length must be a multiple of 3')
      if (f.length % 3 !== 0) throw new Error('cxf: faces length must be a multiple of 3')
      // The Embind constructor accepts JS arrays via val; pass typed arrays directly.
      const raw = new mod.ComputeContext(v, f)
      return makeContextWrapper(raw)
    },

    /** Underlying Embind module — escape hatch for advanced use. */
    _raw: mod,
  }
}

/** Adds JS-side ergonomics on top of the raw Embind ComputeContext. */
function makeContextWrapper(raw) {
  return {
    nV: () => raw.nV(),
    nE: () => raw.nE(),
    nF: () => raw.nF(),

    // Operators / geometry
    assembleMeshOperators: () => raw.assembleMeshOperators(),
    assembleDECOperators:  () => raw.assembleDECOperators(),
    computeFaceFrames:     () => raw.computeFaceFrames(),
    computeVertexNormals:  (type = 0) => raw.computeVertexNormals(type),

    // Spectral
    solveEigenmodes:       (k, sigma = -1e-8) => raw.solveEigenmodes(k, sigma),
    normalizeEigenmodes:   (U, rows, cols)    => raw.normalizeEigenmodes(U, rows, cols),
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
    solvePoisson(sourceVerts, sourceValues) {
      const sv = sourceVerts  instanceof Int32Array   ? sourceVerts  : new Int32Array(sourceVerts)
      const sa = sourceValues instanceof Float64Array ? sourceValues : new Float64Array(sourceValues)
      return raw.solvePoisson(sv, sa)
    },
    computeGeodesicDistance(sourceVerts) {
      const sv = sourceVerts instanceof Int32Array ? sourceVerts : new Int32Array(sourceVerts)
      return raw.computeGeodesicDistance(sv)
    },
    tracePath:        (vStart, vEnd) => raw.tracePath(vStart, vEnd),
    hodgeDecompose:   (omega)        => raw.hodgeDecompose(omega),

    // Geometric
    computeCurvatures:    () => raw.computeCurvatures(),
    computeUVCoordinates: () => raw.computeUVCoordinates(),
    computeIsolines(scalars, numLevels, minVal = 0, maxVal = 0) {
      return raw.computeIsolines(scalars, numLevels, minVal, maxVal)
    },
    computeDirectionField(singVerts, singValues) {
      const sv = singVerts  instanceof Int32Array   ? singVerts  : new Int32Array(singVerts)
      const sa = singValues instanceof Float64Array ? singValues : new Float64Array(singValues)
      return raw.computeDirectionField(sv, sa)
    },
    traceStreamlines(faceField, numSeeds = 15, stepCoef = 0.15, maxSteps = 1000) {
      return raw.traceStreamlines(faceField, numSeeds, stepCoef, maxSteps)
    },

    // Vector field
    whitneyInterpolate: (oneForm) => raw.whitneyInterpolate(oneForm),
    scalarGradient:     (scalar)  => raw.scalarGradient(scalar),

    // Time-varying generators
    generateHeatDiffusion(sources, sourceValues, timesteps, alpha = 1.0) {
      const sv = sources       instanceof Int32Array   ? sources       : new Int32Array(sources)
      const sa = sourceValues  instanceof Float64Array ? sourceValues  : new Float64Array(sourceValues)
      const ts = timesteps     instanceof Float64Array ? timesteps     : new Float64Array(timesteps)
      return raw.generateHeatDiffusion(sv, sa, ts, alpha)
    },
    generateDampedWave(modeIndices, amplitudes, dampings, phases, timesteps) {
      const mi = modeIndices instanceof Int32Array   ? modeIndices : new Int32Array(modeIndices)
      const am = amplitudes  instanceof Float64Array ? amplitudes  : new Float64Array(amplitudes)
      const da = dampings    instanceof Float64Array ? dampings    : new Float64Array(dampings)
      const ph = phases      instanceof Float64Array ? phases      : new Float64Array(phases)
      const ts = timesteps   instanceof Float64Array ? timesteps   : new Float64Array(timesteps)
      return raw.generateDampedWave(mi, am, da, ph, ts)
    },
    generateRandomDecomposed1Form(alphaStrength, betaStrength, gammaStrength, seed = 42) {
      return raw.generateRandomDecomposed1Form(alphaStrength, betaStrength, gammaStrength, seed)
    },

    /** Release WASM heap memory for this context. After delete(),
     *  the wrapper is unusable; create a fresh context for new work. */
    delete: () => raw.delete(),

    /** Underlying Embind handle — escape hatch. */
    _raw: raw,
  }
}

export default initCxf
