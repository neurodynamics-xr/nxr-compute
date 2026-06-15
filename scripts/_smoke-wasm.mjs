// Smoke test for the nxr-compute WASM binding. Loads native/build_wasm/nxr_compute.js,
// builds the same icosahedron fixture used by the native C++ tests,
// runs the precompute pipeline, and asserts shapes / values match the
// expected results. Verifies the entire pipeline:
//   JS → WASM heap → Embind → nxr-compute C++ → Eigen → return → JS
//
// Usage (from repo root):
//   node scripts/_smoke-wasm.mjs

import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot  = path.resolve(__dirname, '..')

const wasmJsPath = path.join(repoRoot, 'build_wasm', 'nxr_compute.js')
const moduleUrl  = pathToFileURL(wasmJsPath).href

console.log(`[smoke] loading ${wasmJsPath}`)
const { default: createNxrComputeModule } = await import(moduleUrl)
const nxrCompute = await createNxrComputeModule()
console.log(`[smoke] module ready — version: ${nxrCompute.version()}`)

// ── Icosahedron fixture (matches test_eigen.cpp) ──────────────
const t = (1 + Math.sqrt(5)) / 2
const raw = [
  -1, t, 0,   1, t, 0,  -1, -t, 0,   1, -t, 0,
   0, -1, t,  0, 1, t,   0, -1, -t,  0, 1, -t,
   t, 0, -1,  t, 0, 1,  -t, 0, -1,  -t, 0, 1,
]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) {
  const len = Math.hypot(raw[i], raw[i + 1], raw[i + 2])
  verts[i]     = raw[i]     / len
  verts[i + 1] = raw[i + 1] / len
  verts[i + 2] = raw[i + 2] / len
}
const faces = new Int32Array([
  0,11, 5,  0, 5, 1,  0, 1, 7,  0, 7,10,  0,10,11,
  1, 5, 9,  5,11, 4, 11,10, 2, 10, 7, 6,  7, 1, 8,
  3, 9, 4,  3, 4, 2,  3, 2, 6,  3, 6, 8,  3, 8, 9,
  4, 9, 5,  2, 4,11,  6, 2,10,  8, 6, 7,  9, 8, 1,
])

// ── Test 1: Manifold + accessors ───────────────────────
const ctx = new nxrCompute.Manifold(verts, faces)
const nV = ctx.nV(), nF = ctx.nF(), nE = ctx.nE()
console.log(`[smoke] context: nV=${nV}, nF=${nF}, nE=${nE}`)

function require(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg)
    process.exit(1)
  }
}
require(nV === 12, 'icosahedron has 12 vertices')
require(nF === 20, 'icosahedron has 20 faces')
require(nE === 30, 'icosahedron has 30 edges')

// ── Test 2: precompute (the visualization-defaults pack) ──────
const t0 = performance.now()
const data = ctx.precompute(6, -1e-8)  // k=6 modes
const elapsedPrecompute = (performance.now() - t0).toFixed(2)
console.log(`[smoke] precompute({k: 6}): ${elapsedPrecompute} ms`)

// Operators
require(data.operators.nV === 12, 'operators.nV')
require(Math.abs(data.operators.totalArea - 9.5745) < 1e-3,
  `unit-sphere icosahedron area should be ≈ 9.5745, got ${data.operators.totalArea}`)
require(data.operators.vertexNormals instanceof Float64Array, 'vertexNormals is Float64Array')
require(data.operators.vertexNormals.length === 12 * 3, 'vertexNormals length nV*3')
require(data.operators.cotanLaplacian.rows === 12, 'cotanLaplacian rows = nV')
require(data.operators.cotanLaplacian.cols === 12, 'cotanLaplacian cols = nV')
require(data.operators.cotanLaplacian.row instanceof Int32Array, 'cotanLaplacian COO.row Int32Array')
console.log(`  operators ✓ (totalArea=${data.operators.totalArea.toFixed(4)}, ` +
            `cotanLaplacian nnz=${data.operators.cotanLaplacian.nnz})`)

// DEC
require(data.dec.d0.rows === 30, 'd0 rows = nE')
require(data.dec.d0.cols === 12, 'd0 cols = nV')
require(data.dec.d1.rows === 20, 'd1 rows = nF')
require(data.dec.d1.cols === 30, 'd1 cols = nE')
console.log(`  DEC ✓ (d0 ${data.dec.d0.rows}×${data.dec.d0.cols}, ` +
            `d1 ${data.dec.d1.rows}×${data.dec.d1.cols})`)

// Eigenmodes — after removeDC, k should be 5 (one DC mode dropped)
require(data.eigenmodes.k === 5, `expected k=5 (post-removeDC), got ${data.eigenmodes.k}`)
require(data.eigenmodes.eigenvalues instanceof Float64Array, 'eigenvalues Float64Array')
require(data.eigenmodes.eigenvalues.length === 5, 'eigenvalues length')
require(data.eigenmodes.eigenvectors.length === 12 * 5, 'eigenvectors length nV*K')
const evals = Array.from(data.eigenmodes.eigenvalues)
console.log(`  eigenmodes ✓ k=${data.eigenmodes.k}, λ=[${evals.map(v => v.toFixed(4)).join(', ')}]`)

// Icosahedron canonical spectrum: triplet at λ=2, doublet at λ≈4.34
require(Math.abs(evals[0] - 2.0) < 1e-6,
  `expected λ_0 ≈ 2.0 on icosahedron, got ${evals[0]}`)
require(Math.abs(evals[3] - 4.34164) < 1e-3,
  `expected λ_3 ≈ 4.34, got ${evals[3]}`)

// Face frames
require(data.frames.e1.length      === 20 * 3, 'face frames e1 [F*3]')
require(data.frames.e2.length      === 20 * 3, 'face frames e2 [F*3]')
require(data.frames.normals.length === 20 * 3, 'face frames normals [F*3]')

// Verify orthonormality: e1·e2 ≈ 0, |e1| ≈ 1, e1×e2 ≈ n
let maxOrth = 0, maxNorm = 0
for (let f = 0; f < 20; f++) {
  const e1x = data.frames.e1[f*3], e1y = data.frames.e1[f*3+1], e1z = data.frames.e1[f*3+2]
  const e2x = data.frames.e2[f*3], e2y = data.frames.e2[f*3+1], e2z = data.frames.e2[f*3+2]
  maxOrth = Math.max(maxOrth, Math.abs(e1x*e2x + e1y*e2y + e1z*e2z))
  maxNorm = Math.max(maxNorm, Math.abs(Math.hypot(e1x, e1y, e1z) - 1))
}
require(maxOrth < 1e-12, `face frames orthogonal (max|e1·e2|=${maxOrth})`)
require(maxNorm < 1e-12, `face frames unit-length (max|‖e1‖−1|=${maxNorm})`)
console.log(`  frames ✓ (orthonormal to ${maxOrth.toExponential(2)})`)

// ── Test 3: M-orthonormality of eigenvectors ──────────────────
//
// U' M U should be ≈ identity. Phase A standardised vMajor
// (row-major) layout for [V × K] eigenvectors: U[v*K + k] gives
// vertex v's value in mode k. Mode k extraction is therefore a
// strided read (stride K), matching the cortical-flow Zarr schema
// `manifold/eigenmodes/eigenvectors` shape [V, K].
const evecs = data.eigenmodes.eigenvectors  // [V*K] row-major (vMajor)
const K = data.eigenmodes.k
const massCOO = data.operators.mass

const massDiag = new Float64Array(nV)
for (let i = 0; i < massCOO.row.length; i++) {
  if (massCOO.row[i] === massCOO.col[i]) {
    massDiag[massCOO.row[i]] = massCOO.data[i]
  }
}

// Mode extraction with vMajor: stride K read.
function extractMode(k) {
  const out = new Float64Array(nV)
  for (let v = 0; v < nV; v++) out[v] = evecs[v * K + k]
  return out
}

let maxOffDiag = 0
let maxDiagErr = 0
for (let j = 0; j < K; j++) {
  const u_j = extractMode(j)
  for (let k = 0; k < K; k++) {
    const u_k = extractMode(k)
    let dot = 0
    for (let v = 0; v < nV; v++) {
      dot += u_j[v] * massDiag[v] * u_k[v]
    }
    if (j === k) maxDiagErr  = Math.max(maxDiagErr,  Math.abs(dot - 1))
    else         maxOffDiag  = Math.max(maxOffDiag,  Math.abs(dot))
  }
}
require(maxOffDiag < 1e-10, `M-orth: max|U'MU|_off = ${maxOffDiag}`)
require(maxDiagErr < 1e-10, `M-orth: max|U'MU - I|_diag = ${maxDiagErr}`)
console.log(`  M-orthonormality ✓ (off-diag max=${maxOffDiag.toExponential(2)}, ` +
            `diag err max=${maxDiagErr.toExponential(2)})`)

// ── Test 4: face-centered scalar gradient ─────────────────────
const u_t = new Float64Array(nV).fill(0)
u_t[0] = 1.0       // delta at vertex 0
const grad = ctx.gradient(u_t)
require(grad.length === nF * 3, 'gradient length nF*3')
let maxGradMag = 0
for (let f = 0; f < nF; f++) {
  const m = Math.hypot(grad[f*3], grad[f*3+1], grad[f*3+2])
  maxGradMag = Math.max(maxGradMag, m)
}
require(maxGradMag > 0, 'gradient nontrivial')
console.log(`  gradient ✓ (max |∇u| = ${maxGradMag.toFixed(4)})`)

// ── Test 5: geodesic distance from a vertex ───────────────────
const dists = ctx.heat(new Int32Array([0]))
require(dists.length === nV, 'geodesic distances length nV')
require(Math.abs(dists[0]) < 1e-9, `distance at source = 0, got ${dists[0]}`)
let maxDist = 0
for (const d of dists) if (d > maxDist) maxDist = d
console.log(`  heat ✓ (max d = ${maxDist.toFixed(4)})`)

// ── Test 6: geodesic path between antipodes ───────────────────
const path0 = ctx.tracePath(0, 3)
require(path0.nPoints >= 2, 'path has at least 2 points')
require(path0.positions.length === path0.nPoints * 3, 'path positions [N*3]')
let pathLen = 0
for (let i = 0; i + 1 < path0.nPoints; i++) {
  const dx = path0.positions[(i+1)*3]   - path0.positions[i*3]
  const dy = path0.positions[(i+1)*3+1] - path0.positions[i*3+1]
  const dz = path0.positions[(i+1)*3+2] - path0.positions[i*3+2]
  pathLen += Math.hypot(dx, dy, dz)
}
require(pathLen > 2.0 && pathLen < Math.PI,
  `discrete antipode path in (chord=2.0, sphere-arc=π), got ${pathLen}`)
console.log(`  tracePath ✓ (antipode path length ${pathLen.toFixed(4)})`)

// ── Test 7: heat diffusion (spectral generator) ───────────────
const ts = Float64Array.from({ length: 5 }, (_, i) => i * 0.1)
const heat = ctx.heatDiffusion(
  new Int32Array([0]),
  new Float64Array([1.0]),
  ts,
  /* alpha = */ 1.0,
)
require(heat.T  === 5, `heat T=5, got ${heat.T}`)
require(heat.nV === 12, `heat nV=12, got ${heat.nV}`)
require(heat.data instanceof Float32Array, 'heat data Float32Array')
require(heat.data.length === 5 * 12, 'heat data length T*V')
console.log(`  heatDiffusion ✓ (T=${heat.T}, nV=${heat.nV})`)

// ── Test 8: vector heat method ───────────────────────────────
{
  const transported = ctx.parallel(
    new Int32Array([0]),
    new Float64Array([1.0, 0.0, 0.0]),
  )
  require(transported.length === nV * 3, 'vhm transport length nV*3')
  let maxNorm = 0
  for (let v = 0; v < nV; v++) {
    maxNorm = Math.max(maxNorm,
      Math.hypot(transported[v*3], transported[v*3+1], transported[v*3+2]))
  }
  require(maxNorm > 1e-6, 'vhm transport produced non-trivial vectors')

  const extended = ctx.extendScalar(
    new Int32Array([0, 6]),
    new Float64Array([1.0, 0.0]),
  )
  require(extended.length === nV, 'vhm extendScalar length nV')

  const logmap = ctx.logMap(0, /* AffineLocal */ 1)
  require(logmap.logCoords.length === nV * 2, 'logmap V*2')
  require(logmap.sourceE1.length === 3 && logmap.sourceE2.length === 3, 'logmap source frame 3')
  require(Math.hypot(logmap.logCoords[0], logmap.logCoords[1]) < 1e-6, 'logmap zero at source')

  const center = ctx.findCenter(new Int32Array([0, 1, 5]), 2)
  require(center.length === 3 && center.every(Number.isFinite), 'findCenter xyz finite')
  console.log(`  vectorHeat ✓ (transport max=${maxNorm.toFixed(3)}, ` +
              `center=[${[...center].map(v => v.toFixed(3)).join(', ')}])`)
}

// ── Test 9: signed heat method ───────────────────────────────
{
  const sd = ctx.signedHeat(
    new Int32Array([11, 5, 1, 7, 10]),
    /* isLoop = */ true,
    /* ZeroSet = */ 1,
  )
  require(sd.length === nV, 'signed distance length nV')
  let mn = Infinity, mx = -Infinity
  for (const d of sd) { if (d < mn) mn = d; if (d > mx) mx = d }
  require(mn < 0 && mx > 0, `signed range straddles zero, got [${mn}, ${mx}]`)
  console.log(`  signedHeat ✓ (range [${mn.toFixed(3)}, ${mx.toFixed(3)}])`)
}

// ── Test 10: smooth NRoSy direction fields ──────────────────
{
  const faceField = ctx.smoothFace(4, false)
  require(faceField.length === nF * 3, 'smooth face field length F*3')

  const vfield = ctx.smoothVertex(2, false)
  require(vfield.vertexVectors.length  === nV * 3, 'smooth vertex field V*3')
  require(vfield.vertexFieldRaw.length === nV * 2, 'smooth vertex field raw V*2')
  require(vfield.nSym === 2, 'nSym preserved')
  console.log(`  smoothField ✓ (face nF=${nF}, vertex nV=${nV})`)

  // ── Test 11: stripe patterns ───────────────────────────────
  const stripes = ctx.compute(vfield.vertexFieldRaw, 8.0, true)
  require(stripes.segmentCount >= 0, 'stripes segmentCount non-negative')
  require(stripes.positions.length === stripes.segmentCount * 6,
          'stripes positions length 2*segs*3')
  console.log(`  stripes ✓ (segs=${stripes.segmentCount})`)
}

// ── Test 12: high-level wrapper supplies defaults ───────────
//
// The raw Embind class above requires every argument. The
// index.mjs wrapper supplies the optional-argument defaults
// declared in index.d.ts. Verify the wrapper is loadable AND
// that calling each new method without its optional args works.
{
  const { initNxrCompute } = await import('../bindings/wasm/js/index.mjs')
  const wrapped = await initNxrCompute()                  // idempotent — re-uses the loaded module
  const wctx = wrapped.createContext(verts, faces)

  // Defaults: strategy=AffineLocal, p=2, isLoop=true, levelSet=ZeroSet,
  //           nSym=4 (face) / 2 (vertex), connectOnSingularities=true.
  const lm = wctx.logMap(0)
  require(lm.logCoords.length === nV * 2, 'wrapped logMap default strategy works')
  const c  = wctx.findCenter([0, 1, 5])
  require(c.length === 3, 'wrapped findCenter default p works')
  const sd = wctx.signedHeat([11, 5, 1, 7, 10])
  require(sd.length === nV, 'wrapped signedHeat default flags work')
  const ff = wctx.smoothFace()
  require(ff.length === nF * 3, 'wrapped smoothFace default nSym works')
  const vf = wctx.smoothVertex()
  const sp = wctx.compute(vf.vertexFieldRaw, 8.0)
  require(sp.segmentCount >= 0, 'wrapped stripes default flag works')
  wctx.delete()
  console.log('  index.mjs defaults ✓')
}

// ── Test 13: operators() command (Dirac + facet operators) ──────
//
// Parity with the MEX 'operators' command. Exercises the raw Embind method
// directly, then the index.mjs wrapper forwarder (undefined arg → null).
{
  const isCOO = (s, r, c) =>
    s && s.rows === r && s.cols === c &&
    s.row instanceof Int32Array && s.data instanceof Float64Array && s.nnz > 0

  // Extrinsic Dirac family + first-order operators.
  const dirac0 = ctx.operators('dirac', 0.0)
  require(isCOO(dirac0, 4*nV, 4*nV), 'dirac(0) is [4V×4V] COO')
  const dirac1 = ctx.operators('dirac', 1.0)
  require(isCOO(dirac1, 4*nV, 4*nV), 'dirac(1) is [4V×4V] COO')
  const dFace = ctx.operators('diracFace', 0.5)
  require(isCOO(dFace, 4*nF, 4*nF), 'diracFace(0.5) is [4F×4F] COO')
  const dD = ctx.operators('diracD')
  require(isCOO(dD, 4*nF, 4*nV), 'diracD is [4F×4V] COO')
  const dFaceD = ctx.operators('diracFaceD')
  require(isCOO(dFaceD, 4*nV, 4*nF), 'diracFaceD is [4V×4F] COO')
  const dIntD = ctx.operators('diracIntrinsicD')
  require(isCOO(dIntD, 4*nF, 4*nV), 'diracIntrinsicD is [4F×4V] COO')

  // Cross-binding anchor (cheap): dirac(0) == cotanL ⊗ I4, so nnz(dirac0) == 4·nnz(cotanL).
  const cotan = ctx.operators('laplacian', 'cotan')
  require(isCOO(cotan, nV, nV), 'laplacian cotan is [V×V] COO')
  require(dirac0.nnz === 4 * cotan.nnz, `dirac(0) nnz == 4·cotan nnz (${dirac0.nnz} vs ${4*cotan.nnz})`)

  // Other facet operators.
  const grad3D = ctx.operators('gradient3D')
  require(isCOO(grad3D, 3*nE, 3*nV), 'gradient3D is [3E×3N] COO')
  const dec = ctx.operators('dec')
  require(isCOO(dec.d0, nE, nV) && isCOO(dec.d1, nF, nE), 'dec returns {d0,d1}')
  const conn = ctx.operators('laplacian', 'connection')
  require(conn.rows === nV && conn.realData instanceof Float64Array &&
          conn.imagData instanceof Float64Array, 'connection is complex COO')

  // Error paths surface as JS Errors with the "[CODE] ..." message shape.
  let threwTau = false
  try { ctx.operators('dirac') } catch { threwTau = true }
  require(threwTau, 'dirac without tau throws')
  let threwFam = false
  try { ctx.operators('nope') } catch { threwFam = true }
  require(threwFam, 'unknown family throws')
  console.log(`  operators ✓ (dirac0 nnz=${dirac0.nnz}, diracD ${dD.rows}×${dD.cols}, ` +
              `diracFaceD ${dFaceD.rows}×${dFaceD.cols})`)

  // Wrapper forwarder: omitted optional arg works (undefined → null).
  const { initNxrCompute } = await import('../bindings/wasm/js/index.mjs')
  const wrapped = await initNxrCompute()
  const wctx = wrapped.createContext(verts, faces)
  const wD = wctx.operators('diracD')
  require(isCOO(wD, 4*nF, 4*nV), 'wrapped operators(diracD) works without arg')
  const wDirac = wctx.operators('dirac', 0.5)
  require(isCOO(wDirac, 4*nV, 4*nV), 'wrapped operators(dirac, tau) works')
  wctx.delete()
  console.log('  operators wrapper ✓')
}

// ── Cleanup ───────────────────────────────────────────────────
ctx.delete()
console.log('[smoke] all assertions passed ✓')
