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
  if (a.rows !== b.rows || a.cols !== b.cols) {
    check(false, `${label}: shape mismatch (addon ${a.rows}×${a.cols} vs wasm ${b.rows}×${b.cols})`)
    return
  }
  // Union-based value comparison: treat missing entries as 0. This is correct
  // because sparse storage allows both representations of the same matrix — one
  // may store a structural zero explicitly, the other prunes it. On a symmetric
  // icosahedron, native x86 FP (AVX2) can preserve near-zero entries (< 1e-16)
  // that WASM's strict-IEEE arithmetic cancels to exactly 0.0 and then filters
  // via `if (B(a,b) != 0.0)`. Both platforms produce numerically identical
  // matrices (max diff < 1e-15); the nnz difference is a storage artefact.
  const ma = cooMap(a, field), mb = cooMap(b, field)
  let maxErr = 0
  for (const [k, va] of ma) maxErr = Math.max(maxErr, Math.abs(va - (mb.get(k) ?? 0)))
  for (const [k, vb] of mb) maxErr = Math.max(maxErr, Math.abs((ma.get(k) ?? 0) - vb))
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
// dense: true — use Eigen's exact GeneralizedSelfAdjointEigenSolver rather than
// Spectra's IRAM. On the highly symmetric icosahedron, Spectra's iterative
// convergence is sensitive to the random starting vector and platform FP, and
// gives platform-dependent wrong answers at degenerate cluster boundaries
// (k = 8 sits exactly at the 4+4 Dirac cluster boundary, where IRAM can
// converge to a mix of vectors from the wrong cluster). The dense path is
// deterministic, platform-independent, and trivially fast for these 12-48-80
// vertex matrices. Tolerance 1e-9 is comfortably met (observed max err < 7e-15).
for (const spec of [
  { operator: 'cotan', k: 6, dense: true },
  { operator: 'graph', k: 6, dense: true },
  { operator: 'dirac', tau: 0.5, k: 8, dense: true },
  { operator: 'diracFace', tau: 0.5, k: 8, dense: true },
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
