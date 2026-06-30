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
