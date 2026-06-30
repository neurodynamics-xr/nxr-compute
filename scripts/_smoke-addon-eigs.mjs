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

// Dirac: blockSize 4, with multiplets the count is rounded up to complete 4-fold clusters
// (reconstructMultiplets closes each cluster → may return more than k eigenvalues)
const dir = await m.eigs({ operator: 'dirac', tau: 0.5, k: 8, multiplets: true })
check(dir.blockSize === 4, 'dirac blockSize is 4')
check(dir.eigenvalues.length % 4 === 0, 'dirac eigenvalue count is divisible by 4 (multiplets)')
check(dir.eigenvalues.length >= 8, 'dirac eigs returns at least k=8 eigenvalues')

// Validation error rejects with INVALID_INPUT (no k)
let rejected = false
try { await m.eigs({ operator: 'cotan' }) } catch (e) { rejected = true; check(e.code === 'INVALID_INPUT', 'missing k rejects INVALID_INPUT') }
check(rejected, 'missing k rejects the promise')

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
