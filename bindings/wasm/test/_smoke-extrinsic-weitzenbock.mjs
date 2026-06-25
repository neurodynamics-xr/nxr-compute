// Smoke test: manifold.operators('extrinsicWeitzenbock') returns real COO [3V×3V].
// Mirrors the MEX test_extrinsic_weitzenbock.m.
//
// Usage (from repo root):
//   node bindings/wasm/test/_smoke-extrinsic-weitzenbock.mjs

import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot  = path.resolve(__dirname, '../../..')

const wasmJsPath = path.join(repoRoot, 'build_wasm', 'nxr_compute.js')
const moduleUrl  = pathToFileURL(wasmJsPath).href

console.log(`[smoke] loading ${wasmJsPath}`)
const { default: createNxrComputeModule } = await import(moduleUrl)
const Module = await createNxrComputeModule()
console.log(`[smoke] module ready — version: ${Module.version()}`)

function require(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg)
    process.exit(1)
  }
}

// ── Icosahedron fixture (matches test_eigen.cpp and _smoke-wasm.mjs) ──
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

const m = new Module.Manifold(verts, faces)
const nV = m.nV()
console.log(`[smoke] nV=${nV}`)

// ── extrinsicWeitzenbock ───────────────────────────────────────
const W = m.operators('extrinsicWeitzenbock')
require(W.rows > 0 && W.cols > 0,              'extrinsicWeitzenbock has non-zero dimensions')
require(W.rows === W.cols,                     `extrinsicWeitzenbock must be square, got [${W.rows}×${W.cols}]`)
require(W.rows === 3 * nV,                     `extrinsicWeitzenbock rows=${W.rows} must equal 3*nV=${3 * nV}`)
require(W.nnz > 0,                             'extrinsicWeitzenbock has nonzeros')
require(W.data instanceof Float64Array,        'data is Float64Array (real)')
require(W.data.length === W.nnz,               `data.length=${W.data.length} == nnz=${W.nnz}`)
require(W.row instanceof Int32Array,           'row is Int32Array')
require(W.col instanceof Int32Array,           'col is Int32Array')
// No imagData field expected for a real operator
require(!('imagData' in W),                   'extrinsicWeitzenbock must not have imagData (real operator)')
console.log(`  extrinsicWeitzenbock ✓ [${W.rows}×${W.cols}] nnz=${W.nnz}`)

m.delete()
console.log('extrinsicWeitzenbock WASM smoke PASSED')
