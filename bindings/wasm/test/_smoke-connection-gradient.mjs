// Smoke test: manifold.operators('connectionGradient', nSym) returns complex COO.
// Mirrors the MEX test_connection_gradient.m.
//
// Usage (from repo root):
//   node bindings/wasm/test/_smoke-connection-gradient.mjs

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
const nV = m.nV(), nE = m.nE()
console.log(`[smoke] nV=${nV}, nE=${nE}`)

// ── connectionGradient nSym=1 ─────────────────────────────────
const G = m.operators('connectionGradient', 1)
require(G.rows > 0 && G.cols > 0,              'connectionGradient has non-zero dimensions')
require(G.rows === nE,                         `connectionGradient rows=${G.rows} must equal nE=${nE}`)
require(G.cols === nV,                         `connectionGradient cols=${G.cols} must equal nV=${nV}`)
require(G.nnz > 0,                             'connectionGradient has nonzeros')
require(G.realData instanceof Float64Array,    'realData is Float64Array')
require(G.imagData instanceof Float64Array,    'imagData is Float64Array')
require(G.realData.length === G.nnz,           `realData.length=${G.realData.length} == nnz=${G.nnz}`)
require(G.imagData.length === G.nnz,           `imagData.length=${G.imagData.length} == nnz=${G.nnz}`)
require(G.row instanceof Int32Array,           'row is Int32Array')
require(G.col instanceof Int32Array,           'col is Int32Array')
console.log(`  connectionGradient(1) ✓ [${G.rows}×${G.cols}] nnz=${G.nnz}`)

// ── connectionGradient nSym=2 has same shape ──────────────────
const G2 = m.operators('connectionGradient', 2)
require(G2.rows === G.rows && G2.cols === G.cols,
  `connectionGradient nSym=2 shape [${G2.rows}×${G2.cols}] == nSym=1 [${G.rows}×${G.cols}]`)
console.log(`  connectionGradient(2) ✓ shape matches nSym=1`)

m.delete()
console.log('connectionGradient WASM smoke PASSED')
