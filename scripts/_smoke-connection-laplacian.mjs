// Smoke test for the connection-Laplacian wiring.
// Builds an icosahedron, asks the WASM binding for the connection
// Laplacian on each domain (vertex / face / edge), and feeds the
// real2N form back into solveEigenmodesFromTriplets together with a
// 2N × 2N identity mass matrix. Asserts:
//
//   • Real2N output dimensions = 2 × baseDim
//   • Symmetric: a basic round-trip dim check (not value-level — that's
//     what test_connection_laplacian.cpp is for)
//   • Eigensolve composes: produces `k` finite, sorted, non-negative
//     eigenvalues from caller-built triplets
//   • Cache hit: identical-options round-trip returns COO with the same
//     nnz (the cache contract surfaced via the binding)
//
// Usage (from repo root):
//   node scripts/_smoke-connection-laplacian.mjs

import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot  = path.resolve(__dirname, '..')
const wasmJsPath = path.join(repoRoot, 'build_wasm', 'nxr_compute.js')

const { default: createNxrComputeModule } = await import(pathToFileURL(wasmJsPath).href)
const nxr = await createNxrComputeModule()

// Icosahedron fixture (same as scripts/_smoke-wasm.mjs).
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

const ctx = new nxr.Manifold(verts, faces)
const nV = ctx.nV(), nF = ctx.nF(), nE = ctx.nE()
console.log(`[smoke-cl] context: nV=${nV}, nF=${nF}, nE=${nE}`)

function require(cond, msg) {
  if (!cond) {
    console.error('FAIL:', msg)
    process.exit(1)
  }
}

const expectedBase = { vertex: nV, face: nF, edge: nE }
for (const domain of ['vertex', 'face', 'edge']) {
  for (const nSym of [1, 2, 4]) {
    const cl = ctx.assembleConnectionLaplacian({ domain, nSym })
    require(cl.format === 'real2N',           `${domain} nSym=${nSym}: default format is real2N`)
    require(cl.baseDim === expectedBase[domain],
            `${domain} nSym=${nSym}: baseDim == ${expectedBase[domain]} (got ${cl.baseDim})`)
    require(cl.outputDim === 2 * expectedBase[domain],
            `${domain} nSym=${nSym}: outputDim == 2 * baseDim`)
    require(cl.K.rows === cl.outputDim && cl.K.cols === cl.outputDim,
            `${domain} nSym=${nSym}: K shape == outputDim × outputDim`)
    require(cl.K.data instanceof Float64Array,
            `${domain} nSym=${nSym}: K.data is Float64Array (real form)`)
    console.log(`  ✓ ${domain} nSym=${nSym}: real2N ${cl.outputDim}×${cl.outputDim}, nnz=${cl.K.nnz}`)
  }
}

// Complex form sanity check.
{
  const cl = ctx.assembleConnectionLaplacian({ domain: 'vertex', nSym: 4, format: 'complex' })
  require(cl.format === 'complex',     'complex format round-trip')
  require(cl.outputDim === nV,         'complex outputDim == V')
  require(cl.K.rows === nV && cl.K.cols === nV,  'K complex shape == V × V')
  require(cl.K.realData instanceof Float64Array, 'K.realData present')
  require(cl.K.imagData instanceof Float64Array, 'K.imagData present')
  require(cl.K.realData.length === cl.K.nnz,     'realData length matches nnz')
  console.log(`  ✓ vertex nSym=4 complex: ${nV}×${nV}, nnz=${cl.K.nnz}`)
}

// ── Cache hit check ─────────────────────────────────────────
//
// Two assemblies with identical options must produce COO views with
// identical structure. The native test asserts bit-identity at the
// matrix level; here we just sanity-check the surfaced nnz.
{
  const a = ctx.assembleConnectionLaplacian({ domain: 'vertex', nSym: 4 })
  const b = ctx.assembleConnectionLaplacian({ domain: 'vertex', nSym: 4 })
  require(a.K.nnz === b.K.nnz, `cache hit: same nnz on repeat call (got ${a.K.nnz} vs ${b.K.nnz})`)
  console.log(`  ✓ cache hit: same nnz on repeat call`)
}

// ── Composition with solveEigenmodesFromTriplets ────────────
//
// Build a 2V × 2V identity mass and feed the connection Laplacian
// (real2N) into the generic eigensolver. Recovers the smallest k
// eigenpair on the connection bundle — the smoothest n-direction
// field.
{
  const cl = ctx.assembleConnectionLaplacian({
    domain: 'vertex', nSym: 4, regularization: 1e-6,
  })
  const N = cl.outputDim
  const k = 6

  const massN = N
  const Mrows = new Int32Array(massN)
  const Mcols = new Int32Array(massN)
  const Mvals = new Float64Array(massN)
  for (let i = 0; i < N; i++) { Mrows[i] = i; Mcols[i] = i; Mvals[i] = 1.0 }

  // Raw embind function takes positional args. The friendly destructured
  // signature lives on the JS wrapper from `initNxrCompute()`.
  const eig = nxr.solveEigenmodesFromTriplets(
    cl.K.row, cl.K.col, cl.K.data,
    Mrows,    Mcols,    Mvals,
    N, k, -1e-8,
    0, 0, 0,
  )

  require(eig.k === k,                 'k modes returned')
  require(eig.nConverged === k,        'all modes converged')
  require(eig.eigenvalues.length === k, 'eigenvalues length == k')
  require(eig.eigenvectors.length === N * k, `eigenvectors flat length == ${N}×${k}`)
  for (let i = 0; i < k; i++) {
    require(eig.eigenvalues[i] >= -1e-9, `λ_${i} >= 0 (got ${eig.eigenvalues[i]})`)
    require(Number.isFinite(eig.eigenvalues[i]), `λ_${i} finite`)
  }
  for (let i = 1; i < k; i++) {
    require(eig.eigenvalues[i] >= eig.eigenvalues[i - 1] - 1e-9, `λ sorted ascending`)
  }
  console.log(`  ✓ solveEigenmodesFromTriplets: k=${k}, λ₀=${eig.eigenvalues[0].toExponential(3)}`)
}

ctx.delete()
console.log('[smoke-cl] all assertions passed ✓')
