// eigsPencil: pre-assembled (A,B) pencil generalized eigensolve.
// Run from repo root:
//   bash scripts/build.sh Release && node bindings/node/test/test_eigs_pencil.mjs
import { initNxrCompute } from '../index.mjs'

let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }
const approx = (a, b, tol = 1e-8) => Math.abs(a - b) <= tol * (1 + Math.abs(b))

const nxr = await initNxrCompute()

// Icosahedron fixture (same as the parity test).
const t = (1 + Math.sqrt(5)) / 2
const rawV = [-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t, 0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) { const l = Math.hypot(rawV[i],rawV[i+1],rawV[i+2]); verts[i]=rawV[i]/l; verts[i+1]=rawV[i+1]/l; verts[i+2]=rawV[i+2]/l }
const faces = new Int32Array([0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1])

const ctx = nxr.createContext(verts, faces)

// ── Parity: eigsPencil(cotanCOO, massCOO) == eigs({operator:'laplacian'}) ──
const K = nxr.operators(ctx, 'laplacian', 'cotan')
const M = nxr.operators(ctx, 'mass', 'galerkin')
const named = await nxr.eigs(ctx, { operator: 'laplacian', subtype: 'cotan', mass: 'galerkin', k: 6, normalize: true })
const pencil = await nxr.eigsPencil(K, M, { k: 6, normalize: true })

check(pencil.eigenvalues.length === 6, 'eigsPencil returns k=6 eigenvalues')
check(pencil.eigenvectors.length === 12 * 6, 'eigenvectors are [nV·k] = 12·6')
let maxEvalErr = 0
for (let i = 0; i < 6; i++) maxEvalErr = Math.max(maxEvalErr, Math.abs(pencil.eigenvalues[i] - named.eigenvalues[i]))
check(maxEvalErr < 1e-6, `eigenvalues match named eigs (maxErr ${maxEvalErr.toExponential(2)})`)
check(approx(pencil.eigenvalues[0], 0, 1e-6) || pencil.eigenvalues[0] < 1e-6, 'λ0 ≈ 0 (constant mode)')
check(pencil.eigenvalues[5] >= pencil.eigenvalues[0], 'eigenvalues ascending')

// ── Ladder: a near-singular diagonal-ish pencil solved for smallest ──
// A = diag(0, 1, 2, 3) (singular smallest mode); B = I. Smallest λ = 0,1,2.
const A4 = { row: Int32Array.of(0,1,2,3), col: Int32Array.of(0,1,2,3), data: Float64Array.of(0,1,2,3), rows: 4, cols: 4 }
const I4 = { row: Int32Array.of(0,1,2,3), col: Int32Array.of(0,1,2,3), data: Float64Array.of(1,1,1,1), rows: 4, cols: 4 }
const lad = await nxr.eigsPencil(A4, I4, { k: 3, normalize: true })
check(approx(lad.eigenvalues[0], 0, 1e-6), `ladder: λ0 ≈ 0 (got ${lad.eigenvalues[0].toExponential(2)})`)
check(approx(lad.eigenvalues[1], 1, 1e-6), `ladder: λ1 ≈ 1 (got ${lad.eigenvalues[1]})`)
check(approx(lad.eigenvalues[2], 2, 1e-6), `ladder: λ2 ≈ 2 (got ${lad.eigenvalues[2]})`)

// ── Validation: non-square / mismatched sizes reject ──
let rejected = false
try { await nxr.eigsPencil(A4, { ...I4, rows: 5, cols: 5 }, { k: 2 }) } catch (e) { rejected = true }
check(rejected, 'mismatched A/B sizes reject')

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
