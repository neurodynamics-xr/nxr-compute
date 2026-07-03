// eigsPencil iterative (matrix-free CG) shift-invert path.
// The CG path applies (A - σB)⁻¹ via ConjugateGradient — no direct factorization,
// so it handles dense pencils (e.g. the connectome) that SimplicialLLT can't.
// Correctness: iterative eigenvalues must match the direct path (and named eigs).
// Run: bash scripts/build.sh Release && node bindings/node/test/test_eigs_pencil_iterative.mjs
import { initNxrCompute } from '../index.mjs'

let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

const nxr = await initNxrCompute()

// Icosahedron fixture.
const t = (1 + Math.sqrt(5)) / 2
const rawV = [-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t, 0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) { const l = Math.hypot(rawV[i],rawV[i+1],rawV[i+2]); verts[i]=rawV[i]/l; verts[i+1]=rawV[i+1]/l; verts[i+2]=rawV[i+2]/l }
const faces = new Int32Array([0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1])

const ctx = nxr.createContext(verts, faces)
const K = nxr.operators(ctx, 'laplacian', 'cotan')
const M = nxr.operators(ctx, 'mass', 'galerkin')

const direct = await nxr.eigsPencil(K, M, { k: 6, normalize: true })
const iter   = await nxr.eigsPencil(K, M, { k: 6, normalize: true, iterative: true })

check(direct.mode === 'direct', `default path reports mode='direct' (got '${direct.mode}')`)
check(iter.mode === 'iterative', `iterative:true reports mode='iterative' (got '${iter.mode}')`)
check(iter.eigenvalues.length === 6, 'iterative returns k=6 eigenvalues')

let maxErr = 0
for (let i = 0; i < 6; i++) maxErr = Math.max(maxErr, Math.abs(iter.eigenvalues[i] - direct.eigenvalues[i]))
check(maxErr < 1e-6, `iterative eigenvalues match direct (maxErr ${maxErr.toExponential(2)})`)
check(Math.abs(iter.eigenvalues[0]) < 1e-6, 'iterative λ0 ≈ 0 (constant mode)')
check(iter.eigenvalues.every((v, i, a) => i === 0 || v >= a[i-1] - 1e-9), 'iterative eigenvalues ascending')

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
