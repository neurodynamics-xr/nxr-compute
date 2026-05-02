// Smoke test for the N-API addon's structured-error contract.
//
// Every synchronous addon entry point catches nxr::compute::Error
// and re-raises as a Napi::Error with .code / .hint set to the
// string-named ErrorCode enumerator. This smoke calls a handful of
// methods with bad input and asserts the structured fields are
// populated — without this, JS callers can't switch on err.code,
// only pattern-match on err.message which is brittle.

import path from 'node:path'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, '$1')), '..')
const addon = require(path.join(repoRoot, 'nxr_compute_addon.node'))

// ── Icosahedron fixture ──────────────────────────────────────
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

const ctx = addon.createContext(verts, faces)

let failures = 0
function assertThrows(label, expectedCode, fn) {
  try {
    fn()
    console.error(`FAIL: ${label} — expected throw, got none`)
    failures++
    return
  } catch (err) {
    if (err.code !== expectedCode) {
      console.error(`FAIL: ${label} — expected .code === '${expectedCode}', got ${JSON.stringify(err.code)}`)
      console.error(`  message: ${err.message}`)
      failures++
      return
    }
    if (!err.message.startsWith(`[${expectedCode}]`)) {
      console.error(`FAIL: ${label} — message should start with [${expectedCode}], got ${JSON.stringify(err.message)}`)
      failures++
      return
    }
    console.log(`  ${label} ✓ (code=${err.code}, hint=${err.hint ?? '∅'})`)
  }
}

console.log('[errors] verifying structured error translation')

// 1. Out-of-range source vertex → InvalidInput from C++ checkVertexInRange.
assertThrows('vectorHeatLogMap with out-of-range vertex', 'INVALID_INPUT',
  () => addon.vectorHeatLogMap(ctx, /* sourceVertex = */ 9999))

// 2. Out-of-range source vertex via transport.
assertThrows('vectorHeatTransport with out-of-range vertex', 'INVALID_INPUT',
  () => addon.vectorHeatTransport(ctx,
    new Int32Array([99]),
    new Float64Array([1, 0, 0])))

// 3. Mismatched array shape — Nx3 sourceVectors length must be 3*verts.
assertThrows('vectorHeatTransport with mismatched array lengths', 'INVALID_INPUT',
  () => addon.vectorHeatTransport(ctx,
    new Int32Array([0, 1]),
    new Float64Array([1, 0, 0])))  // length 3, but 2 verts ⇒ need 6

// 4. Empty curve to signed heat.
assertThrows('signedHeatDistance on empty curve', 'INVALID_INPUT',
  () => addon.signedHeatDistance(ctx, new Int32Array([]), true, 1))

// 5. Empty source set to findCenter.
assertThrows('vectorHeatFindCenter on empty source set', 'INVALID_INPUT',
  () => addon.vectorHeatFindCenter(ctx, new Int32Array([]), 2))

// 6. NotPrecomputed: heat diffusion before solveEigenmodes.
assertThrows('generateHeatDiffusion before solveEigenmodes', 'NOT_PRECOMPUTED',
  () => addon.generateHeatDiffusion(ctx, {
    sourceVerts: new Int32Array([0]),
    sourceValues: new Float64Array([1.0]),
    timesteps: new Float64Array([0.0, 0.1]),
    alpha: 1.0,
  }))

// 7. nSym < 1 ⇒ InvalidInput from computeSmoothFaceField.
assertThrows('computeSmoothFaceField with nSym=0', 'INVALID_INPUT',
  () => addon.computeSmoothFaceField(ctx, 0, false))

if (failures > 0) {
  console.error(`[errors] ${failures} failure(s)`)
  process.exit(1)
}
console.log('[errors] all assertions passed ✓')
