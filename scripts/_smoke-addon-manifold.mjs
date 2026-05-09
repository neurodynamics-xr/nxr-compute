// Smoke test for the N-API addon's six-group `nxr.manifold.*` shim.
// Mirrors `scripts/_smoke-wasm.mjs` but loads the native addon via
// `bindings/node/index.mjs` and exercises the same nested namespace.
//
// Verifies that the JS shim correctly forwards to the addon's flat
// surface for every group, and that not-yet-wired leaves throw a
// stable Error with code === 'NOT_WIRED_IN_ADDON'.

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

let failures = 0
function check(cond, msg) {
  if (cond) console.log(`  ✓ ${msg}`)
  else { console.error(`  ✗ ${msg}`); failures++ }
}

const mctx = nxr.createManifoldContext(verts, faces)

// ── nV / nE / nF accessors ───────────────────────────────────
check(mctx.nV() === 12, 'nV === 12')
check(mctx.nF() === 20, 'nF === 20')
check(mctx.nE() === 30, 'nE === 30')

// ── operator group ───────────────────────────────────────────
const stiffness = mctx.operator.stiffness()
check(stiffness && stiffness.rows === 12 && stiffness.cols === 12,
      'operator.stiffness() returns 12×12 sparse')

const d0 = mctx.operator.d0()
check(d0 && d0.rows === 30 && d0.cols === 12,
      'operator.d0() returns 30×12 sparse')

const cl = mctx.operator.connectionLaplacian({ domain: 'vertex', nSym: 4 })
check(cl.format === 'real2N' && cl.outputDim === 24,
      'operator.connectionLaplacian({vertex, nSym=4}) → real2N 24×24')

const cl2 = mctx.operator.connectionLaplacian({ domain: 'face', nSym: 2 })
check(cl2.outputDim === 40,
      'operator.connectionLaplacian({face, nSym=2}) → real2N 40×40')

// ── solve group ──────────────────────────────────────────────
// solve.eigen is async (addon runs it in a libuv worker thread).
// The addon implicitly normalizes + drops the DC mode, so 6 requested
// → 5 returned (the constant mode at λ ≈ 0 is removed).
const eig = await mctx.solve.eigen(6)
check(eig.k === eig.eigenvalues.length,
      'solve.eigen result is internally consistent (k === eigenvalues.length)')
check(eig.eigenvalues.length === 5,
      `solve.eigen(6) → 5 modes after DC removal (got ${eig.eigenvalues.length})`)
check(eig.eigenvalues.every(Number.isFinite),
      'all eigenvalues finite')
check(eig.eigenvalues.every(v => v > -1e-9),
      'all eigenvalues non-negative')

const phi = mctx.solve.poisson([0, 3], [1.0, -1.0])
check(phi.length === 12,
      'solve.poisson(sources, vals) → φ length 12')

// ── measure group ────────────────────────────────────────────
const dist = mctx.measure.distance([0])
check(dist.length === 12 && dist[0] < 1e-9 && dist.every(Number.isFinite),
      'measure.distance([0]) → length 12, d(0,0) ≈ 0')

const distSigned = mctx.measure.distance.signed([0, 1, 2], false)
check(distSigned.length === 12 && distSigned.every(Number.isFinite),
      'measure.distance.signed(...) returns finite values')

const curv = mctx.measure.curvature()
check(curv.gaussian.length === 12,
      'measure.curvature() returns per-vertex K')

// ── interpolate group ───────────────────────────────────────
const sff = mctx.interpolate.smoothFaceField(4, false)
check(sff.length === 60,   // 20 faces * 3
      'interpolate.smoothFaceField(4) → 20×3 vectors')

const svf = mctx.interpolate.smoothVertexField(2, false)
check(svf.vertexFieldRaw.length === 24,   // 12 verts * 2
      'interpolate.smoothVertexField(2) → 12×2 raw + lifted')

// ── query group ─────────────────────────────────────────────
const v = mctx.query.vertex(7)
check(v.vertexIndex === 7, 'query.vertex(7) → { vertexIndex: 7 }')

const center = mctx.query.center([0, 1, 2], 2)
check(center.length === 3 && center.every(Number.isFinite),
      'query.center([0,1,2]) → 3D point')

// ── uv group: stripes (rest are NOT_WIRED) ───────────────────
const stripes = mctx.uv.stripe(svf.vertexFieldRaw, 3.0, true)
check(stripes.segmentCount > 0,
      `uv.stripe(...) → ${stripes.segmentCount} segments`)

// ── NOT_WIRED_IN_ADDON leaves: stable error code ─────────────
function expectNotWired(label, fn) {
  try { fn(); failures++; console.error(`  ✗ ${label} — expected throw`) }
  catch (e) {
    if (e.code === 'NOT_WIRED_IN_ADDON') console.log(`  ✓ ${label} throws NOT_WIRED_IN_ADDON`)
    else { failures++; console.error(`  ✗ ${label} — wrong code: ${e.code}, msg=${e.message}`) }
  }
}
expectNotWired('measure.frame',         () => mctx.measure.frame())
expectNotWired('uv.bff',                () => mctx.uv.bff())
expectNotWired('operator.star0',        () => mctx.operator.star0())
expectNotWired('operator.star2',        () => mctx.operator.star2())
expectNotWired('operator.star1Inverse', () => mctx.operator.star1Inverse())

// ── functional namespace forwarding ─────────────────────────
const eig2 = await nxr.nxr.manifold.solve.eigen(mctx, 4)
check(eig2.eigenvalues.length === 3,
      `nxr.nxr.manifold.solve.eigen(mctx, 4) forwards correctly (4 requested - DC removed = 3 modes, got ${eig2.eigenvalues.length})`)

const cl3 = nxr.nxr.manifold.operator.connectionLaplacian(mctx, { domain: 'edge', nSym: 4 })
check(cl3.outputDim === 60,   // 30 edges * 2
      'nxr.nxr.manifold.operator.connectionLaplacian(mctx, {edge}) forwards correctly')

console.log(failures === 0
  ? '\n[smoke-addon-manifold] all assertions passed ✓'
  : `\n[smoke-addon-manifold] ${failures} FAILED ✗`)
process.exit(failures === 0 ? 0 : 1)
