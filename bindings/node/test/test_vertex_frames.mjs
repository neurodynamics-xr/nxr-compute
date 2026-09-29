// vertexFrames: the per-vertex tangent frame, and that it IS the vertex connection Laplacian's gauge.
// Run from repo root:
//   bash scripts/build.sh Release && node bindings/node/test/test_vertex_frames.mjs
//
// Geometric validity (unit, orthonormal, right-handed, deterministic) mirrors the MEX test. The GAUGE check is the one the
// MEX test defers: on the unit sphere the tangential projection of a constant vector is the gradient of an l = 1 harmonic,
// and its Bochner (connection-Laplacian) energy is λ₁ − Ric = 2 − 1 = 1. Written in these frames the Rayleigh quotient
// zᴴKz / zᴴMz is ≈ 1; written in any other gauge (each vertex's frame turned by its own angle, or mirrored) it is not.
//
// THE MASS IS LUMPED ⊗ I₂ (the registry's natural_mass). A tangent field's coordinates live in per-vertex frames, and the
// galerkin mass ⊗ I₂ couples neighbouring vertices' coordinates WITHOUT transporting them between frames — measured here,
// it moves the sphere's Bochner spectrum from 1, 5 (lumped, converged) to 1.83, 9.1 (galerkin, not converging).
import { initNxrCompute } from '../index.mjs'

let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }

const nxr = await initNxrCompute()

// ── an icosphere: the icosahedron subdivided three times onto the unit sphere (642 vertices) ──
function icosphere(levels) {
  const t = (1 + Math.sqrt(5)) / 2
  let V = [[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]]
  let F = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
  const unit = (p) => { const l = Math.hypot(...p); return p.map((x) => x / l) }
  V = V.map(unit)
  for (let s = 0; s < levels; s++) {
    const mid = new Map(), NF = []
    const m = (a, b) => {
      const k = a < b ? `${a},${b}` : `${b},${a}`
      if (!mid.has(k)) { mid.set(k, V.length); V.push(unit([0, 1, 2].map((i) => (V[a][i] + V[b][i]) / 2))) }
      return mid.get(k)
    }
    for (const [a, b, c] of F) { const ab = m(a, b), bc = m(b, c), ca = m(c, a); NF.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]) }
    F = NF
  }
  return { verts: Float64Array.from(V.flat()), faces: Int32Array.from(F.flat()), nV: V.length }
}
const { verts, faces, nV } = icosphere(3)
const ctx = nxr.createContext(verts, faces)

// ── 1. the frame: [nV×3] each, unit, orthonormal, right-handed, deterministic ──
const vf = nxr.vertexFrames(ctx)
check(vf.e1.length === 3 * nV && vf.e2.length === 3 * nV && vf.normals.length === 3 * nV, `e1, e2, normals are [nV×3] (nV = ${nV})`)
const row = (a, i) => [a[3 * i], a[3 * i + 1], a[3 * i + 2]]
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
let unitErr = 0, orthoErr = 0, handErr = 0, outward = 0
for (let i = 0; i < nV; i++) {
  const e1 = row(vf.e1, i), e2 = row(vf.e2, i), n = row(vf.normals, i)
  unitErr = Math.max(unitErr, Math.abs(dot(e1, e1) - 1), Math.abs(dot(e2, e2) - 1), Math.abs(dot(n, n) - 1))
  orthoErr = Math.max(orthoErr, Math.abs(dot(e1, e2)), Math.abs(dot(e1, n)), Math.abs(dot(e2, n)))
  const c = cross(e1, e2); handErr = Math.max(handErr, Math.hypot(c[0] - n[0], c[1] - n[1], c[2] - n[2]))
  if (dot(n, row(verts, i)) > 0.99) outward++
}
check(unitErr < 1e-9, `unit length (max err ${unitErr.toExponential(1)})`)
check(orthoErr < 1e-9, `orthonormal (max |dot| ${orthoErr.toExponential(1)})`)
check(handErr < 1e-9, `right-handed: e1 × e2 = n (max err ${handErr.toExponential(1)})`)
check(outward === nV, `normals point outward on the sphere (${outward}/${nV})`)
const vf2 = nxr.vertexFrames(ctx)
check(vf.e1.every((x, i) => x === vf2.e1[i]) && vf.e2.every((x, i) => x === vf2.e2[i]), 'deterministic')

// ── 2. the GAUGE: the connection Laplacian's energy of a gradient field is 1 in these frames, and not in any other ──
const CL = nxr.assembleConnectionLaplacian(ctx, { domain: 'vertex', nSym: 1 })
const M = nxr.operators(ctx, 'mass', 'lumped')
const Mg = nxr.operators(ctx, 'mass', 'galerkin')
check(CL.format === 'real2N' && CL.outputDim === 2 * nV, `the connection Laplacian is real2N, [2nV × 2nV] (BLOCK layout: re then im)`)
const quad = (A, x, y = x) => { let s = 0; for (let k = 0; k < A.data.length; k++) s += x[A.row[k]] * A.data[k] * y[A.col[k]]; return s }
/** zᴴKz / zᴴMz for the tangent field X written in per-vertex frames (e1, e2) turned by `turn[i]` (and mirrored if `mirror`). */
function energy(a, { turn = null, mirror = false, mass = M } = {}) {
  const x = new Float64Array(2 * nV)
  for (let i = 0; i < nV; i++) {
    const n = row(vf.normals, i), an = dot(a, n)
    const X = [a[0] - an * n[0], a[1] - an * n[1], a[2] - an * n[2]]
    let re = dot(X, row(vf.e1, i)), im = dot(X, row(vf.e2, i))
    if (mirror) im = -im
    if (turn) { const c = Math.cos(turn[i]), s = Math.sin(turn[i]); [re, im] = [c * re - s * im, s * re + c * im] }
    x[i] = re; x[nV + i] = im
  }
  const re = x.subarray(0, nV), im = x.subarray(nV)
  return quad(CL.K, x) / (quad(mass, re) + quad(mass, im))
}
let seed = 1; const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
const turn = Float64Array.from({ length: nV }, () => 2 * Math.PI * rand())
for (const [name, a] of [['x', [1, 0, 0]], ['z', [0, 0, 1]]]) {
  const e = energy(a), eTurned = energy(a, { turn }), eMirror = energy(a, { mirror: true })
  check(Math.abs(e - 1) < 0.01, `grad of ${name}: Bochner energy ≈ 1 in these frames, under the lumped mass (got ${e.toFixed(4)})`)
  const eG = energy(a, { mass: Mg })
  check(eG > 1.5, `…and NOT ≈ 1 under galerkin ⊗ I₂ (${eG.toFixed(3)}) — the natural mass is lumped`)
  check(eTurned > 10 * e, `…and ≫ 1 with each vertex's frame turned at random (${eTurned.toFixed(1)}) — the gauge is THESE frames`)
  check(eMirror > 1.5 * e, `…and > 1 with the frames mirrored, e2 → −e2 (${eMirror.toFixed(3)}) — the handedness is theirs too`)
}

// ── 3. the pencil (K, lumped ⊗ I₂) through eigsPencil: the sphere's Bochner spectrum l(l+1) − 1 — 1 (×6), 5 (×10) ──
const blocked = (A) => {
  const n = A.data.length, row = new Int32Array(2 * n), col = new Int32Array(2 * n), data = new Float64Array(2 * n)
  for (let k = 0; k < n; k++) { row[k] = A.row[k]; col[k] = A.col[k]; data[k] = A.data[k]; row[n + k] = A.row[k] + nV; col[n + k] = A.col[k] + nV; data[n + k] = A.data[k] }
  return { row, col, data, rows: 2 * nV, cols: 2 * nV }
}
/* k = 30 closes three whole levels (6 + 10 + 14). NOTE: a k that cuts INSIDE a degenerate cluster (k = 16 here) returns
   part of that cluster and then modes from the NEXT level — measured: 5 of the ten at 5, then five at 11. Ask for a margin. */
const eig = await nxr.eigsPencil(CL.K, blocked(M), { k: 30, normalize: true })
const lam = Array.from(eig.eigenvalues)
const near = (c, tol) => lam.filter((x) => Math.abs(x - c) < tol).length
check(near(1, 1e-2) === 6, `eigsPencil(K, lumped ⊗ I₂): six modes at l(l+1) − 1 = 1 (${lam.slice(0, 6).map((x) => x.toFixed(3)).join(' ')})`)
check(near(5, 0.1) === 10, `ten at 5 (${near(5, 0.1)})`)
check(near(11, 0.3) === 14, `fourteen at 11 (${near(11, 0.3)})`)

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
