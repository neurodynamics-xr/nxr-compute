// createContext(v, f, { intrinsicDelaunay: true }): the vertex connection Laplacian on intrinsic-Delaunay cotan weights.
// Run from repo root:
//   bash scripts/build.sh Release && node bindings/node/test/test_intrinsic_delaunay.mjs
//
// WHY: on the REAL cortex (cortex_pial_low, one hemisphere, 10 242 vertices) the vertex connection Laplacian assembled on
// the embedded mesh is INDEFINITE — λ_min −790 nearest zero and down to −2e4 (measured 2026-09-29) — while the scalar cotan
// Laplacian is PSD regardless (a sum of per-triangle gradient energies; its spectrum barely moves: 132.7 → 132.1). On the
// intrinsic Delaunay triangulation the connection Laplacian is PSD (lowest 1716 on that hemisphere). A jittered sphere is
// not enough to break the raw operator (it stays PSD with 137 non-Delaunay edges), so this test proves what it can on a
// synthetic mesh: the flag TAKES EFFECT (the cotan weights change, the extrinsic Dirac does not), the intrinsic operator is
// PSD with the sphere's Bochner level, and vertexFrames remain its gauge.
import { initNxrCompute } from '../index.mjs'

let failures = 0
const check = (cond, msg) => { if (cond) console.log(`  ✓ ${msg}`); else { console.error(`  ✗ ${msg}`); failures++ } }
const nxr = await initNxrCompute()

function icosphere(levels) {
  const t = (1 + Math.sqrt(5)) / 2
  let V = [[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]]
  let F = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
  const unit = (p) => { const l = Math.hypot(...p); return p.map((x) => x / l) }
  V = V.map(unit)
  for (let s = 0; s < levels; s++) {
    const mid = new Map(), NF = []
    const m = (a, b) => { const k = a < b ? `${a},${b}` : `${b},${a}`; if (!mid.has(k)) { mid.set(k, V.length); V.push(unit([0, 1, 2].map((i) => (V[a][i] + V[b][i]) / 2))) } return mid.get(k) }
    for (const [a, b, c] of F) { const ab = m(a, b), bc = m(b, c), ca = m(c, a); NF.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]) }
    F = NF
  }
  return { V, F }
}
// jitter each vertex along the sphere by a random tangent step, then back onto the sphere — obtuse triangles, same shape
const { V, F } = icosphere(3)
let seed = 11; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5
const J = V.map((p) => { const q = [p[0] + 0.12 * rnd(), p[1] + 0.12 * rnd(), p[2] + 0.12 * rnd()]; const l = Math.hypot(...q); return q.map((x) => x / l) })
const verts = Float64Array.from(J.flat()), faces = Int32Array.from(F.flat()), nV = J.length

const blocked = (A) => {
  const n = A.data.length, row = new Int32Array(2 * n), col = new Int32Array(2 * n), data = new Float64Array(2 * n)
  for (let k = 0; k < n; k++) { row[k] = A.row[k]; col[k] = A.col[k]; data[k] = A.data[k]; row[n + k] = A.row[k] + nV; col[n + k] = A.col[k] + nV; data[n + k] = A.data[k] }
  return { row, col, data, rows: 2 * nV, cols: 2 * nV }
}
const lowest = async (ctx) => {
  const CL = nxr.assembleConnectionLaplacian(ctx, { domain: 'vertex', nSym: 1 })
  const M = nxr.operators(ctx, 'mass', 'lumped')
  const r = await nxr.eigsPencil(CL.K, blocked(M), { k: 10, normalize: true })
  return { lam: Array.from(r.eigenvalues), CL, M }
}
const raw = nxr.createContext(verts, faces)
const idt = nxr.createContext(verts, faces, { intrinsicDelaunay: true })

// the jitter must make NON-DELAUNAY edges (opposite angles summing past π — a negative cotan weight), or it proves nothing
const ang = (a, b, c) => { const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]; return Math.acos((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v))) }
const opposite = new Map()
for (const f of F) for (let k = 0; k < 3; k++) {
  const i = f[k], j = f[(k + 1) % 3], o = f[(k + 2) % 3], key = i < j ? `${i},${j}` : `${j},${i}`
  opposite.set(key, (opposite.get(key) ?? 0) + ang(J[o], J[i], J[j]))
}
const nonDelaunay = [...opposite.values()].filter((s) => s > Math.PI + 1e-9).length
check(nonDelaunay > 10, `the jittered sphere has non-Delaunay edges (${nonDelaunay} of ${opposite.size})`)

const r0 = await lowest(raw), r1 = await lowest(idt)
// the flag takes effect: the non-Delaunay edges are flipped, so the intrinsic cotan Laplacian differs
const L0 = nxr.operators(raw, 'laplacian', 'cotan'), L1 = nxr.operators(idt, 'laplacian', 'cotan')
const same = L0.data.length === L1.data.length && L0.data.every((x, k) => Math.abs(x - L1.data[k]) < 1e-12 && L0.row[k] === L1.row[k] && L0.col[k] === L1.col[k])
check(!same, `the intrinsic context's cotan Laplacian differs from the embedded one (nnz ${L0.data.length} vs ${L1.data.length})`)
check(r1.lam[0] > -1e-8, `intrinsic Delaunay: the connection Laplacian is PSD (λ_min ${r1.lam[0].toExponential(2)}; embedded ${r0.lam[0].toFixed(3)})`)
check(r1.lam.slice(0, 6).every((x) => Math.abs(x - 1) < 0.05), `…its six lowest ≈ 1, the sphere's Bochner level (${r1.lam.slice(0, 6).map((x) => x.toFixed(3)).join(' ')})`)

// the frames are still the gauge on the intrinsic triangulation (connection_laplacian.cpp: the vertex gauge coincides) —
// on a mesh this irregular a smooth field's DISCRETE energy is inflated (embedded 4.1), so the check is relative: the
// intrinsic energy is lower than the embedded one and far below a wrong gauge's
const vf = nxr.vertexFrames(idt)
const q = (A, y) => { let s = 0; for (let k = 0; k < A.data.length; k++) s += y[A.row[k]] * A.data[k] * y[A.col[k]]; return s }
const energy = (r, turn) => {
  const a = [0, 0, 1], x = new Float64Array(2 * nV)
  for (let i = 0; i < nV; i++) {
    const n = [vf.normals[3 * i], vf.normals[3 * i + 1], vf.normals[3 * i + 2]], an = a[0] * n[0] + a[1] * n[1] + a[2] * n[2]
    const X = [a[0] - an * n[0], a[1] - an * n[1], a[2] - an * n[2]]
    let re = X[0] * vf.e1[3 * i] + X[1] * vf.e1[3 * i + 1] + X[2] * vf.e1[3 * i + 2], im = X[0] * vf.e2[3 * i] + X[1] * vf.e2[3 * i + 1] + X[2] * vf.e2[3 * i + 2]
    if (turn) { const c = Math.cos(turn[i]), s = Math.sin(turn[i]); [re, im] = [c * re - s * im, s * re + c * im] }
    x[i] = re; x[nV + i] = im
  }
  return q(r.CL.K, x) / (q(r.M, x.subarray(0, nV)) + q(r.M, x.subarray(nV)))
}
const eI = energy(r1), eR = energy(r0), eT = energy(r1, Float64Array.from({ length: nV }, () => 2 * Math.PI * (rnd() + 0.5)))
check(eI < eR && eI < 2, `vertexFrames on the intrinsic context: grad of z, energy ${eI.toFixed(3)} (embedded ${eR.toFixed(3)})`)
check(eT > 20 * eI, `…and ${eT.toFixed(1)} with each frame turned at random — the frames are the gauge`)

// the flag changes only the intrinsic operators' weights: the Dirac (extrinsic) is the same on both contexts
const d0 = nxr.operators(raw, 'diracD'), d1 = nxr.operators(idt, 'diracD')
check(d0.data.length === d1.data.length && d0.data.every((x, i) => x === d1.data[i]), 'the extrinsic Dirac D is identical on both contexts')

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
