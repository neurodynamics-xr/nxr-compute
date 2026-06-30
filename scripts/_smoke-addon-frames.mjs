// Smoke test: addon frames() / measure.frame(). Asserts shape + orthonormality
// of the per-face frame (e1 ⟂ e2, e1 × e2 == normal).
// Usage (from repo root): node scripts/_smoke-addon-frames.mjs
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

const fr = m.measure.frame()
check(fr.e1.length === 20 * 3, 'e1 is [20×3]')
check(fr.e2.length === 20 * 3 && fr.normals.length === 20 * 3, 'e2 and normals are [20×3]')

let maxDot = 0, maxCross = 0
const dot = (a, b, o) => a[o] * b[o] + a[o + 1] * b[o + 1] + a[o + 2] * b[o + 2]
for (let f = 0; f < 20; f++) {
  const o = f * 3
  maxDot = Math.max(maxDot, Math.abs(dot(fr.e1, fr.e2, o)))
  // e1 × e2 should equal normals
  const cx = fr.e1[o+1]*fr.e2[o+2] - fr.e1[o+2]*fr.e2[o+1]
  const cy = fr.e1[o+2]*fr.e2[o]   - fr.e1[o]  *fr.e2[o+2]
  const cz = fr.e1[o]  *fr.e2[o+1] - fr.e1[o+1]*fr.e2[o]
  maxCross = Math.max(maxCross, Math.abs(cx-fr.normals[o]), Math.abs(cy-fr.normals[o+1]), Math.abs(cz-fr.normals[o+2]))
}
check(maxDot < 1e-9, `e1 ⟂ e2 (max |dot| ${maxDot.toExponential(2)})`)
check(maxCross < 1e-9, `e1 × e2 == normal (max err ${maxCross.toExponential(2)})`)

console.log(failures === 0 ? '\nPASS' : `\nFAIL (${failures})`)
process.exit(failures === 0 ? 0 : 1)
