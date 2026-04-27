// One-off smoke test for the time-varying field generators.
//
// Loads cxf_addon.node directly (no Electron), builds a small
// icosahedron mesh, solves eigenmodes, then exercises both new
// N-API bindings end-to-end. Verifies output shapes, sanity-checks
// values, and prints timing.
//
// Usage (from repo root):
//   node scripts/_smoke-time-varying.mjs

import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..')
const require = createRequire(import.meta.url)

const addon = require(path.join(repoRoot, 'cxf_addon.node'))

// ── Icosahedron mesh (matches test_eigen.cpp / test_factor_cache.cpp) ──
const t = (1 + Math.sqrt(5)) / 2
const rawVerts = [
  -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
   0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
   t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
]
const verts = new Float64Array(36)
for (let i = 0; i < 36; i += 3) {
  const len = Math.hypot(rawVerts[i], rawVerts[i + 1], rawVerts[i + 2])
  verts[i]     = rawVerts[i]     / len
  verts[i + 1] = rawVerts[i + 1] / len
  verts[i + 2] = rawVerts[i + 2] / len
}
const faces = new Int32Array([
  0,11,5, 0,5,1,  0,1,7,  0,7,10, 0,10,11,
  1,5,9,  5,11,4, 11,10,2,10,7,6, 7,1,8,
  3,9,4,  3,4,2,  3,2,6,  3,6,8,  3,8,9,
  4,9,5,  2,4,11, 6,2,10, 8,6,7,  9,8,1,
])
const nV = verts.length / 3   // 12
const nF = faces.length / 3   // 20
console.log(`[smoke] mesh: nV=${nV}, nF=${nF}`)

// ── Build context + ops + eigenmodes ──────────────────────────────
const ctx = addon.createContext(verts, faces)
const ops = addon.assembleMeshOperators(ctx)
console.log(`[smoke] ops: nV=${ops.nV}, nE=${ops.nE}, nF=${ops.nF}, totalArea=${ops.totalArea.toFixed(4)}`)

const k = 6
const eig = await addon.solveEigenmodes(ctx, k)
console.log(`[smoke] eigenmodes: k=${eig.k} (post-removeDC), nV=${eig.nV}`)
const eigvals = Array.from(eig.eigenvalues).map(v => v.toFixed(4))
console.log(`[smoke] eigenvalues: [${eigvals.join(', ')}]`)

// ── generateHeatDiffusion ─────────────────────────────────────────
//
// Initial condition: positive delta at vertex 0, negative at vertex 3
// (the two are antipodal on an icosahedron, so this is a clean
// dipole). Diffuse over 5 timesteps.
{
  const sourceVerts  = new Int32Array([0, 3])
  const sourceValues = new Float64Array([1.0, -1.0])
  const timesteps    = new Float64Array([0.0, 0.1, 0.5, 2.0, 10.0])
  const alpha        = 1.0

  const t0 = performance.now()
  const heat = addon.generateHeatDiffusion(ctx, {
    sourceVerts, sourceValues, timesteps, alpha,
  })
  const elapsed = (performance.now() - t0).toFixed(3)

  console.log(`[smoke] heat diffusion: ${elapsed} ms`)
  console.log(`[smoke]   shape: T=${heat.T}, nV=${heat.nV}, data.length=${heat.data.length}`)

  if (heat.T !== 5)        throw new Error(`expected T=5, got ${heat.T}`)
  if (heat.nV !== 12)      throw new Error(`expected nV=12, got ${heat.nV}`)
  if (heat.data.length !== 60) throw new Error(`expected 60 floats, got ${heat.data.length}`)

  // Per-frame diagnostics: max|u(t)| should decrease over time as the
  // dipole spreads and cancels.
  for (let ti = 0; ti < 5; ti++) {
    const offset = ti * 12
    let amax = 0
    for (let v = 0; v < 12; v++) {
      const x = Math.abs(heat.data[offset + v])
      if (x > amax) amax = x
    }
    console.log(`[smoke]   t=${timesteps[ti].toFixed(2)}  max|u|=${amax.toFixed(6)}`)
  }

  // Sanity: at t=10 (effectively asymptotic), u → uniform (mean is
  // ~0 since the dipole has zero net mass, so all values should be tiny).
  const tail = heat.data.slice(48, 60)  // last frame
  const tailMax = Math.max(...Array.from(tail).map(Math.abs))
  if (tailMax > 1e-3) throw new Error(`expected u(t=10) ≈ 0, got max|u|=${tailMax}`)
  console.log(`[smoke]   ✓ asymptotic decay (max|u(10)|=${tailMax.toExponential(3)})`)
}

// ── generateDampedWave ────────────────────────────────────────────
//
// Excite mode 0 with amplitude 1.0 and small damping, evaluate over
// one full period (T = 2π / √λ_0). Expect amplitude to decrease
// monotonically along the time axis.
{
  const lambda0 = eig.eigenvalues[0]
  const period  = (2 * Math.PI) / Math.sqrt(lambda0)
  const T = 21
  const timesteps = new Float64Array(T)
  for (let i = 0; i < T; i++) timesteps[i] = (i / (T - 1)) * period

  const t0 = performance.now()
  const wave = addon.generateDampedWave(ctx, {
    modeIndices: new Int32Array([0]),
    amplitudes:  new Float64Array([1.0]),
    dampings:    new Float64Array([0.2]),
    phases:      new Float64Array([0.0]),
    timesteps,
  })
  const elapsed = (performance.now() - t0).toFixed(3)

  console.log(`[smoke] damped wave: ${elapsed} ms`)
  console.log(`[smoke]   shape: T=${wave.T}, nV=${wave.nV}, lambda_0=${lambda0.toFixed(3)}, period=${period.toFixed(3)}`)

  if (wave.T !== T)        throw new Error(`expected T=${T}, got ${wave.T}`)
  if (wave.nV !== 12)      throw new Error(`expected nV=12, got ${wave.nV}`)

  // Initial frame should be amplitude · phi_0 (cos(0) = 1, exp(0) = 1).
  // Just check it's not all zeros.
  const frame0 = wave.data.slice(0, 12)
  const frame0Max = Math.max(...Array.from(frame0).map(Math.abs))
  if (frame0Max < 1e-3) throw new Error(`expected non-trivial frame 0, got max|u|=${frame0Max}`)
  console.log(`[smoke]   ✓ frame 0 nontrivial (max|u|=${frame0Max.toFixed(4)})`)

  // Final frame (one period later) should have amplitude attenuated
  // by exp(-γ · period) = exp(-0.2 · 4.44) ≈ 0.41.
  const frameLast = wave.data.slice((T - 1) * 12, T * 12)
  const lastMax = Math.max(...Array.from(frameLast).map(Math.abs))
  const expectedRatio = Math.exp(-0.2 * period)
  const observedRatio = lastMax / frame0Max
  console.log(`[smoke]   damping check: expected ratio≈${expectedRatio.toFixed(3)}, observed=${observedRatio.toFixed(3)}`)
  if (Math.abs(observedRatio - expectedRatio) > 0.01) {
    throw new Error(`damping factor wrong: expected ${expectedRatio}, got ${observedRatio}`)
  }
  console.log(`[smoke]   ✓ damping factor matches exp(-γT)`)
}

console.log('[smoke] all addon time-varying generator assertions passed ✓')
