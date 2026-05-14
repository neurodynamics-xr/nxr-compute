# Three.js integration guide

This guide walks through using **nxr-compute** as the computational backend of a
three.js application: installation, bootstrapping the WASM module,
lifecycle and memory management, and complete end-to-end example flows.

For a viz-by-viz cookbook (every nxr-compute output → three.js technique), see
[visualization-recipes.md](visualization-recipes.md).

> **Status**: the WASM binding ships. Prebuilt artifacts are committed
> to the repo at `dist/wasm/nxr_compute.js` and `dist/wasm/nxr_compute.wasm`,
> refreshed by `bash scripts/build-wasm.sh`. Smoke-tested end-to-end via
> `node scripts/_smoke-wasm.mjs`. Not on npm yet — install as a git
> dependency (see below).

---

## Installation

The package isn't published to npm; install it as a git dependency
pinned to a commit SHA or tag:

```json
// package.json
{
  "dependencies": {
    "@neurodynamics-xr/nxr-compute": "github:neurodynamics-xr/nxr-compute#<sha-or-tag>"
  }
}
```

The git install ships:

- `dist/wasm/nxr_compute.wasm` — the compiled WASM binary (~1.2 MB, gzip ~400-600 KB)
- `dist/wasm/nxr_compute.js` — the Emscripten factory + Embind glue
- `bindings/wasm/js/index.mjs` — the JS shim (`initNxrCompute` + the `nxr.manifold.*` namespace)
- `bindings/wasm/js/index.d.ts` — TypeScript declarations

The shim's `import` statement points at `dist/wasm/nxr_compute.js`
relatively, so consumers don't need to wire the WASM path manually.

### Vite / webpack configuration

WASM modules in three.js apps usually need explicit asset handling.
Vite is friendliest:

```javascript
// vite.config.js
export default {
  // The shim imports nxr_compute.js, which fetches nxr_compute.wasm at
  // runtime relative to itself. Vite handles this automatically when
  // the package is in node_modules; just opt out of optimisation:
  optimizeDeps: { exclude: ['@neurodynamics-xr/nxr-compute'] }
}
```

For webpack 5, enable `experiments.asyncWebAssembly = true` and ensure
`.wasm` is treated as a static asset (`type: 'asset/resource'` in
the asset module rule).

---

## Bootstrap

```javascript
import { initNxrCompute } from '@neurodynamics-xr/nxr-compute'

const nxrCompute = await initNxrCompute({
  // optional — usually not needed; the loader auto-discovers nxr_compute.wasm
  // alongside nxr_compute.js
  locateFile: (filename) => `/assets/${filename}`,
})

// `nxr-compute` now exposes the API: nxrCompute.createContext, nxrCompute.version, etc.
console.log(nxrCompute.version())  // → "nxr-compute 0.1.0"
```

The init is async because Emscripten loads the WASM binary
asynchronously. Do this once at app startup; the resolved `nxr-compute` object is
safe to share across components.

### React example (with @react-three/fiber)

```jsx
import { createContext, useContext, useEffect, useState } from 'react'
import { initNxrCompute } from '@neurodynamics-xr/nxr-compute'

const NxrComputeContext = createContext(null)

export function NxrComputeProvider({ children }) {
  const [nxrCompute, setNxrCompute] = useState(null)
  useEffect(() => { initNxrCompute().then(setNxrCompute) }, [])
  if (!nxrCompute) return null  // or a loading spinner
  return <NxrComputeContext.Provider value={nxrCompute}>{children}</NxrComputeContext.Provider>
}

export const useNxrCompute = () => useContext(NxrComputeContext)
```

---

## Building a context

A `ComputeContext` is a stateful handle that owns one mesh and its
derived structure (operators, Cholesky factors, eigenmodes — whatever
you've computed so far). Create it once per mesh:

```javascript
const ctx = nxrCompute.createContext(verticesFloat64, facesInt32)
```

### Argument formats

- `vertices`: `Float64Array` of length `V × 3`. Row-major xyz triples
  (`x0, y0, z0, x1, y1, z1, …`). This matches three.js's
  `BufferAttribute('position')` storage exactly.
- `faces`: `Int32Array` of length `F × 3`. Row-major vertex indices,
  **0-based** (matches three.js conventions; if your mesh source uses
  1-based indexing, subtract 1 before passing in).

### Bringing your own mesh data

nxr-compute doesn't load files — you do. Common sources:

```javascript
// From a three.js loader (OBJ, PLY, GLTF, …)
const geometry = await new OBJLoader().loadAsync('/mesh.obj')
const vertices = new Float64Array(geometry.attributes.position.array)
const faces    = new Int32Array(geometry.index.array)

// From an HTTP fetch of a custom binary format
const buffer = await fetch('/mesh.bin').then(r => r.arrayBuffer())
const { vertices, faces } = parseMyFormat(buffer)

// From a Zarr store via @zarrita/core
import { open as zarrOpen, get as zarrGet } from '@zarrita/core'
const root = await zarrOpen(myStore)
const verticesNode = await zarrOpen(myStore, { path: '/manifold/vertices', kind: 'array' })
const vertices = new Float64Array((await zarrGet(verticesNode)).data)
// ... same for faces

// From cortical-flow's CLI output (which produces Zarr stores
// readable by zarr-fs.ts)
```

nxr-compute is agnostic to source. It only sees the typed arrays.

---

## Running compute

### The "visualization defaults" precompute

For a viewer that wants the standard set of visualization-essential
outputs in one call:

```javascript
const data = ctx.precompute({ k: 300 })

// data.operators  — { cotanLaplacian, mass, vertexDualAreas,
//                     vertexNormals, totalArea, nV, nE, nF }
// data.dec        — { d0, d1, hodge0, hodge1, hodge1Inverse, hodge2 }
// data.eigenmodes — { eigenvectors [V*K float64],
//                     eigenvalues  [K   float64], k, nConverged }
// data.faceFrames — { e1 [F*3 float64],
//                     e2 [F*3 float64],
//                     normals [F*3 float64] }
```

This single call gives a three.js viewer everything it needs for static
mesh display, scalar field colormaps, eigenmode visualization, and
GPU-side tangent vector reconstruction.

### Per-operation calls

Anything not in the precompute pack is on-demand:

```javascript
// Geodesic distance from one or more source vertices (heat method)
const distances = ctx.computeGeodesicDistance(new Int32Array([1024, 4096]))
// → Float64Array [V] of distances

// Geodesic path between two vertices (flip-out)
const path = ctx.tracePath(vStart, vEnd)
// → Float64Array [N * 3] of polyline points

// Hodge decomposition of a 1-form ω on edges
// On the N-API addon, hodgeDecompose returns a Promise — the solve
// runs in a libuv worker thread so the JS event loop stays free.
// WASM is sync today; wrap in a Web Worker per docs/wasm-web-worker.md
// if you need it off the main thread in the browser.
const hodge = await ctx.hodgeDecompose(omegaFloat64)
// → { exactPotential, coExactPotentialV, gamma,
//     dAlphaVectors, deltaBetaVectors, gammaVectors, … }

// BFF parametrization (requires open mesh — closed meshes throw)
const uvs = ctx.computeUVCoordinates()
// → Float64Array [V * 2]

// Curvatures
const curvatures = ctx.computeCurvatures()
// → { gaussian, mean, kMin, kMax, principalDir }

// Time-varying generators (need eigenmodes computed first)
const heat = ctx.generateHeatDiffusion(
  new Int32Array([0]),                                  // sources
  new Float64Array([1.0]),                              // amplitudes
  Float64Array.from({ length: 200 }, (_, i) => i * 0.01), // timesteps
  1.0                                                    // alpha
)
// → { data: Float32Array [T * V], T, nV }
```

See [visualization-recipes.md](visualization-recipes.md) for what to do
with each output.

---

## Lifecycle and memory management

The `ComputeContext` holds C++ state in WASM linear memory:

- The `ManifoldSurfaceMesh` and `VertexPositionGeometry` structures
- Cached operators (`MeshOperators`, `DECOperators`)
- Cached Cholesky / LU factors (`CholeskyCache`)
- Cached eigenmodes (after the first eigensolve)

This state survives across calls — that's why a Hodge solve after a
Poisson solve is fast, even though both depend on the cotangent
Laplacian. The factor was built once and cached.

When you're done with a mesh, **release the context** to free the WASM
heap:

```javascript
ctx.delete()
```

After `delete()`, the handle is invalid. Don't call methods on it.

### Sharing a context across React components

Keep the context in a state container that survives renders. With
@react-three/fiber:

```jsx
function MeshAnalysis({ vertices, faces }) {
  const nxrCompute = useNxrCompute()
  const ctxRef = useRef(null)
  const [precomputed, setPrecomputed] = useState(null)

  useEffect(() => {
    if (!nxrCompute) return
    ctxRef.current = nxrCompute.createContext(vertices, faces)
    setPrecomputed(ctxRef.current.precompute({ k: 300 }))
    return () => {
      ctxRef.current?.delete()
      ctxRef.current = null
    }
  }, [nxr-compute, vertices, faces])

  // Use precomputed.* in your three.js components
  return precomputed ? <BrainViewer data={precomputed} /> : null
}
```

The cleanup function in `useEffect` ensures contexts are released when
the mesh changes or the component unmounts.

### Memory expectations

For a typical cortical mesh (fsaverage6, ~40k vertices) with k=300
eigenmodes computed:

| What | Size |
|---|---|
| ManifoldSurfaceMesh + VertexPositionGeometry | ~10 MB |
| MeshOperators (sparse L, M, vertexDualAreas, vertexNormals) | ~5 MB |
| DECOperators (sparse d0, d1, hodge stars) | ~10 MB |
| CholeskyCache (3 sparse factors) | ~30 MB |
| Eigenvectors (40k × 300 × 8 bytes float64) | ~96 MB |

Total: ~150 MB in WASM heap per loaded mesh. Manageable; the WASM heap
is sized to handle several meshes simultaneously if needed. For tighter
memory budgets:

- Skip precompute and only compute what you need (e.g., omit eigenmodes
  if you don't need spectral viz).
- Solve fewer eigenmodes (`k: 100` is plenty for many tasks).
- Use `Float32Array` views where you don't need full float64 precision
  (the WASM side computes in float64; downcast at the boundary).
- Release contexts you don't need (`ctx.delete()`).

---

## Three end-to-end example flows

### 1. Static mesh + eigenmode visualization

The minimum viable nxr-compute-backed three.js scene: load a mesh, compute the
spectral basis, show one eigenmode as a colormap.

```javascript
import * as THREE from 'three'
import { initNxrCompute } from '@neurodynamics-xr/nxr-compute'

const nxrCompute = await initNxrCompute()

// Load mesh data however you like
const { vertices, faces } = await loadMyMesh()
const verticesF64 = new Float64Array(vertices)  // V*3
const facesI32    = new Int32Array(faces)       // F*3

// Compute
const ctx = nxrCompute.createContext(verticesF64, facesI32)
const data = ctx.precompute({ k: 50 })

// three.js geometry
const geometry = new THREE.BufferGeometry()
geometry.setAttribute('position', new THREE.BufferAttribute(
  new Float32Array(verticesF64), 3))
geometry.setAttribute('normal',   new THREE.BufferAttribute(
  new Float32Array(data.operators.vertexNormals), 3))
geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(facesI32), 1))

// Pick eigenmode 5 (the 5th non-DC mode, indices are 0-based)
const nV = data.operators.nV
const mode5 = data.eigenmodes.eigenvectors.subarray(5 * nV, 6 * nV)
geometry.setAttribute('scalar',
  new THREE.BufferAttribute(new Float32Array(mode5), 1))

// Custom shader material that maps `scalar` through a colormap LUT
const material = new THREE.ShaderMaterial({
  vertexShader: /* glsl */`
    attribute float scalar;
    varying float vScalar;
    void main() {
      vScalar = scalar;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.);
    }
  `,
  fragmentShader: /* glsl */`
    varying float vScalar;
    uniform sampler2D colormap;
    void main() {
      // map scalar from [-1,1] to [0,1] and look up in colormap LUT
      float t = clamp(vScalar * 0.5 + 0.5, 0., 1.);
      gl_FragColor = texture(colormap, vec2(t, 0.5));
    }
  `,
  uniforms: { colormap: { value: viridisLUT } },
})

scene.add(new THREE.Mesh(geometry, material))
```

To swap modes interactively, just update the `scalar` BufferAttribute
without recomputing anything:

```javascript
function showEigenmode(k) {
  const mode = data.eigenmodes.eigenvectors.subarray(k * nV, (k + 1) * nV)
  geometry.attributes.scalar.array.set(mode)
  geometry.attributes.scalar.needsUpdate = true
}
```

### 2. Time-varying flow with per-frame gradient arrows

Synthesize a heat-diffusion field on the mesh, scrub through time with a
slider, render face-centered gradient arrows that update on every frame.

```javascript
const ctx = nxrCompute.createContext(verticesF64, facesI32)
ctx.precompute({ k: 100 })  // need eigenmodes for the spectral generator

// Build a 200-frame heat diffusion from a delta source at vertex 0
const heat = ctx.generateHeatDiffusion(
  new Int32Array([0]),
  new Float64Array([1.0]),
  Float64Array.from({ length: 200 }, (_, i) => i * 0.005),
  /* alpha = */ 1.0,
)
// heat.data is a Float32Array [T * V] = [200 * nV]

// Per frame: extract u_t, compute gradient, update arrow instances
function onFrame(t) {
  const u_t = heat.data.subarray(t * nV, (t + 1) * nV)
  const flowVecs = ctx.scalarGradient(new Float64Array(u_t))
  // flowVecs is Float64Array [F * 3] — face-centered vectors
  updateArrowInstances(flowVecs)  // your three.js InstancedMesh update logic
}

// Drive from a Timeline UI
timelineSlider.addEventListener('input', (e) => onFrame(e.target.value | 0))
```

For 40k vertices and a per-frame gradient SpMV, this runs comfortably at
60 fps. If you need more throughput (or want full GPU-side advection),
upload `data.faceFrames` once and run the integration in TSL — see the
particle advection recipe.

### 3. Interactive geodesic measurement tool

Click a vertex, see distance everywhere; click a second, draw the geodesic
path between them.

```javascript
let firstPick = null

function onMeshClick(vertexIndex) {
  if (firstPick === null) {
    firstPick = vertexIndex
    const distances = ctx.computeGeodesicDistance(new Int32Array([vertexIndex]))
    showColormap(new Float32Array(distances))
  } else {
    const path = ctx.tracePath(firstPick, vertexIndex)
    // path is Float64Array [N * 3] — polyline points
    drawPath(new Float32Array(path))
    firstPick = null
  }
}

function showColormap(distances) {
  geometry.attributes.scalar.array.set(distances)
  geometry.attributes.scalar.needsUpdate = true
}

function drawPath(points) {
  const lineGeometry = new THREE.BufferGeometry()
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(points, 3))
  scene.add(new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color: 0xff3366 })))
}
```

The first call (heat method, ~10 ms on fsaverage6) and second call
(flip-out, ~5 ms) are both interactive-fast. nxr-compute caches the Cholesky
factor of the cotan Laplacian after the first heat solve, so subsequent
geodesic distance queries from different sources are even faster.

---

## Best practices

- **Convert to Float32 at the three.js boundary, not in the math.**
  nxr-compute's outputs are float64 (precision matters for math); three.js
  prefers float32 (bandwidth matters for rendering). Cast at the
  `BufferAttribute` boundary, never modify nxr-compute's outputs in place.

- **Don't recompute when you can re-bind.** Eigenmodes computed once
  give you K different visualizations (one per mode); animate by
  swapping which subarray you bind, not by recomputing.

- **Keep one context per mesh.** Don't create a new context for every
  Hodge solve — that throws away all the cached state. Bind operations
  to a long-lived context.

- **Use the precompute pack at app load.** It runs the most expensive
  steps (eigensolve, operator assembly) once. Subsequent interactive
  operations are O(milliseconds).

- **For per-frame compute, use the spectral generators.** Heat diffusion
  via spectral evolution is O(K) per frame regardless of mesh size — far
  cheaper than building / factoring per-frame matrices.

- **Profile before optimizing.** nxr-compute's hot paths are in C++; JS-side
  marshalling is usually a few percent of total time. If you're slow,
  it's almost always (a) recomputing something cacheable, or (b)
  three.js-side, not nxr-compute-side.

- **Release contexts on unmount.** Browsers are getting better at
  garbage-collecting WASM heaps but explicit `ctx.delete()` is faster
  and lower-memory.

---

## When to use nxr-compute vs alternatives

nxr-compute's strengths:

- **Same code in browser and native.** If you have a web app and a
  desktop app sharing the cortical analysis pipeline, nxr-compute gives you
  bit-identical results with the same calls.
- **Mesh-aware solvers, not just generic linear algebra.** nxr-compute knows
  about Hodge decomposition, geodesics, BFF — the cortical-surface
  domain. You can do these in pure linear-algebra libs, but you'd be
  rebuilding them.
- **No system dependencies.** No SuiteSparse install, no MKL, no CUDA.
  Ships everywhere geometry-central does.

nxr-compute is **not** a good fit if:

- You only need basic mesh operations (face areas, normals) — three.js
  itself has these. Don't ship a 1-2 MB WASM module for that.
- You need GPU-accelerated solvers — nxr-compute is CPU-only. For large-scale
  GPU sparse linear algebra, look at libraries built on cuSPARSE or
  WebGPU compute shaders.
- You need formats nxr-compute doesn't support — e.g., volumetric meshes, point
  clouds, signed distance functions. nxr-compute is specifically about
  triangle-meshed surfaces.

---

## Troubleshooting

**`nxrCompute.createContext` throws "input mesh is not manifold."**
Your mesh has duplicate edges, T-junctions, or non-manifold vertices.
Check that face indices are 0-based, that you have no degenerate
triangles (zero-area), and that no edge is shared by more than two
faces. Most cortical surfaces from FreeSurfer pass; auto-generated
meshes often fail.

**`computeUVCoordinates` throws "mesh has no boundary."**
BFF needs an open mesh. Cut your closed surface to disc topology first,
or pick a different parametrization method (nxr-compute doesn't currently expose
a closed-mesh parametrization).

**`solveEigenmodes` is very slow / doesn't converge.**
Check that your mesh is well-formed (no near-degenerate triangles, no
huge aspect ratios). Try a smaller k. The default sigma (-1e-8) targets
the smallest eigenvalues; if you need a different range, use a
non-default σ.

**WASM heap exhausted.**
You probably have too many contexts alive. Call `.delete()` on any
context whose mesh is no longer being viewed. Or solve fewer eigenmodes
— k=300 on a 100k-vertex mesh is ~240 MB just for the eigenvector
matrix.

**Three.js rendering shows nothing.**
Check that you're using `Uint32Array` for the index BufferAttribute (the
default 16-bit type tops out at 65535 vertices). Verify face winding —
if surfaces appear black, faces may be wound CW vs CCW; flip with
`material.side = THREE.DoubleSide` for testing.

**Memory leak across hot reloads.**
Make sure your component cleanup actually calls `ctx.delete()`. React
strict mode double-renders effects, so guard your cleanup logic.
