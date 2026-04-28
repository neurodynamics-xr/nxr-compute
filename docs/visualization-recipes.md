# Visualization recipes

A cookbook mapping every nxr-compute output to a concrete three.js visualization
technique. Each recipe is self-contained: the nxr-compute API call, the three.js
consumption code, and notes on when to use it.

For installation and lifecycle, see
[three-js-integration.md](three-js-integration.md). For the data flow
model, see [architecture.md](architecture.md).

Throughout this document, `ctx` refers to a `ComputeContext` you've already
created (`const ctx = nxrCompute.createContext(verticesF64, facesI32)`), and
`data = ctx.precompute({ k: 300 })` has been called.

---

## Quick reference table

| Visualization                 | nxr-compute call                                | Output shape           | three.js delivery                   |
|------------------------------|----------------------------------------|------------------------|-------------------------------------|
| Lit mesh                      | `data.operators.normals`                | `[V × 3]` float64      | `BufferAttribute('normal', 3)`      |
| Eigenmode colormap            | `data.eigenmodes.eigenvectors[k]`       | `[V]` float64 (column) | `BufferAttribute('scalar', 1)`      |
| Curvature colormap            | `ctx.computeCurvatures().mean`          | `[V]` float64          | `BufferAttribute('scalar', 1)`      |
| Geodesic distance             | `ctx.computeGeodesicDistance(srcs)`     | `[V]` float64          | `BufferAttribute('scalar', 1)`      |
| Geodesic path                 | `ctx.tracePath(vA, vB)`                 | `[N × 3]` float64      | `LineSegments` / `Line`             |
| Gradient arrows               | `ctx.scalarGradient(scalar)`            | `[F × 3]` float64      | `InstancedMesh` of arrows           |
| Time-varying activity         | `ctx.generateHeatDiffusion(...)`        | `[T × V]` float32      | `DataArrayTexture` + TSL            |
| Hodge: exact / co-exact / harmonic | `ctx.hodgeDecompose(ω)`            | 3 × `[F × 3]` float64  | 3 layered `InstancedMesh` arrows    |
| Trivial-connection field      | `ctx.computeDirectionField(sing)`       | `[F × 3]` per-face     | per-face short oriented lines       |
| Streamlines                   | `ctx.traceStreamlines(faceField, …)`    | `[2N × 3]` float64     | `LineSegments`                      |
| Particle advection            | `data.faceFrames` (e1, e2, n) + flow    | `[F × 6]` + `[F × 3]`  | InstancedMesh + TSL compute pass    |
| LIC textures                  | `ctx.computeUVCoordinates()` + flow     | `[V × 2]` + `[F × 3]`  | UV map + flow texture + LIC shader  |
| Isolines                      | `ctx.computeIsolines(scalar, n)`        | `[2N × 3]` float64     | `LineSegments`                      |
| Principal direction lines     | `ctx.computeCurvatures().principalDir`  | `[V × 3]` float64      | per-vertex line indicators          |
| Damped wave time-series       | `ctx.generateDampedWave(...)`           | `[T × V]` float32      | `DataArrayTexture` + TSL            |

---

## Recipe 1: Lit mesh

The starting point — render the surface itself with proper normals.

```javascript
const geometry = new THREE.BufferGeometry()
geometry.setAttribute('position', new THREE.BufferAttribute(
  new Float32Array(verticesF64), 3))
geometry.setAttribute('normal', new THREE.BufferAttribute(
  new Float32Array(data.operators.normals), 3))
geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(facesI32), 1))

const mesh = new THREE.Mesh(
  geometry,
  new THREE.MeshStandardMaterial({ color: 0xeeeeee })
)
scene.add(mesh)
```

**Notes:**
- nxr-compute's normals are angle-weighted by default. They're smoother than
  three.js's `geometry.computeVertexNormals()` for cortical-surface-like
  meshes.
- For meshes with > 65k vertices, use `Uint32Array` for the index buffer
  (otherwise three.js clamps to 16-bit and your geometry looks wrong).

---

## Recipe 2: Scalar field colormap

Display any per-vertex scalar (eigenmode, curvature, distance,
divergence, …) as a color map on the surface.

```javascript
// Pick what to display
const scalarField = data.eigenmodes.eigenvectors.subarray(0, nV)  // mode 0
// or:  const scalarField = ctx.computeCurvatures().mean
// or:  const scalarField = ctx.computeGeodesicDistance(new Int32Array([1024]))

geometry.setAttribute('scalar', new THREE.BufferAttribute(
  new Float32Array(scalarField), 1))

const material = new THREE.ShaderMaterial({
  vertexShader: /* glsl */`
    attribute float scalar;
    varying float vScalar;
    varying vec3 vNormal;
    void main() {
      vScalar = scalar;
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.);
    }
  `,
  fragmentShader: /* glsl */`
    varying float vScalar;
    varying vec3 vNormal;
    uniform sampler2D colormap;
    uniform vec2  range;            // [min, max] in scalar space
    void main() {
      float t = clamp((vScalar - range.x) / (range.y - range.x), 0., 1.);
      vec3 base = texture(colormap, vec2(t, 0.5)).rgb;
      // Cheap Lambert from interpolated normal
      float lambert = max(0., dot(vNormal, vec3(0., 0., 1.))) * 0.7 + 0.3;
      gl_FragColor = vec4(base * lambert, 1.);
    }
  `,
  uniforms: {
    colormap: { value: viridisLUT },              // 1D LUT texture
    range:    { value: new THREE.Vector2(-1, 1) },
  },
})
```

**To swap fields without rebuilding geometry**, replace just the scalar:

```javascript
function setScalarField(arr) {
  geometry.attributes.scalar.array.set(arr)
  geometry.attributes.scalar.needsUpdate = true
}
```

**Tips:**
- Compute a sensible color range from the data: `Math.min(...scalarField)` /
  `Math.max(...scalarField)`. For symmetric quantities (eigenmodes), use
  `[-max(|val|), +max(|val|)]` and a divergent colormap (`coolwarm`,
  `RdBu`).
- For unsigned quantities (geodesic distance), use a sequential colormap
  (`viridis`, `magma`).

---

## Recipe 3: Eigenmode visualization

A specialization of the scalar-field colormap, keyed by mode index.
Common UI: a slider that scrubs through eigenmodes 0..K-1.

```javascript
const nV = data.operators.nV
const K  = data.eigenmodes.k
const eigenvectors = data.eigenmodes.eigenvectors  // [V × K] column-major

function showMode(k) {
  // Column k starts at offset k*nV (column-major storage)
  const mode = eigenvectors.subarray(k * nV, (k + 1) * nV)
  setScalarField(new Float32Array(mode))

  // Auto-fit color range
  let amax = 0
  for (const v of mode) amax = Math.max(amax, Math.abs(v))
  material.uniforms.range.value.set(-amax, amax)
}

modeSlider.addEventListener('input', (e) => showMode(e.target.value | 0))
```

**Why this is fast:** the eigenvectors array is already in JS memory after
`precompute`. Switching modes is just changing the subarray bound — no
recomputation, no IPC, no compute. The total cost is one buffer copy
(O(V) = ~40k floats on fsaverage6).

---

## Recipe 4: Curvature visualization

```javascript
const curv = ctx.computeCurvatures()
// curv.gaussian      — Float64Array [V] — K (Gaussian)
// curv.mean          — Float64Array [V] — H (mean)
// curv.kMin          — Float64Array [V] — κ_min
// curv.kMax          — Float64Array [V] — κ_max
// curv.principalDir  — Float64Array [V × 3] — direction of κ_max

setScalarField(new Float32Array(curv.mean))
material.uniforms.range.value.set(-0.5, 0.5)  // tune to your mesh
```

**Principal direction line indicators** (anisotropic features, ridges):

```javascript
function buildDirectionLineSegments(scalePerV = 0.005) {
  const segPositions = new Float32Array(nV * 2 * 3)
  for (let v = 0; v < nV; v++) {
    const px = verticesF64[v*3], py = verticesF64[v*3+1], pz = verticesF64[v*3+2]
    const dx = curv.principalDir[v*3]   * scalePerV
    const dy = curv.principalDir[v*3+1] * scalePerV
    const dz = curv.principalDir[v*3+2] * scalePerV
    segPositions.set([
      px - dx, py - dy, pz - dz,
      px + dx, py + dy, pz + dz,
    ], v * 6)
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(segPositions, 3))
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x00ddee }))
}
```

---

## Recipe 5: Geodesic distance and path

Click-to-measure interactions. Two pieces:

**Distance from sources** (heat method) — fast, returns a scalar field:

```javascript
function showDistanceFrom(vertexIdx) {
  const distances = ctx.computeGeodesicDistance(new Int32Array([vertexIdx]))
  setScalarField(new Float32Array(distances))
  // Use a sequential colormap; range should be [0, max]
  let max = 0
  for (const d of distances) if (d > max) max = d
  material.uniforms.range.value.set(0, max)
}
```

**Path between two vertices** (flip-out) — returns a polyline:

```javascript
function drawGeodesicPath(vA, vB) {
  const path = ctx.tracePath(vA, vB)
  // path is Float64Array [N × 3]
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(path), 3))
  return new THREE.Line(g, new THREE.LineBasicMaterial({
    color: 0xff3366,
    linewidth: 2,  // Note: ignored in WebGL; use Line2 for thick lines
  }))
}
```

**Two-click flow:**

```javascript
let firstPick = null
function onClick(vertexIdx) {
  if (firstPick === null) {
    firstPick = vertexIdx
    showDistanceFrom(vertexIdx)
  } else {
    scene.add(drawGeodesicPath(firstPick, vertexIdx))
    firstPick = null
  }
}
```

**Performance:** both calls cache. After the first
`computeGeodesicDistance`, the cotan Laplacian's Cholesky factor is in
`CholeskyCache`; subsequent geodesic distance queries from different
sources are sub-10ms even on fsaverage6.

---

## Recipe 6: Gradient flow arrows

Compute the gradient of a scalar field on faces and render as arrow
glyphs at face centroids.

```javascript
import { ConeGeometry, InstancedMesh, Object3D, Vector3, Quaternion } from 'three'

// Pre-compute face centroids (nxr-compute doesn't return these by default;
// derive from V and F)
function computeFaceCentroids() {
  const c = new Float32Array(nF * 3)
  for (let f = 0; f < nF; f++) {
    const a = facesI32[f*3], b = facesI32[f*3+1], cIdx = facesI32[f*3+2]
    c[f*3+0] = (verticesF64[a*3]   + verticesF64[b*3]   + verticesF64[cIdx*3])   / 3
    c[f*3+1] = (verticesF64[a*3+1] + verticesF64[b*3+1] + verticesF64[cIdx*3+1]) / 3
    c[f*3+2] = (verticesF64[a*3+2] + verticesF64[b*3+2] + verticesF64[cIdx*3+2]) / 3
  }
  return c
}
const centroids = computeFaceCentroids()

// Build the instanced mesh once
const arrowGeom = new ConeGeometry(0.02, 0.1, 6).rotateX(Math.PI / 2)
const arrowMat  = new THREE.MeshBasicMaterial({ color: 0xff8800 })
const arrows    = new InstancedMesh(arrowGeom, arrowMat, nF)

const dummy   = new Object3D()
const fwd     = new Vector3(0, 0, 1)
const tmp     = new Vector3()
const tmpQ    = new Quaternion()
function updateArrows(faceVectorsF64) {
  for (let f = 0; f < nF; f++) {
    dummy.position.set(centroids[f*3], centroids[f*3+1], centroids[f*3+2])
    tmp.set(faceVectorsF64[f*3], faceVectorsF64[f*3+1], faceVectorsF64[f*3+2])
    const len = tmp.length()
    if (len < 1e-9) {
      dummy.scale.set(0, 0, 0)  // hide degenerate
    } else {
      tmp.divideScalar(len)
      tmpQ.setFromUnitVectors(fwd, tmp)
      dummy.quaternion.copy(tmpQ)
      dummy.scale.set(1, 1, len * 5)  // length scaling factor — tune to your scale
    }
    dummy.updateMatrix()
    arrows.setMatrixAt(f, dummy.matrix)
  }
  arrows.instanceMatrix.needsUpdate = true
}

// Compute and render
const flowVecs = ctx.scalarGradient(new Float64Array(myScalarField))
updateArrows(flowVecs)
scene.add(arrows)
```

**Performance:** `scalarGradient` is one SpMV per call — ~5 ms on
fsaverage6. The bottleneck is the JS loop updating `nF` instance
matrices; for very large meshes, do this on the GPU instead (see the
particle advection recipe).

---

## Recipe 7: Time-varying activity playback

Generate a `[T × V]` time series and scrub through it with a Timeline UI.
Two paths: **CPU per-frame** (simple, works for slow update rates) and
**GPU sampling** (sub-frame scrubbing, requires uploading once to a
3D texture).

### CPU per-frame (simple)

```javascript
const heat = ctx.generateHeatDiffusion(
  new Int32Array([0]),
  new Float64Array([1.0]),
  Float64Array.from({ length: 200 }, (_, i) => i * 0.005),
  /* alpha = */ 1.0,
)
// heat.data: Float32Array [T * V] = [200 * nV]

function showFrame(t) {
  const frame = heat.data.subarray(t * nV, (t + 1) * nV)
  setScalarField(frame)
}
timelineSlider.addEventListener('input', e => showFrame(+e.target.value))
```

### GPU 3D-texture (smooth scrubbing)

Upload the entire time series as a `DataArrayTexture` (one slice per
frame) and let the shader sample at `(uv, t)`:

```javascript
import { DataArrayTexture, RedFormat, FloatType } from 'three'

const tex = new DataArrayTexture(heat.data, nV, 1, heat.T)
tex.format       = RedFormat
tex.type         = FloatType
tex.needsUpdate  = true

// In your shader (using TSL is even cleaner; this is GLSL):
//   uniform sampler2DArray activityTex;
//   uniform float currentFrame;
//   uniform int   nV;
//   ...
//   in fragment shader, with a per-vertex `vertexId` attribute:
//     float u = (vertexId + 0.5) / float(nV);
//     float scalar = texture(activityTex, vec3(u, 0.5, currentFrame)).r;
//
// Setting currentFrame as a fractional value gives you sub-frame
// interpolation between time samples for free.
```

For TSL (three.js's WebGPU-native shading language), do the lookup with
`textureSample(uniformTex, vec3(uv, time))` and bind `time` as a uniform
that the slider updates.

### Damped wave instead of heat

Same shape, different generator:

```javascript
const wave = ctx.generateDampedWave(
  new Int32Array([0, 1, 2]),       // mode indices
  new Float64Array([1.0, 0.5, 0.3]),  // amplitudes
  new Float64Array([0.1, 0.1, 0.1]),  // damping rates
  new Float64Array([0.0, 0.0, 0.0]),  // phases
  Float64Array.from({ length: 200 }, (_, i) => i * 0.05),
)
// wave.data: Float32Array [T * V]
```

---

## Recipe 8: Hodge decomposition viz

Take a 1-form ω on edges (random, or a measured flow), decompose into
gradient + curl + harmonic. Render each component as its own arrow layer.

```javascript
// Generate an ω with known structure for testing
const omega = ctx.generateRandomDecomposed1Form({
  alphaStrength: 1.0,   // gradient component
  betaStrength:  0.7,   // curl component
  gammaStrength: 0.0,   // harmonic component
  seed: 42,
})

const result = ctx.hodgeDecompose(omega)
// Per-vertex scalar potentials
//   result.exactPotential       — α (gradient potential)
//   result.coExactPotentialV    — β (curl potential, averaged to vertices)
//
// Per-edge 1-forms
//   result.dAlpha, result.deltaBeta, result.gamma  — Float64Array [E]
//
// Per-face 3D vector fields (Whitney-interpolated from above)
//   result.omegaVectors, result.dAlphaVectors,
//   result.deltaBetaVectors, result.gammaVectors  — Float64Array [F × 3]

// Three layers of arrows, toggleable from UI
const arrowsExact    = makeArrowLayer(result.dAlphaVectors,    0xff3366)
const arrowsCoExact  = makeArrowLayer(result.deltaBetaVectors, 0x33ff66)
const arrowsHarmonic = makeArrowLayer(result.gammaVectors,     0x6633ff)
scene.add(arrowsExact, arrowsCoExact, arrowsHarmonic)

// Or as scalar fields on the mesh
function showGradientPotential() {
  setScalarField(new Float32Array(result.exactPotential))
}
function showCurlPotential() {
  setScalarField(new Float32Array(result.coExactPotentialV))
}
```

**When this is interesting:** a flow field with non-trivial Hodge
structure has visible vortices (in the harmonic component on
higher-genus surfaces) or sources / sinks (in the divergence of the
exact part). The decomposition isolates each.

---

## Recipe 9: Trivial-connection direction field

A globally consistent tangent vector field on the surface, with
prescribed singularities. The standard tool for cortical "fiber
direction" visualization.

```javascript
// Place singularities (Gauss-Bonnet requires Σ σ = χ for a smooth field)
// On a sphere χ = 2; two +1 singularities at antipodes is the canonical example
const singularityVerts  = new Int32Array([0, 100])
const singularityValues = new Float64Array([1.0, 1.0])

const dir = ctx.computeDirectionField(singularityVerts, singularityValues)
// dir.connections        — Float64Array [E] — per-edge angles
// dir.directionVectors   — Float64Array [F × 3] — per-face directions
// dir.orthogonalVectors  — Float64Array [F × 3] — per-face orthogonals
// dir.eulerCharacteristic — number
// dir.gaussBonnetSatisfied — boolean

// Render as short oriented lines on each face (no arrowhead — direction,
// not vector — typically with both ends symmetric)
function makeDirectionLineSegments(faceVectors, halfLen = 0.01) {
  const positions = new Float32Array(nF * 2 * 3)
  for (let f = 0; f < nF; f++) {
    const cx = centroids[f*3], cy = centroids[f*3+1], cz = centroids[f*3+2]
    const dx = faceVectors[f*3]   * halfLen
    const dy = faceVectors[f*3+1] * halfLen
    const dz = faceVectors[f*3+2] * halfLen
    positions.set([cx - dx, cy - dy, cz - dz, cx + dx, cy + dy, cz + dz], f * 6)
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xffaa00 }))
}

scene.add(makeDirectionLineSegments(dir.directionVectors))
```

---

## Recipe 10: Streamlines

Trace particles through a face vector field; render the trails as line
segments.

```javascript
const stream = ctx.traceStreamlines(
  faceVectorField,    // Float64Array [F × 3]
  /* numSeeds = */    50,
  /* stepCoef = */    0.15,   // step length, in units of mean edge length
  /* maxSteps = */    1000,
)
// stream.positions      — Float64Array [2N × 3] — endpoint pairs
// stream.segmentCount   — N

const g = new THREE.BufferGeometry()
g.setAttribute('position', new THREE.BufferAttribute(
  new Float32Array(stream.positions), 3))
const lines = new THREE.LineSegments(
  g,
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 })
)
scene.add(lines)
```

**Tip:** for animated streamlines (the field "flowing"), don't recompute
the streamlines per frame — use a fixed set and animate intensity along
each line via a custom shader that uses arc-length + time as input.

---

## Recipe 11: GPU particle advection on the surface

Render thousands of particles flowing along a face vector field, fully
on the GPU. Uses face frames so the integration happens in 2D
face-local coordinates and gets lifted to 3D in the shader.

```javascript
// Once: upload face frames (constant per mesh)
const frameBuffer = new Float32Array(nF * 6)
for (let f = 0; f < nF; f++) {
  frameBuffer[f*6+0] = data.faceFrames.e1[f*3+0]
  frameBuffer[f*6+1] = data.faceFrames.e1[f*3+1]
  frameBuffer[f*6+2] = data.faceFrames.e1[f*3+2]
  frameBuffer[f*6+3] = data.faceFrames.e2[f*3+0]
  frameBuffer[f*6+4] = data.faceFrames.e2[f*3+1]
  frameBuffer[f*6+5] = data.faceFrames.e2[f*3+2]
}
// Bind frameBuffer as a DataTexture or storage buffer accessible to your
// compute pass (TSL: storageBuffer, GLSL: buffer texture).
```

**TSL compute pass sketch** (each particle stored as `{ faceIdx, u, v }`
in a storage buffer; `flowFaceField` updated per frame):

```javascript
import { texture, storageObject, vec2, vec3, … } from 'three/tsl'

const advect = Fn(({ particles, flowField, faceFrames, dt }) => {
  const i = instanceIndex
  const p = particles.element(i)            // { faceIdx, u, v, …}
  const flow2D = projectToFaceFrame(flowField.element(p.faceIdx),
                                     faceFrames.element(p.faceIdx))
  // Update local (u, v)
  p.u.assign(p.u.add(flow2D.x.mul(dt)))
  p.v.assign(p.v.add(flow2D.y.mul(dt)))
  // Cross-face logic when (u,v) leaves the face's barycentric region…
  // (omitted for brevity — see cortical-flow's ParticleFlow3D for a worked impl)
})

// Per frame:
particles.material.uniformsNode.flowField.value = currentFlowFieldBuffer
particles.material.uniformsNode.dt.value = clock.getDelta()
```

The full particle-on-surface advection is a substantial piece of
shader code (face-crossing, barycentric arithmetic, etc.). The point
of nxr-compute's role here is that **the face frames you need are precomputed
and constant** — you upload them once at app start and never touch them
again. The per-frame cost is just the flow field update.

If you want a simpler particle viz without the GPU advection logic, fall
back to the streamlines recipe — it does CPU-side integration in nxr-compute
and gives you the polylines.

---

## Recipe 12: Line Integral Convolution (LIC) on the surface

Render flow textures using BFF parametrization and a noise-LIC shader.
Best for visualizing dense flow detail at multiple scales.

```javascript
// Pre-requisite: cut the mesh to disc topology (BFF requires open mesh)
// Use cortical-flow's atlas-cut convention or a manual cut

const uvs = ctx.computeUVCoordinates()  // Float64Array [V × 2]
geometry.setAttribute('uv', new THREE.BufferAttribute(
  new Float32Array(uvs), 2))

// Bind a 2D noise texture
const noiseTex = new THREE.DataTexture(generateNoiseUint8(512, 512), 512, 512)
noiseTex.needsUpdate = true

// Encode the face vector field as a 2D texture (project each face vector
// to its (e1, e2) frame, then rasterize into UV space)
const flowTex = projectFlowToUVTexture(faceVectorField, data.faceFrames, uvs, 512)

const licMaterial = new THREE.ShaderMaterial({
  vertexShader: /* as above, plus passing UV */,
  fragmentShader: /* glsl */`
    uniform sampler2D noise;
    uniform sampler2D flow;
    varying vec2 vUv;
    void main() {
      // Standard LIC: integrate noise along streamlines of the flow
      // field, both forward and backward, blend the convolution result.
      vec2 p = vUv;
      float sum = 0.;
      for (int i = -8; i <= 8; i++) {
        vec2 v = texture(flow, p).xy;
        p += v * 0.002 * float(i > 0 ? 1 : -1);
        sum += texture(noise, p).r;
      }
      float lic = sum / 17.;
      gl_FragColor = vec4(vec3(lic), 1.);
    }
  `,
  uniforms: {
    noise: { value: noiseTex },
    flow:  { value: flowTex },
  },
})
```

**When LIC is right:** dense flow fields, high-frequency structure
(turbulent or highly-detailed flows). For sparse fields, streamlines or
arrows are more readable.

**Why BFF specifically:** the conformal property keeps angles roughly
preserved, so streamlines in UV space visually match streamlines on the
surface. With non-conformal parametrizations (uniform UV mapping),
streamlines look locally distorted.

---

## Recipe 13: Isolines / contours

Extract scalar-field contour lines as polyline segments.

```javascript
const iso = ctx.computeIsolines(
  scalarFieldF64,
  /* numLevels = */ 20,    // evenly-spaced contour values
  /* minValue =  */ 0,     // 0 = auto-detect from data
  /* maxValue =  */ 0,
)
// iso.positions     — Float64Array [2N × 3] — segment endpoints (pairs)
// iso.segmentCount  — N

const g = new THREE.BufferGeometry()
g.setAttribute('position', new THREE.BufferAttribute(
  new Float32Array(iso.positions), 3))
scene.add(new THREE.LineSegments(
  g,
  new THREE.LineBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.4 })
))
```

**Useful for:** layering crisp contour lines over a colormap to give
quantitative reference. Also for visualizing geodesic distance "ripples"
emanating from a source.

---

## Recipe 14: Poisson source-sink visualization

Solve `L φ = -M (ρ - ρ̄)` with sources at user-clicked vertices, render
the resulting potential field.

```javascript
function showPoissonField(sourceVertices, sourceValues) {
  const phi = ctx.solvePoisson(
    new Int32Array(sourceVertices),
    new Float64Array(sourceValues),
  )
  setScalarField(new Float32Array(phi))
  let amax = 0
  for (const v of phi) amax = Math.max(amax, Math.abs(v))
  material.uniforms.range.value.set(-amax, amax)
}

// User clicks: alternate "+1" and "-1" sources
let nextSign = 1
const sourceMap = new Map()  // vertexIdx → +1 or -1
function onClick(vertexIdx) {
  if (sourceMap.has(vertexIdx)) sourceMap.delete(vertexIdx)
  else { sourceMap.set(vertexIdx, nextSign); nextSign *= -1 }
  showPoissonField(
    [...sourceMap.keys()],
    [...sourceMap.values()],
  )
}
```

After the first call, the Cholesky factor of `L + ε I` is cached on the
context. Adding/removing sources is sub-10ms even on fsaverage6 — a
genuinely interactive tool.

---

## Performance summary

Approximate timings on fsaverage6 (~40k vertices, ~80k faces) in WASM
on a modern desktop:

| Operation                                  | Cost                              |
|--------------------------------------------|-----------------------------------|
| `createContext` + `precompute({ k: 300 })` | ~5 s (eigensolve dominates)       |
| `precompute({ k: 100 })`                   | ~1.5 s                            |
| `scalarGradient` per call                  | ~3 ms                             |
| `computeGeodesicDistance` first call       | ~50 ms (factors L)                |
| `computeGeodesicDistance` subsequent       | ~5 ms (uses cache)                |
| `tracePath`                                | ~10 ms                            |
| `solvePoisson` per call (after first)      | ~5 ms                             |
| `hodgeDecompose` first call                | ~100 ms (factors A and B)         |
| `hodgeDecompose` subsequent                | ~10 ms                            |
| `generateHeatDiffusion` (200 frames)       | ~30 ms                            |
| `computeIsolines`                          | ~10 ms                            |
| `traceStreamlines` (50 seeds, 1000 steps)  | ~50 ms                            |
| `computeUVCoordinates` (BFF)               | ~200 ms                           |
| `computeFaceFrames`                        | ~3 ms                             |

The CholeskyCache is the reason most operations stay sub-10ms after the
first heavy call. If you find an operation slower than this list,
suspect:

1. You created a new `ComputeContext` each time (cache lost).
2. You're solving with very high k (eigensolve cost grows superlinearly).
3. Your mesh has degenerate triangles (Cholesky factors slow down
   dramatically near singular matrices).
4. You're forcing JS-side recomputation when a `subarray` view would do.

---

## Further reading

- [`architecture.md`](architecture.md) — the full architectural model
- [`three-js-integration.md`](three-js-integration.md) — installation,
  lifecycle, and end-to-end examples
- `geometry-processing-js` repo — the inspiration for nxr-compute's structure
- The cortical-flow Electron app's `useTimeVaryingFlow` hook — a worked
  example of nxr-compute-driven per-frame visualization
