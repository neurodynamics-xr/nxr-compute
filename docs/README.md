# nxr-compute — cortical-flow compute library

**nxr-compute** is a portable C++ math library for analysis and visualization on
cortical surface meshes. It provides the computational primitives behind
cortical-flow: mesh operators, eigenmodes, vector field decomposition,
geodesics, parametrization, time-varying field generators, and more.

It is **library**, not a framework. nxr-compute doesn't load files, doesn't render
anything, and doesn't know about your application — it takes vertices and
faces, runs math, and returns values. Bindings deliver this math to
specific environments (the browser via WebAssembly, MATLAB via MEX,
Electron via N-API, the command line via a standalone executable).

The architectural inspiration is `geometry-processing-js`'s `linear-algebra`
package: a small, focused C++ math core wrapped by thin per-target shells.

---

## Status — which bindings ship today

| Binding   | Artifact            | Use case                                     | Status |
|-----------|---------------------|----------------------------------------------|--------|
| CLI       | `nxr-compute` (executable)  | Offline batch precompute (mesh → Zarr store) | ✓ Ships |
| MATLAB    | `nxr_compute.mexw64`        | Brainstorm / SPM / native MATLAB pipelines   | ✓ Ships |
| Electron  | `nxr_compute_addon.node`    | Node.js / Electron desktop apps via N-API    | ✓ Ships |
| WASM      | `nxr_compute.wasm` + `nxr_compute.js` | Browser apps (three.js, plain JS, …)       | ✓ Ships |

All four binding shells run from the same `nxr-compute` C++ static library
(~13 .cpp files) — same code, same numerics, four different deliveries.
Eigenmode spectra, M-orthonormality, geodesic path lengths, and other
quantitative outputs match across all four bindings to machine
precision.

---

## What's in this folder

- **[architecture.md](architecture.md)** — the four-layer model, what
  `ManifoldSurfaceMesh` is, the data flow at the API boundary, the
  dependency policy. Read this first if you want to understand what nxr-compute is
  and how it relates to your application.

- **[three-js-integration.md](three-js-integration.md)** — installation,
  bootstrapping the WASM module, lifecycle and memory management,
  end-to-end example projects. Read this if you're building a three.js
  app and want to consume nxr-compute as a compute backend.

- **[visualization-recipes.md](visualization-recipes.md)** — a cookbook
  mapping every nxr-compute output to a concrete visualization technique
  (colormaps, arrow glyphs, streamlines, particle advection, LIC, geodesic
  paths, eigenmode reconstructions, time-varying playback, etc.). Each
  recipe is a self-contained code snippet showing both the nxr-compute API call
  and the three.js consumption.

---

## A 10-line taste

```javascript
import { initNxrCompute } from '@nxr-compute/wasm'
const nxrCompute = await initNxrCompute()

const ctx = nxrCompute.createContext(verticesFloat64, facesInt32)
const data = ctx.precompute({ k: 300 })
//   data.operators  — cotan Laplacian, mass, normals, areas
//   data.dec        — DEC operators (d0, d1, ★)
//   data.eigenmodes — first 300 Laplace-Beltrami eigenmodes, M-orthonormal
//   data.faceFrames — per-face orthonormal tangent bases (e1, e2, n)

const eigenmode_5 = data.eigenmodes.eigenvectors.subarray(5 * nV, 6 * nV)
// → drop into a three.js BufferAttribute as a vertex scalar field
```

Everything nxr-compute computes — every solver, every generator, every operator —
returns flat typed arrays or simple structs. Zero coupling to any
particular renderer or app framework.

---

## What nxr-compute does NOT do

- **No file I/O** in the math layer. nxr-compute doesn't read FreeSurfer files, doesn't
  write Zarr, doesn't parse OBJs. File format adapters live in a sibling
  library (`cxf-io`) which the CLI links but the WASM and MEX bindings
  don't — browsers and MATLAB have their own ecosystem readers.
- **No dependencies beyond Eigen + Spectra + geometry-central.** No
  SuiteSparse, no MKL, no PARDISO. Everything compiles to WASM and
  runs in MATLAB without symbol collisions.
- **No rendering, no UI, no networking.** nxr-compute is pure compute. Bring
  three.js, react, your visualization framework — nxr-compute provides the math
  underneath.

---

## License

Same as cortical-flow itself. Underlying dependencies (Eigen, Spectra,
geometry-central) are MPL2 / MIT — compatible with any consumer license.
