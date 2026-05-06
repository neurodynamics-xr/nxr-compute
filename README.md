# nxr-compute

Portable C++ math/compute engine for halfedge-mesh-based scientific computing.

The library is intentionally narrow: it takes vertices and faces in, runs
linear algebra and discrete differential geometry, and returns matrices
out. It does not load files, render anything, or know about your
application. Bindings expose the math to Node.js / Electron, browsers
(WebAssembly), MATLAB (MEX), and a CLI smoke test.

The architectural inspiration is `geometry-processing-js`'s
`linear-algebra` package: a small, focused C++ math core wrapped by
thin per-target shells.

---

## Capabilities

- Mesh operators: cotangent Laplacian, lumped Voronoi mass, vertex /
  face areas, normals (six estimators), DEC operators (`d0`, `d1`,
  `★0`, `★1`, `★1⁻¹`).
- Spectral: generalised eigenmodes via Spectra IRAM with shift-invert,
  M-orthonormalisation, DC removal.
- Solvers: Poisson on a manifold, heat-method geodesic distance,
  Hodge–Helmholtz decomposition, trivial-connection direction fields.
- Field generators: random vertex / face / 1-form, eigenmode fields,
  spectral heat diffusion and damped wave time series.
- Geometry: principal curvatures with directions, BFF parametrization
  (open meshes), flip-out geodesic paths, isolines, streamline tracing.
- Cancellation + progress contracts per call (atomic flag bridge).
- Graph-Laplacian-agnostic primary signatures alongside mesh
  convenience overloads — same API for graph signal processing.

See [docs/usage.md](docs/usage.md) for the full API and binding-by-binding
patterns; [docs/extensions.md](docs/extensions.md) for design rationale.

## Dependency policy

Three header-only or vendored dependencies, nothing else:

- **Eigen 3.4** — sparse linear algebra
- **Spectra 1.0** — sparse eigensolver (Eigen-only)
- **geometry-central** — halfedge mesh, DEC, heat method, BFF

CHOLMOD / SuiteSparse / MKL / PARDISO / CUDA are deliberately excluded —
they would break the WASM and MEX targets. The library compiles
identically for native, browser, and MATLAB consumers.

## Bindings

| Binding | Path | Output | Status |
|---|---|---|---|
| Node.js (N-API) | `bindings/node/` | `nxr_compute_addon.node` | ✓ |
| WebAssembly (Embind) | `bindings/wasm/` | `nxr_compute.js` + `.wasm` | ✓ |
| MATLAB MEX | `bindings/mex/` | `nxr_compute.mexw64` | ✓ |
| CLI smoke | `bindings/cli/` | `nxr_compute.exe` | ✓ |

## API namespaces (WASM/JS)

The package is `@neurodynamics-xr/nxr-compute` — the engine. The JS API
exposes the manifold-DG concepts under a six-group nested namespace
`nxr.manifold.*`. The C++ embind class (`ContextWrapper`) and the WASM
artifact (`nxr_compute.wasm`) stay flat; the namespace tree is a thin JS
shim on top.

```
nxr.manifold
├── solve.{poisson, heat, eigen, hodge}
├── operator.{d0, d1, star0, star1, star2, star1Inverse,
│             mass, stiffness, laplacian}
├── query.{vertex, line*, circle*, region*, isoline, center}
├── measure.{distance, distance.signed, area*, density*,
│            curvature, normal, frame}
├── uv.{bff, logMap, stripe, stripeFreq}
└── interpolate.{transport, extend, directionField,
                 smoothFaceField, smoothVertexField}
```

`*` marks round-1 placeholders — see "Stubbed methods" below.

Two equivalent forms are supported:

```js
import { initNxrCompute } from '@neurodynamics-xr/nxr-compute'
const nxr  = await initNxrCompute()
const mctx = nxr.createManifoldContext(verts, faces)

// Per-context (preferred):
mctx.solve.eigen(300)
mctx.operator.d0()
mctx.measure.distance([0])
mctx.measure.distance.signed([0, 1, 2], true)
mctx.uv.bff()
mctx.interpolate.transport([0], [1, 0, 0])

// Functional (same compute context, identical behaviour):
nxr.nxr.manifold.solve.eigen(mctx, 300)
nxr.nxr.manifold.operator.d0(mctx)

// Free WASM heap when done:
mctx.dispose()
```

One-liner per group:

| Group | Example |
|---|---|
| `solve` | `mctx.solve.poisson([0], [1.0])` — solve Δu = f with Dirichlet sources |
| `operator` | `mctx.operator.laplacian()` — cotangent Laplacian as `SparseMatrixCOO` |
| `query` | `mctx.query.center([0, 1, 2])` — Karcher mean of three vertices |
| `measure` | `mctx.measure.curvature()` — Gaussian + mean + principal curvatures |
| `uv` | `mctx.uv.logMap(0)` — logarithmic map at vertex 0 |
| `interpolate` | `mctx.interpolate.smoothFaceField(4)` — smooth cross field |

**Caching.** `operator.d0()` … `operator.star1Inverse()` share one
internal `assembleDECOperators` call; `operator.mass()`, `.stiffness()`,
and `.laplacian()` share one `assembleMeshOperators` call. Use
`mctx.operator.invalidateCache()` to force re-assembly.

**Flat surface (legacy).** `nxr.createContext(verts, faces)` is
unchanged — same method names as before. `mctx._flat` exposes the same
flat surface on the same underlying ContextWrapper, useful for mixed
namespaced + flat code during migration.

### Stubbed methods (round 1)

These return a `{ method: 'todo', name }` marker and emit a one-shot
`console.warn`. Full implementations land in follow-up rounds:

- `query.line(v1, v2)` — flip-out geodesic between two vertices
- `query.circle(v, r)` — geodesic-distance isoline at radius r
- `query.region(v, r)` — vertex set within geodesic radius r
- `measure.area(region)` — integrate vertex/face areas over a selection
- `measure.density(region, field)` — average a scalar field over a region

`solve.hodge` is wired (delegates to the existing `hodgeDecompose`) but
its grouping is **provisional** — it may move to `operator.hodgeDecomp`
or `interpolate.hodge` in a later refactor.

## Build

```bash
git clone --recursive git@github.com:neurodynamics-xr/nxr-compute.git
cd nxr-compute
npm install                 # cmake-js + node-addon-api for the addon
bash scripts/build.sh       # library + addon + cli + mex + tests
bash scripts/build-wasm.sh  # WASM (separate toolchain — needs emsdk)
```

Outputs land under `build/` (native) and `build_wasm/` (Emscripten).
The N-API addon is also copied to the repo root as
`nxr_compute_addon.node` for the Node smoke scripts to require directly.

## Tests

Seven native test executables and one WASM smoke:

```bash
ctest --test-dir build -C Release   # all 7 native tests
node scripts/_smoke-wasm.mjs        # WASM end-to-end
./build/Release/nxr_compute smoke   # CLI canonical-spectrum check
```

## Layout

```
nxr-compute/
├── include/nxr/             public headers (compute.h, errors.h, …)
├── src/                     library sources
├── test/                    7 unit/smoke tests
├── bindings/{node,wasm,mex,cli}/  binding shells
├── deps/geometry-central/   submodule
├── docs/                    usage, extensions, architecture, recipes
└── scripts/                 build + smoke entry points
```

## License

TBD. The library bundles geometry-central (MIT) and pulls Spectra
(MPL-2.0).
