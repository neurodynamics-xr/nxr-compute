# nxr-compute

A portable C++ math engine for halfedge-mesh-based scientific computing,
shipped with native bindings to **MATLAB**, **Node.js**, **WebAssembly**,
and a CLI.

The library is intentionally narrow: vertices and faces go in, linear
algebra and discrete differential geometry come out. It does not load
files, render anything, or know about your application. The architectural
inspiration is `geometry-processing-js`'s `linear-algebra` package: a
small, focused C++ math core wrapped by thin per-target shells.

The numerical core is built on top of
**[geometry-central](https://geometry-central.net/)** — a header-only
halfedge mesh library by Nick Sharp and collaborators. nxr-compute
exposes geometry-central's algorithms (cotangent Laplacian, DEC
operators, heat method, Hodge decomposition, BFF, smooth direction
fields, etc.) through stable APIs across all four binding targets,
producing identical numerics regardless of host language.

---

## What it computes

- **Mesh operators** — cotangent Laplacian, lumped + Galerkin mass
  matrices, vertex dual areas, six vertex-normal estimators, DEC
  operators (`d0`, `d1`, `★0`, `★1`, `★1⁻¹`, `★2`).
- **Spectral analysis** — generalized eigenmodes
  (`K φ = λ M φ`) via Spectra's IRAM with shift-invert,
  M-orthonormalization, and DC mode removal.
- **Solvers** — Poisson on a manifold, heat-method geodesic distance,
  signed heat distance, vector heat method, Hodge–Helmholtz
  decomposition, trivial-connection direction fields.
- **Field generators** — eigenmode reconstruction, spectral heat
  diffusion, damped wave time series, smooth vertex / face direction
  fields with optional curvature alignment.
- **Geometry** — principal curvatures with directions, BFF
  parametrization (open meshes), flip-out geodesic paths, isolines,
  streamline tracing.
- **Graph-Laplacian-agnostic primary signatures** alongside mesh
  convenience overloads — the same solvers work on cotangent
  Laplacians + lumped mass (mesh case) or graph Laplacians +
  node-weight diagonals (graph signal processing case).
- **Cancellation + progress contracts** per long-running call (atomic
  flag bridge usable from JS, MATLAB, WASM, and CLI).

See **[docs/usage.md](docs/usage.md)** for the full API surface and
**[docs/extensions.md](docs/extensions.md)** for design rationale.

---

## Install

Pick the binding that matches your host language.

### MATLAB (MEX)

Pre-built MEX binaries for Windows, Linux, and macOS Apple Silicon ship
as GitHub Release assets.

**Download:** https://github.com/neurodynamics-xr/nxr-compute/releases/latest

Two zip variants per release — pick the one matching your MATLAB
version:

| Your MATLAB | Download | Includes |
|---|---|---|
| R2023a | `nxr-compute-mex-r2023a-vX.Y.Z.zip` | Windows + Linux MEX |
| R2023b or newer | `nxr-compute-mex-r2023b-vX.Y.Z.zip` | Windows + Linux + macOS Apple Silicon MEX (native `.mexmaca64`) |

**Install:**

```matlab
% After unzipping nxr-compute-mex-r2023X.zip somewhere:
addpath('/path/to/nxr-compute-mex-r2023X')

% Quick test on a tetrahedron:
V = [0 0 0; 1 0 0; 0 1 0; 0 0 1];
F = int32([1 2 3; 1 2 4; 1 3 4; 2 3 4] - 1);     % 0-based
mctx = nxr.manifold.context(V, F);
disp(mctx.nV);                                     % → 4

% Solve the smallest 4 eigenmodes:
eig = nxr.manifold.solve.eigen(mctx, 4);
disp(eig.eigenvalues);
```

MATLAB autoloads the `+nxr` package and selects the correct `.mex*`
binary for your platform based on file extension. No platform-specific
configuration needed.

**One-shot download from the terminal:**

```bash
# Replace r2023b with r2023a as appropriate
gh release download v0.1.0 -p 'nxr-compute-mex-r2023b-*.zip' \
    --repo neurodynamics-xr/nxr-compute
unzip nxr-compute-mex-r2023b-v0.1.0.zip
# Then in MATLAB: addpath('nxr-compute-mex-r2023b')
```

**macOS Intel:** not shipped pre-built. GitHub's `macos-13` runner pool
has unreliable availability in 2026 and is being retired. Intel-Mac
users on MATLAB R2023a need to build from source (see below) until
GitHub provides a viable Intel runner replacement or we add a
cross-compile path.

### WebAssembly (browser, Node, Bun)

```bash
npm install @neurodynamics-xr/nxr-compute
```

```js
import { initNxrCompute } from '@neurodynamics-xr/nxr-compute'

const nxr = await initNxrCompute()
const mctx = nxr.createManifoldContext(vertices, faces)

const eig = mctx.solve.eigen(300)
console.log(eig.eigenvalues.slice(0, 5))

mctx.dispose()    // free WASM heap
```

For off-thread dispatch in a Web Worker, see
[docs/wasm-web-worker.md](docs/wasm-web-worker.md).

### Node.js / Electron (N-API addon)

The N-API addon is built from source on `npm install` via `cmake-js`.
Requires a working C++ toolchain on the target machine.

```bash
npm install @neurodynamics-xr/nxr-compute
# Then use the native API via:
```

```js
import nxr from '@neurodynamics-xr/nxr-compute/native'
// ...same surface as the WASM binding, just faster and without the
// browser-targeted 2 GB heap cap.
```

Used by [cortical-flow](https://github.com/neurodynamics-xr/cortical-flow)
for off-main-thread spectral analysis inside Electron.

### CLI

A minimal smoke harness that round-trips a canonical icosahedron
spectrum, useful as a build verification step. Not distributed
pre-built — `bash scripts/build.sh` produces `nxr_compute.exe` /
`nxr_compute` in `build/Release/`.

---

## API namespaces (`nxr.manifold.*`)

The MATLAB, WASM, and N-API bindings all expose the manifold-DG
concepts under a six-group nested namespace. The MEX C++ dispatcher
and the WASM `ContextWrapper` stay flat internally; the namespace
tree is a thin shim per host language.

```
nxr.manifold
├── solve.{poisson, heat, eigen, hodge}
├── operator.{d0, d1, star0, star1, star2, star1Inverse,
│             mass, stiffness, laplacian, connectionLaplacian}
├── query.{vertex, isoline, center}
├── measure.{distance, signedDistance, curvature, normal, frame}
├── uv.{bff, logMap, stripe, stripeFreq}
└── interpolate.{transport, extend, directionField,
                 smoothFaceField, smoothVertexField}
```

Per-context (preferred) and functional forms are both supported in
WASM / N-API:

```js
// Per-context
mctx.solve.eigen(300)
mctx.measure.distance.signed([0, 1, 2], true)

// Functional (same compute context, identical behaviour)
nxr.manifold.solve.eigen(mctx, 300)
```

MATLAB follows the package convention — all leaves take `mctx` as
their first argument:

```matlab
eig = nxr.manifold.solve.eigen(mctx, 300);
M   = nxr.manifold.operator.mass(mctx);
```

### Mass-matrix variants

Two variants, names matching geometry-central's vocabulary exactly:

- **`"lumped"`** (default) — diagonal, sourced from geometry-central's
  [`vertexLumpedMassMatrix`](https://geometry-central.net/surface/geometry/quantities/#vertex-lumped-mass-matrix).
  Each diagonal entry is the sum of `A_T/3` over incident triangles
  (geometry-central's
  [`vertexDualAreas`](https://geometry-central.net/surface/geometry/quantities/#vertex-dual-area)).
- **`"galerkin"`** — sparse, full FEM mass, sourced from
  [`vertexGalerkinMassMatrix`](https://geometry-central.net/surface/geometry/quantities/#vertex-galerkin-mass-matrix).
  Per-triangle element matrix is `(A_T / 12) · [[2 1 1][1 2 1][1 1 2]]`,
  the L² integral of P1 hat functions. Eigenvalues converge to the
  continuous Laplace–Beltrami spectrum faster on coarse meshes.

```js
mctx.assembleManifoldOperators('lumped')      // default
mctx.assembleManifoldOperators('galerkin')
```

```matlab
% MATLAB: variant defaults to lumped; see test/test_mass_variants.cpp
% for cross-variant numerical comparison.
```

---

## Dependencies

Three header-only or vendored dependencies, nothing else:

- **[geometry-central](https://geometry-central.net/)** — halfedge
  mesh, DEC, heat method, BFF. By Nick Sharp et al.,
  [GitHub](https://github.com/nmwsharp/geometry-central). Vendored as
  a git submodule.
  - [Surface API docs](https://geometry-central.net/surface/)
  - [Geometry quantities](https://geometry-central.net/surface/geometry/quantities/)
  - [Heat method](https://geometry-central.net/surface/algorithms/geodesic_distance/)
  - [Vector heat method](https://geometry-central.net/surface/algorithms/vector_heat_method/)
  - [Boundary First Flattening](https://geometry-central.net/surface/algorithms/parameterization/#boundary-first-flattening)
- **[Eigen 3.4](https://eigen.tuxfamily.org/)** — sparse linear algebra
  (header-only, transitively via geometry-central).
- **[Spectra 1.0](https://spectralib.org/)** — sparse eigensolver
  (Eigen-only, header-only).

**Deliberately excluded:** CHOLMOD / SuiteSparse / MKL / PARDISO /
CUDA. They would give a 3–10× win on x86 native but break the WASM
and MEX targets — and `nxr_compute` must compile identically for
native, browser, and MATLAB consumers. The build saying "Building
without SuiteSparse" is the desired state, not a missed optimisation.

---

## Build from source

```bash
git clone --recursive https://github.com/neurodynamics-xr/nxr-compute.git
cd nxr-compute
npm install                 # cmake-js + node-addon-api for the addon
bash scripts/build.sh       # library + N-API addon + CLI + MEX + tests
bash scripts/build-wasm.sh  # WASM (separate emscripten toolchain)
```

**Build prerequisites:**

| Binding | Toolchain |
|---|---|
| All | CMake 3.20+, C++17 compiler |
| N-API addon | Node.js 18+, cmake-js (via `npm install`) |
| MEX | MATLAB R2023a or newer with the MEX SDK |
| WASM | emsdk (emscripten) 3.x; requires Python 3.10+ |
| CLI | None beyond the C++ toolchain |

Targets gate themselves on toolchain availability — if MATLAB isn't
installed, the MEX target is skipped without failing the rest of the
build.

**Outputs land under `build/Release/`:**

```
nxr_compute.lib / .a              # static library
nxr_compute_addon.node            # N-API addon (also copied to repo root)
nxr_compute.mexw64 / .mexa64 /    # MATLAB MEX
  .mexmaci64 / .mexmaca64
nxr_compute.exe / nxr_compute     # CLI smoke
test_*.exe                        # 13 native test executables
```

WASM lands under `build_wasm/` (separate toolchain).

## Tests

```bash
ctest --test-dir build -C Release   # all 13 native tests
node scripts/_smoke-wasm.mjs        # WASM end-to-end smoke
./build/Release/nxr_compute smoke   # CLI canonical-spectrum check
```

Native tests include numerical correctness checks for eigenmodes,
the Cholesky cache contract, mass-matrix variants, connection
Laplacians, field generators, visualization primitives, the
cancellation + progress contracts, and zero-copy passthrough
accessors.

## Layout

```
nxr-compute/
├── include/nxr/             public headers (compute.h, errors.h, …)
├── src/                     library sources
├── test/                    13 unit/smoke tests
├── bindings/{node,wasm,mex,cli}/  binding shells
├── deps/geometry-central/   git submodule
├── docs/                    usage, extensions, architecture, recipes
├── scripts/                 build + smoke entry points
└── .github/workflows/       CI: cross-platform MEX publishing
```

## Distribution channels

| Binding | Channel | URL |
|---|---|---|
| MATLAB MEX | GitHub Releases | https://github.com/neurodynamics-xr/nxr-compute/releases |
| WASM | npm | https://www.npmjs.com/package/@neurodynamics-xr/nxr-compute |
| N-API addon | npm (built on install) | same as WASM |
| CLI | source only | clone + `bash scripts/build.sh` |

CI builds MEX binaries for Windows + Linux (R2023a and R2023b tracks)
and macOS Apple Silicon (R2023b only) on every `v*` tag push, and
attaches both R2023a and R2023b zips to the corresponding GitHub
Release. See `.github/workflows/publish-mex.yml`.

## License

TBD — to be set before declaring 1.0. The library bundles
geometry-central ([MIT](https://github.com/nmwsharp/geometry-central/blob/master/LICENSE))
and pulls Spectra ([MPL-2.0](https://github.com/yixuan/spectra/blob/master/LICENSE)),
so the final choice will compose cleanly with those.

## Citing

If you use nxr-compute in academic work, please cite geometry-central
in addition to this library — the heavy lifting on the geometric
side happens there. Suggested citation for geometry-central is on
its [docs site](https://geometry-central.net/about/).
