# `+nxr/+manifold` — MATLAB six-group package

Mirrors the JS / WASM / N-API binding's `nxr.manifold.*` surface.
Each leaf is a thin wrapper around the underlying `nxr_compute('cmd', ...)`
MEX dispatcher.

## Setup

```matlab
addpath('bindings/mex/matlab')                 % this package tree
addpath('build/bindings/mex/Release')          % compiled MEX binary
```

Build the MEX with `bash scripts/build.sh Release` (requires MATLAB
R2018a+ found by CMake).

## Stateful context handle (MEX dispatcher level)

The `nxr_compute` MEX dispatcher mirrors the WASM `ContextWrapper`: a
mesh is bound **once** via `create`, and the returned `uint64` handle
keeps the geometry-central mesh, assembled operators, Cholesky factors,
and the heat / vector-heat solvers alive across calls. The integer
handle is the MATLAB analogue of WASM's `new Module.Manifold(...)`
Embind proxy.

```matlab
h = nxr_compute('create', V, F);            % build once (V: Nx3, F: Mx3, 1-based)

ops = nxr_compute('assembleManifoldOperators', h);  % cached after first call
eig = nxr_compute('solve', h, 6);                   % K,M pulled from cached ops
nxr_compute('precompute', h, 6);                    % normalized + DC-removed; caches modes

d   = nxr_compute('heat', h, 1);                    % geodesic distance (cached solver)
hd  = nxr_compute('hodge', h, omega);               % shares the CholeskyCache
cur = nxr_compute('curvatures', h);
hf  = nxr_compute('heatDiffusion', h, 1, 1.0, 0:0.1:1, 1.0);  % needs a prior solve

nxr_compute('destroy', h);                  % frees the C++ context
```

Repeated ops on one handle skip the halfedge rebuild + refactorization
that the stateless `(V, F)` convention re-pays every call. Both
conventions coexist (additive dispatch): a `uint64` scalar as the
second argument selects the handle path; `V, F` arrays select the
legacy stateless path. Handle-mode commands take **1-based** vertex /
face indices (matching the rest of the MEX marshalling).

Errors: a stale / destroyed handle raises `MException` identifier
`nxr:invalidHandle`; `heatDiffusion` / `dampedWave` before a
`solve` / `precompute` raises `nxr:notPrecomputed`.

Handle commands (handle is always the second argument):
`assembleManifoldOperators`, `assembleDECOperators`,
`assembleConnectionLaplacian`, `frames`, `normals`, `solve`,
`precompute`, `poisson`, `heat`, `tracePath`, `hodge`, `curvatures`,
`bff`, `isoline`, `trivial`, `streamline`, `whitney`, `gradient`,
`heatDiffusion`, `dampedWave`, `randomDecomposed1Form`, `parallel`,
`extendScalar`, `logMap`, `findCenter`, `signedHeat`, `smoothFace`,
`smoothVertex`, `compute`, `computeFreq`.

A MATLAB `handle`-class wrapper over this is intentionally **not**
provided here — like the JS wrappers over the WASM `Manifold`, it is an
application-side concern.

## Usage

```matlab
mctx = nxr.manifold.context(V, F);

% solve
eig = nxr.manifold.solve.eigen(mctx, 6);
oneShot = nxr.manifold.solve.precompute(mctx, 6);

% operator
K = nxr.manifold.operator.stiffness(mctx);
M = nxr.manifold.operator.mass(mctx);

% measure  (1-based indices)
d   = nxr.manifold.measure.signedDistance(mctx, [1 2 3], false);
g   = nxr.manifold.measure.distance(mctx, 1);        % geodesic from vertex 1
cur = nxr.manifold.measure.curvature(mctx);

% solve
phi = nxr.manifold.solve.poisson(mctx, [1 7], [1 -1]);
hd  = nxr.manifold.solve.hodge(mctx, omega);          % omega: nE x 1

% uv
lm = nxr.manifold.uv.logMap(mctx, 1);

% interpolate
faceField = nxr.manifold.interpolate.smoothFace(mctx, 4);
vertField = nxr.manifold.interpolate.smoothVertex(mctx, 2);
stripes   = nxr.manifold.uv.stripe(mctx, vertField.vertexFieldRaw, 3.0);

% query
center = nxr.manifold.query.center(mctx, [1 2 3]);
```

## Parity with the WASM six-group surface

Every `nxr.manifold.*` leaf the WASM binding exposes is now wired to the
MEX dispatcher via a transient context handle
(`nxr.manifold.impl.withHandle`): each leaf does `create → op → destroy`,
preserving the functional API's stateless, rebuild-per-call semantics
while reusing the dispatcher's handle commands.

Newly wired (previously `notWired`):

- `solve.poisson`, `solve.hodge`, `solve.heat` (spectral diffusion —
  runs an internal `precompute`; see its help for the `k` argument)
- `operator.d0`, `operator.d1`, `operator.star0`, `operator.star1`,
  `operator.star2`, `operator.star1Inverse`, `operator.connectionLaplacian`
- `measure.distance` (geodesic), `measure.curvature`, `measure.frame`,
  `measure.normal` (all estimators, not just the cached angle-weighted one)
- `query.isoline`
- `uv.bff`
- `interpolate.trivial` (was the misnamed `directionField.m`; the file is
  now `trivial.m`, matching the WASM `interpolate.trivial` name)

Still stubs — intentionally, because they are `stubMarker` stubs in the
WASM binding too (so this is already parity): `query.line`,
`query.circle`, `query.region`, `measure.area`, `measure.density`. They
return a `struct('method', 'todo', ...)` marker and warn once.

## API differences from JS / WASM

- **Context is a value-typed struct, not an opaque handle.** Every
  call passes `mctx` by copy. The MEX dispatcher itself is stateless;
  `mctx` carries `V, F, K, M, nV, nE, nF, totalArea, vertexDualAreas,
  vertexNormals` so every leaf has what it needs without re-assembling.
- **`measure.distance.signed` becomes a sibling.** MATLAB doesn't
  support method-attached function dispatch, so the signed-heat leaf
  is renamed to `measure.signedDistance(mctx, ...)`.
- **1-based indices (MATLAB-native).** Pass 1-based vertex / face
  indices to these wrappers, as everywhere else in MATLAB. The MEX
  marshalling converts to nxr-compute's internal 0-based convention at
  the boundary (`mxToVertexIndices` / `mxToFaceBuffer` subtract 1), so
  callers never see 0-based indices. (The JS / WASM bindings, by
  contrast, are 0-based to match three.js / Eigen / NumPy.)
- **`solve.eigen` is synchronous** here, unlike the N-API addon
  (which runs it in a libuv worker). The eigensolver releases MATLAB
  Ctrl-C via `utIsInterruptPending`.
