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

## Usage

```matlab
mctx = nxr.manifold.context(V, F);

% solve
eig = nxr.manifold.solve.eigen(mctx, 6);
oneShot = nxr.manifold.solve.precompute(mctx, 6);

% operator
K = nxr.manifold.operator.stiffness(mctx);
M = nxr.manifold.operator.mass(mctx);

% measure
d = nxr.manifold.measure.signedDistance(mctx, [0 1 2], false);

% uv
lm = nxr.manifold.uv.logMap(mctx, 0);

% interpolate
faceField = nxr.manifold.interpolate.smoothFaceField(mctx, 4);
vertField = nxr.manifold.interpolate.smoothVertexField(mctx, 2);
stripes   = nxr.manifold.uv.stripe(mctx, vertField.vertexFieldRaw, 3.0);

% query
center = nxr.manifold.query.center(mctx, [0 1 2]);
```

## Surface delta vs WASM / N-API

Several leaves are wired in the WASM and N-API bindings but not yet in
the MEX dispatcher. Calling them throws an `MException` with identifier
`nxr:notWiredInMex` and a message starting `[NOT_WIRED_IN_MEX]`. To
enable any of them:

1. Add a `cmdXxx(nlhs, plhs, nrhs, prhs)` function in
   `bindings/mex/src/nxr_compute_mex.cpp`.
2. Add an `else if (cmd == "...")` line in `mexFunction()`.
3. Replace the `nxr.manifold.impl.notWired(...)` body in the
   corresponding `.m` leaf with the actual `nxr_compute('...', ...)`
   call.

Currently NOT_WIRED_IN_MEX:

- `solve.poisson`, `solve.heat`, `solve.hodge`
- `operator.connectionLaplacian`, `operator.d0`, `operator.d1`,
  `operator.star0`, `operator.star1`, `operator.star2`,
  `operator.star1Inverse`
- `query.isoline`
- `measure.distance`, `measure.curvature`, `measure.frame`
- `uv.bff`
- `interpolate.directionField`

Stubs (round-1 placeholders, not exposed in any binding yet):
`query.line`, `query.circle`, `query.region`, `measure.area`,
`measure.density`.

## API differences from JS / WASM

- **Context is a value-typed struct, not an opaque handle.** Every
  call passes `mctx` by copy. The MEX dispatcher itself is stateless;
  `mctx` carries `V, F, K, M, nV, nE, nF, totalArea, vertexDualAreas,
  vertexNormals` so every leaf has what it needs without re-assembling.
- **`measure.distance.signed` becomes a sibling.** MATLAB doesn't
  support method-attached function dispatch, so the signed-heat leaf
  is renamed to `measure.signedDistance(mctx, ...)`.
- **0-based vs 1-based indices.** MATLAB conventions are 1-based, but
  the underlying nxr-compute uses 0-based vertex / face indices to
  match three.js / Eigen / NumPy. Pass 0-based indices through these
  wrappers; the package does not auto-convert.
- **`solve.eigen` is synchronous** here, unlike the N-API addon
  (which runs it in a libuv worker). The eigensolver releases MATLAB
  Ctrl-C via `utIsInterruptPending`.
