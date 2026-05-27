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

> **Note:** this delta is about the **functional `.m` leaves** in
> `+nxr/+manifold/`, not the dispatcher. The MEX dispatcher now
> implements the full WASM `ContextWrapper` surface in **handle mode**
> (see the section above) — `poisson`, `heat`, `hodge`, `curvatures`,
> `bff`, `isoline`, `assembleConnectionLaplacian`, etc. are all callable
> as `nxr_compute('<cmd>', h, ...)`. The leaves below remain stubbed
> only because pointing them at the handle API is an application-side
> decision (the functional package is deliberately left unchanged).

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
