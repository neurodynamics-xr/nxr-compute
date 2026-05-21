% Smoke test for the MATLAB nxr.manifold.* package.
%
% Run from the repo root:
%   /Applications/MATLAB_R2023b.app/bin/matlab -batch \
%     "run('scripts/_smoke-matlab-manifold.m')"
%
% Or interactively:
%   >> addpath('bindings/mex/matlab', 'build/bindings/mex/Release')
%   >> run scripts/_smoke-matlab-manifold.m
%
% Mirrors scripts/_smoke-addon-manifold.mjs and _smoke-wasm.mjs —
% same icosahedron fixture, same six-group surface checks, plus
% verification that NOT_WIRED_IN_MEX leaves throw with the stable
% identifier 'nxr:notWiredInMex'.

repoRoot  = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(repoRoot);
addpath(fullfile(repoRoot, 'bindings', 'mex', 'matlab'));
addpath(fullfile(repoRoot, 'build', 'bindings', 'mex', 'Release'));

% ── Icosahedron fixture ──────────────────────────────────────
t = (1 + sqrt(5)) / 2;
raw = [
  -1  t  0;   1  t  0;  -1 -t  0;   1 -t  0;
   0 -1  t;   0  1  t;   0 -1 -t;   0  1 -t;
   t  0 -1;   t  0  1;  -t  0 -1;  -t  0  1
];
V = raw ./ vecnorm(raw, 2, 2);
F = int32([
  0 11  5;  0  5  1;  0  1  7;  0  7 10;  0 10 11;
  1  5  9;  5 11  4; 11 10  2; 10  7  6;  7  1  8;
  3  9  4;  3  4  2;  3  2  6;  3  6  8;  3  8  9;
  4  9  5;  2  4 11;  6  2 10;  8  6  7;  9  8  1
]);

failures = 0;
function check(cond, msg)
    if cond
        fprintf('  ✓ %s\n', msg);
    else
        fprintf(2, '  ✗ %s\n', msg);
        evalin('base', 'failures = failures + 1;');
    end
end

mctx = nxr.manifold.context(V, F);

% ── Sizes ────────────────────────────────────────────────────
check(mctx.nV == 12, 'mctx.nV == 12');
check(mctx.nF == 20, 'mctx.nF == 20');
check(mctx.nE == 30, 'mctx.nE == 30');

% ── operator group ───────────────────────────────────────────
K = nxr.manifold.operator.stiffness(mctx);
check(size(K,1) == 12 && size(K,2) == 12, 'operator.stiffness == 12×12');

M = nxr.manifold.operator.mass(mctx);
check(size(M,1) == 12 && size(M,2) == 12, 'operator.mass == 12×12');

L = nxr.manifold.operator.laplacian(mctx);
check(isequal(L, K), 'operator.laplacian aliases stiffness');

% ── solve group ──────────────────────────────────────────────
eig = nxr.manifold.solve.eigen(mctx, 6);
check(numel(eig.eigenvalues) == 6, 'solve.eigen(6) returns 6 modes');
check(all(isfinite(eig.eigenvalues)), 'solve.eigen all finite');
check(all(eig.eigenvalues > -1e-9),  'solve.eigen all non-negative');

oneShot = nxr.manifold.solve.precompute(mctx, 6);
check(numel(oneShot.eigenvalues) == 5, ...
      'solve.precompute(6) returns 5 modes (DC dropped)');

% ── measure group ───────────────────────────────────────────
d = nxr.manifold.measure.signedDistance(mctx, int32([0 1 2]), false);
check(numel(d) == 12 && all(isfinite(d)), 'measure.signedDistance length 12');

% ── interpolate group ──────────────────────────────────────
sff = nxr.manifold.interpolate.smoothFace(mctx, 4);
check(numel(sff) == 60, 'interpolate.smoothFace → 20×3 vectors');

svf = nxr.manifold.interpolate.smoothVertex(mctx, 2);
check(numel(svf.vertexFieldRaw) == 24, 'interpolate.smoothVertex → 12×2 raw');

% ── uv group ───────────────────────────────────────────────
stripes = nxr.manifold.uv.stripe(mctx, svf.vertexFieldRaw, 3.0);
check(stripes.segmentCount > 0, sprintf('uv.stripe(...) → %d segments', stripes.segmentCount));

% ── query group ────────────────────────────────────────────
v = nxr.manifold.query.vertex(mctx, 7);
check(v.vertexIndex == 7, 'query.vertex(7).vertexIndex == 7');

c = nxr.manifold.query.center(mctx, int32([0 1 2]));
check(numel(c) == 3 && all(isfinite(c)), 'query.center → 3D point');

% ── NOT_WIRED_IN_MEX leaves ───────────────────────────────
function expectNotWired(label, fn)
    try
        fn();
        fprintf(2, '  ✗ %s — expected throw\n', label);
        evalin('base', 'failures = failures + 1;');
    catch err
        if strcmp(err.identifier, 'nxr:notWiredInMex')
            fprintf('  ✓ %s throws nxr:notWiredInMex\n', label);
        else
            fprintf(2, '  ✗ %s — wrong identifier: %s\n', label, err.identifier);
            evalin('base', 'failures = failures + 1;');
        end
    end
end

expectNotWired('measure.distance',                 @() nxr.manifold.measure.distance(mctx, int32(0)));
expectNotWired('measure.curvature',                @() nxr.manifold.measure.curvature(mctx));
expectNotWired('measure.frame',                    @() nxr.manifold.measure.frame(mctx));
expectNotWired('uv.bff',                           @() nxr.manifold.uv.bff(mctx));
expectNotWired('operator.connectionLaplacian',     @() nxr.manifold.operator.connectionLaplacian(mctx, struct()));
expectNotWired('operator.d0',                      @() nxr.manifold.operator.d0(mctx));
expectNotWired('solve.poisson',                    @() nxr.manifold.solve.poisson(mctx, int32(0), 1));
expectNotWired('interpolate.trivial',       @() nxr.manifold.interpolate.trivial(mctx, int32(0), 1));

if failures == 0
    fprintf('\n[smoke-matlab-manifold] all assertions passed ✓\n');
    exit(0);
else
    fprintf(2, '\n[smoke-matlab-manifold] %d FAILED ✗\n', failures);
    exit(1);
end
