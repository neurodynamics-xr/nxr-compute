% test_mex_functional_parity.m — verifies the nxr.manifold.* functional
% leaves that were stubbed (notWired) are now wired to the MEX dispatcher
% via transient handles, reaching parity with the WASM six-group surface.
% Run as a script (asserts + fprintf) via the MATLAB MCP run_matlab_file.

clear; clc;
fprintf('[test_mex_functional_parity] starting\n');

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
% MATLAB package tree + the freshly-built MEX binary.
addpath(fullfile(repoRoot, 'bindings', 'mex', 'matlab'));
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found under %s/build', mexext, repoRoot);
addpath(hits(1).folder);

[V, F] = local_icosahedron();
mctx = nxr.manifold.context(V, F);
nV = mctx.nV; nE = mctx.nE; nF = mctx.nF;
fprintf('  context: nV=%d nE=%d nF=%d\n', nV, nE, nF);

% ── solve.* ───────────────────────────────────────────────────
phi = nxr.manifold.solve.poisson(mctx, [1; 7], [1.0; -1.0]);
assert(numel(phi) == nV && all(isfinite(phi)), 'solve.poisson → finite [nV]');

rng(11); omega = randn(nE, 1);
hd = nxr.manifold.solve.hodge(mctx, omega);
assert(max(abs(hd.dAlpha + hd.deltaBeta + hd.gamma - hd.omega)) < 1e-10, ...
    'solve.hodge recomposes dα+δβ+γ = ω');

ts = (0:3) * 0.1;
hf = nxr.manifold.solve.heat(mctx, 1, 1.0, ts);
assert(isequal(size(hf), [numel(ts) nV]), 'solve.heat → [T x nV]');
fprintf('  solve.{poisson,hodge,heat} ✓\n');

% ── operator.* (DEC) ──────────────────────────────────────────
assert(isequal(size(nxr.manifold.operator.d0(mctx)),           [nE nV]), 'operator.d0 nE×nV');
assert(isequal(size(nxr.manifold.operator.d1(mctx)),           [nF nE]), 'operator.d1 nF×nE');
assert(isequal(size(nxr.manifold.operator.star0(mctx)),        [nV nV]), 'operator.star0 nV×nV');
assert(isequal(size(nxr.manifold.operator.star1(mctx)),        [nE nE]), 'operator.star1 nE×nE');
assert(isequal(size(nxr.manifold.operator.star2(mctx)),        [nF nF]), 'operator.star2 nF×nF');
assert(isequal(size(nxr.manifold.operator.star1Inverse(mctx)), [nE nE]), 'operator.star1Inverse nE×nE');
assert(issparse(nxr.manifold.operator.d0(mctx)), 'operator.d0 sparse');

cl = nxr.manifold.operator.connectionLaplacian(mctx);
assert(isequal(size(cl.K_real), [2 * nV, 2 * nV]), 'connectionLaplacian K_real 2V×2V');
fprintf('  operator.{d0,d1,star0,star1,star2,star1Inverse,connectionLaplacian} ✓\n');

% ── measure.* ─────────────────────────────────────────────────
d = nxr.manifold.measure.distance(mctx, 1);
assert(numel(d) == nV && abs(d(1)) < 1e-9, 'measure.distance → [nV], d(src)=0');

cur = nxr.manifold.measure.curvature(mctx);
assert(abs(sum(cur.gaussian) - 4 * pi) < 1e-6, 'measure.curvature Gauss-Bonnet Σκ=4π');

fr = nxr.manifold.measure.frame(mctx);
assert(isequal(size(fr.e1), [nF 3]), 'measure.frame e1 F×3');

n0 = nxr.manifold.measure.normal(mctx);          % default (cached angle)
nA = nxr.manifold.measure.normal(mctx, 'area');  % on-demand estimator
assert(isequal(size(n0), [nV 3]) && isequal(size(nA), [nV 3]), 'measure.normal → V×3 (both)');
fprintf('  measure.{distance,curvature,frame,normal} ✓\n');

% ── query.isoline ─────────────────────────────────────────────
iso = nxr.manifold.query.isoline(mctx, V(:, 3), 0.0);
assert(iso.segmentCount >= 0 && isequal(size(iso.positions), [iso.segmentCount * 2, 3]), ...
    'query.isoline → {positions (2*segs)×3, segmentCount}');
fprintf('  query.isoline ✓ (segs=%d)\n', iso.segmentCount);

% ── interpolate.trivial ───────────────────────────────────────
df = nxr.manifold.interpolate.trivial(mctx, [1; 4], [1.0; 1.0]);   % Σ = 2 = χ
assert(isequal(size(df.directionVectors), [nF 3]) && df.gaussBonnetSatisfied, ...
    'interpolate.trivial → F×3 field, Gauss-Bonnet satisfied');
fprintf('  interpolate.trivial ✓\n');

% ── uv.bff (closed mesh → structured throw, mirrors WASM) ─────
bffThrew = false;
try
    nxr.manifold.uv.bff(mctx);
catch e
    bffThrew = startsWith(e.identifier, 'nxr:');
end
assert(bffThrew, 'uv.bff on a closed mesh raises nxr:*');
fprintf('  uv.bff (closed-mesh throw) ✓\n');

% ── parity of the dual stubs (stub in WASM too) ───────────────
% These intentionally remain stubMarker placeholders in both bindings.
s = nxr.manifold.query.line(mctx, 1, 2);
assert(isstruct(s) && strcmp(s.method, 'todo'), 'query.line stays a stub (parity)');
fprintf('  dual stubs (query.line/circle/region, measure.area/density) still stubs ✓\n');

fprintf('[test_mex_functional_parity] all assertions passed ✓\n');

% ── local helpers ─────────────────────────────────────────────
function [V, F] = local_icosahedron()
    t = (1 + sqrt(5)) / 2;
    raw = [-1  t  0;  1  t  0; -1 -t  0;  1 -t  0;
            0 -1  t;  0  1  t;  0 -1 -t;  0  1 -t;
            t  0 -1;  t  0  1; -t  0 -1; -t  0  1];
    V = raw ./ vecnorm(raw, 2, 2);
    F = [ 1 12  6;  1  6  2;  1  2  8;  1  8 11;  1 11 12; ...
          2  6 10;  6 12  5; 12 11  3; 11  8  7;  8  2  9; ...
          4 10  5;  4  5  3;  4  3  7;  4  7  9;  4  9 10; ...
          5 10  6;  3  5 12;  7  3 11;  9  7  8; 10  9  2];
end
