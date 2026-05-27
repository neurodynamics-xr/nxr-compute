% test_mex_context.m — stateful MEX context: lifecycle, caching,
% coexistence, and the invalid-handle error contract. Run as a script
% (asserts + fprintf), mirroring nxr_compute_mex_smoke.m.
%
% Run from MATLAB (or the matlab MCP run_matlab_file tool).

clear; clc;
fprintf('[test_mex_context] starting\n');

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
hits = dir(fullfile(repoRoot, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not found under %s/build', mexext, repoRoot);
addpath(hits(1).folder);

[V, F] = local_icosahedron();
nV = size(V, 1);
nF = size(F, 1);

% ── create / destroy ──────────────────────────────────────────
h1 = nxr_compute('create', V, F);
assert(isa(h1, 'uint64') && isscalar(h1), 'create must return a uint64 scalar handle');
h2 = nxr_compute('create', V, F);
assert(h1 ~= h2, 'distinct contexts must get distinct handles');
fprintf('  create/destroy ✓ (h1=%d, h2=%d)\n', h1, h2);

% ── assembleManifoldOperators: handle path == stateless path ──
opsH = nxr_compute('assembleManifoldOperators', h1);
opsS = nxr_compute('assembleManifoldOperators', V, F);
assert(opsH.nV == nV && opsH.nF == nF, 'handle ops sizes');
assert(norm(opsH.cotanLaplacian - opsS.cotanLaplacian, 'fro') < 1e-12, 'K handle==stateless');
assert(norm(opsH.mass - opsS.mass, 'fro') < 1e-12, 'M handle==stateless');
assert(abs(opsH.totalArea - opsS.totalArea) < 1e-12, 'area handle==stateless');

% Cache hit: second assemble returns the cached struct (numerically identical).
t1 = tic; nxr_compute('assembleManifoldOperators', h1);            cold = toc(t1);
t2 = tic; opsH2 = nxr_compute('assembleManifoldOperators', h1);   warm = toc(t2);
assert(norm(opsH2.cotanLaplacian - opsH.cotanLaplacian, 'fro') < 1e-15, 'cached K identical');
fprintf('  assembleManifoldOperators handle path ✓ (cold %.3g s, warm %.3g s)\n', cold, warm);

% ── solve in handle mode: K/M come from cached ops ────────────
eig = nxr_compute('solve', h1, 6);
assert(eig.k == 6, 'handle solve returns k modes');
assert(isequal(size(eig.eigenvectors), [nV 6]), 'eigvec V×k');
assert(issorted(eig.eigenvalues), 'eigvals ascending');
assert(abs(eig.eigenvalues(2) - 2.0) < 1e-6, 'icosa λ₁≈2 (got %g)', eig.eigenvalues(2));
eigS = nxr_compute('solve', h1, 6, -1e-8);   % optional sigma accepted in handle mode
assert(eigS.k == 6 && max(abs(eigS.eigenvalues - eig.eigenvalues)) < 1e-10, ...
    'handle solve honors optional sigma');
fprintf('  solve handle path ✓ (λ=[%s])\n', ...
    strjoin(arrayfun(@(x) sprintf('%.3f', x), eig.eigenvalues, 'UniformOutput', false), ', '));

% ── two coexisting contexts on different meshes ───────────────
% h2 was created on V; make a scaled context to prove independence.
V2 = V * 2;                              % 2× linear → 4× area
hScaled = nxr_compute('create', V2, F);
opsScaled = nxr_compute('assembleManifoldOperators', hScaled);
assert(abs(opsScaled.totalArea - 4 * opsH.totalArea) < 1e-6, ...
    'scaled mesh should have 4× area (got %g vs %g)', opsScaled.totalArea, opsH.totalArea);

% Destroy h1; hScaled and h2 must survive.
nxr_compute('destroy', h1);
opsScaled2 = nxr_compute('assembleManifoldOperators', hScaled);
assert(abs(opsScaled2.totalArea - opsScaled.totalArea) < 1e-12, 'hScaled survives h1 destroy');
opsH2b = nxr_compute('assembleManifoldOperators', h2);
assert(opsH2b.nV == nV, 'h2 survives h1 destroy');
fprintf('  coexistence ✓ (scaled area %.4f = 4 × %.4f)\n', opsScaled.totalArea, opsH.totalArea);

% ── invalid / destroyed handle raises nxr:invalidHandle ───────
caught = false;
try
    nxr_compute('assembleManifoldOperators', h1);   % h1 was destroyed
catch e
    caught = strcmp(e.identifier, 'nxr:invalidHandle');
end
assert(caught, 'using a destroyed handle must raise nxr:invalidHandle');

% A never-issued handle also raises.
caught2 = false;
try
    nxr_compute('assembleManifoldOperators', uint64(999999));
catch e
    caught2 = strcmp(e.identifier, 'nxr:invalidHandle');
end
assert(caught2, 'unknown handle must raise nxr:invalidHandle');
fprintf('  invalid-handle contract ✓\n');

% ── remaining wired ops in handle mode (Phase 1.5) ────────────
hOps = h2;   % still alive; created on the icosahedron

pc = nxr_compute('precompute', hOps, 6);
assert(pc.k == 5, 'precompute drops DC mode → k=5 (got %d)', pc.k);

tp = nxr_compute('parallel', hOps, 1, [1 0 0]);
assert(isequal(size(tp), [nV 3]), 'parallel → V×3');
assert(any(vecnorm(tp, 2, 2) > 1e-6), 'parallel nontrivial');

ex = nxr_compute('extendScalar', hOps, [1; 7], [1.0; 0.0]);
assert(numel(ex) == nV, 'extendScalar → nV');

lm = nxr_compute('logMap', hOps, 1);
assert(isequal(size(lm.logCoords), [nV 2]), 'logMap → V×2');
assert(norm(lm.logCoords(1, :)) < 1e-6, 'logMap zero at source');

ct = nxr_compute('findCenter', hOps, [1; 2; 6]);
assert(numel(ct) == 3 && all(isfinite(ct)), 'findCenter → finite 3-vec');

sd = nxr_compute('signedHeat', hOps, [12; 6; 2; 8; 11], 1);
assert(numel(sd) == nV && min(sd) < 0 && max(sd) > 0, 'signedHeat straddles zero');

ff = nxr_compute('smoothFace', hOps, 4);
assert(isequal(size(ff), [nF 3]), 'smoothFace → F×3');
ff2 = nxr_compute('smoothFace', hOps, 4);
assert(isequal(ff, ff2), 'smoothFace cached identical');

vf = nxr_compute('smoothVertex', hOps, 2);
assert(isequal(size(vf.vertexVectors), [nV 3]), 'smoothVertex vectors V×3');
assert(numel(vf.vertexFieldRaw) == 2 * nV, 'smoothVertex raw 2*nV');
assert(vf.nSym == 2, 'smoothVertex nSym round-trip');

st = nxr_compute('compute', hOps, vf.vertexFieldRaw, 8.0);
assert(st.segmentCount >= 0, 'stripes segmentCount >= 0');
assert(isequal(size(st.positions), [st.segmentCount * 2, 3]), 'stripes positions (2*segs)×3');
fprintf('  remaining wired ops (handle) ✓\n');

% ── cleanup ───────────────────────────────────────────────────
nxr_compute('destroy', h2);
nxr_compute('destroy', hScaled);
fprintf('[test_mex_context] all assertions passed ✓\n');

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
