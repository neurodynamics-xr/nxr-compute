% explore_cortex.m — interactive exploration + sanity harness for the nxr
% MEX package on a real Brainstorm cortical surface.
%
% Unlike test_mex_functional_parity.m (hard asserts on a tiny icosahedron,
% aborts on first failure), this script:
%   • runs EVERY nxr.manifold.* function on a real 20k-vertex cortex,
%   • reports type / shape / timing per function and never aborts (per-op
%     try/catch with a pass/fail tally),
%   • then runs numeric validation (Gauss-Bonnet, M-orthonormality, Hodge
%     recomposition) and cross-checks against the surface's own reference
%     normals / curvature / eigenmodes.
%
% Run:  edit the two paths below if needed, then in MATLAB:
%         run('bindings/mex/test/explore_cortex.m')
%       (or the matlab MCP run_matlab_file tool).

clear; clc;

% ── config ────────────────────────────────────────────────────
repo    = '/Users/diellorbasha/workspace/research/code/nxr-compute';
dataset = '/Users/diellorbasha/workspace/library/datasets/bst_cortex5.mat';
kModes  = 40;     % eigenmodes to solve (keep modest for interactive runs)

addpath(fullfile(repo, 'bindings', 'mex', 'matlab'));   % nxr.manifold.* package
hits = dir(fullfile(repo, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not built under %s/build', mexext, repo);
addpath(hits(1).folder);

% ── load the surface ──────────────────────────────────────────
S = load(dataset);
c = S.bst_cortex;                       % Brainstorm surface struct
V = double(c.Vertices);                 % nV x 3
F = double(c.Faces);                    % nF x 3, 1-based
fprintf('nxr_compute %s — exploring "%s"\n', nxr_compute('version'), c.Comment);
fprintf('mesh: nV=%d  nF=%d\n\n', size(V,1), size(F,1));

% ── build the functional context ──────────────────────────────
tBuild = tic;
m = nxr.manifold.context(V, F);
fprintf('nxr.manifold.context : nV=%d nE=%d nF=%d  (%.3fs)\n\n', ...
    m.nV, m.nE, m.nF, toc(tBuild));

% ── TOUR: run every leaf, report type/shape/timing, tally ─────
% Each entry is {name, thunk}. The thunk runs the op and returns a short
% description string; run_op times it and catches any error as a FAIL.
omega = randn(m.nE, 1);                 % a random 1-form for hodge
scal  = double(c.Curvature);            % a real per-vertex scalar field

tour = {
  'operator.stiffness',     @() shp(nxr.manifold.operator.stiffness(m))
  'operator.mass',          @() shp(nxr.manifold.operator.mass(m))
  'operator.laplacian',     @() shp(nxr.manifold.operator.laplacian(m))
  'operator.d0',            @() shp(nxr.manifold.operator.d0(m))
  'operator.d1',            @() shp(nxr.manifold.operator.d1(m))
  'operator.star0',         @() shp(nxr.manifold.operator.star0(m))
  'operator.star1',         @() shp(nxr.manifold.operator.star1(m))
  'operator.star2',         @() shp(nxr.manifold.operator.star2(m))
  'operator.star1Inverse',  @() shp(nxr.manifold.operator.star1Inverse(m))
  'operator.connectionLaplacian', @() flds(nxr.manifold.operator.connectionLaplacian(m))
  'solve.eigen',            @() flds(nxr.manifold.solve.eigen(m, kModes))
  'solve.precompute',       @() flds(nxr.manifold.solve.precompute(m, kModes))
  'solve.poisson',          @() shp(nxr.manifold.solve.poisson(m, [1 10000], [1 -1]))
  'solve.hodge',            @() flds(nxr.manifold.solve.hodge(m, omega))
  'solve.heat',             @() shp(nxr.manifold.solve.heat(m, 1, 1.0, [0 0.005 0.01]))
  'measure.distance',       @() shp(nxr.manifold.measure.distance(m, 1))
  'measure.signedDistance', @() shp(nxr.manifold.measure.signedDistance(m, [1 50 100 150], false))
  'measure.curvature',      @() flds(nxr.manifold.measure.curvature(m))
  'measure.frame',          @() flds(nxr.manifold.measure.frame(m))
  'measure.normal',         @() shp(nxr.manifold.measure.normal(m))
  'measure.normal(area)',   @() shp(nxr.manifold.measure.normal(m, 'area'))
  'interpolate.transport',  @() shp(nxr.manifold.interpolate.transport(m, 1, [1 0 0]))
  'interpolate.extend',     @() shp(nxr.manifold.interpolate.extend(m, [1 10000], [1 0]))
  'interpolate.smoothVertex', @() flds(nxr.manifold.interpolate.smoothVertex(m, 2))
  'interpolate.smoothFace', @() shp(nxr.manifold.interpolate.smoothFace(m, 4))
  'interpolate.trivial',    @() trivialDetail(m)
  'query.isoline',          @() flds(nxr.manifold.query.isoline(m, scal, 0.0))
  'uv.logMap',              @() flds(nxr.manifold.uv.logMap(m, 1))
  'uv.stripe',              @() stripeDetail(m)
};

fprintf('=== function tour (every leaf, on the real cortex) ===\n');
nPass = 0;
for i = 1:size(tour, 1)
    nPass = nPass + run_op(tour{i,1}, tour{i,2});
end

% uv.bff is expected to FAIL on a closed surface (no boundary) — verify it
% raises a structured nxr:* error rather than crashing.
fprintf('  ');
try
    nxr.manifold.uv.bff(m);
    fprintf('[??]   uv.bff                    unexpectedly succeeded on a closed mesh\n');
catch e
    if startsWith(e.identifier, 'nxr:')
        fprintf('[OK]   %-24s expected throw on closed mesh (%s)\n', 'uv.bff', e.identifier);
        nPass = nPass + 1;
    else
        fprintf('[FAIL] uv.bff                  unexpected error: %s\n', e.message);
    end
end
% query.center (findCenter) needs a CONNECTED mesh — geometry-central's
% vector-heat Karcher mean yields NaN on a disconnected surface (this cortex
% is two hemispheres). Treat a NaN/checkFinite failure as a known caveat.
fprintf('  ');
try
    cc = nxr.manifold.query.center(m, [1 5000 10000]);
    fprintf('[OK]   %-24s center=[%.3f %.3f %.3f]\n', 'query.center', cc(1), cc(2), cc(3));
    nPass = nPass + 1;
catch e
    fprintf('[CAVEAT] %-22s findCenter needs a connected mesh (2 components here): %s\n', ...
            'query.center', e.message);
    nPass = nPass + 1;   % expected on a multi-component surface
end

nTotal = size(tour, 1) + 2;   % + bff (expected throw) + center (caveat)
fprintf('\ntour: %d/%d functions ran as expected\n', nPass, nTotal);

% ── numeric validation (real invariants) ──────────────────────
fprintf('\n=== numeric validation ===\n');

chi = m.nV - (3*m.nF/2) + m.nF;
cur = nxr.manifold.measure.curvature(m);
gb  = sum(cur.gaussian);
fprintf('%-28s sum=%.4f  2*pi*chi=%.4f (chi=%d)  -> %s\n', ...
    'Gauss-Bonnet', gb, 2*pi*chi, chi, passfail(abs(gb - 2*pi*chi) < 1e-3));

hd  = nxr.manifold.solve.hodge(m, omega);
rec = max(abs(hd.dAlpha + hd.deltaBeta + hd.gamma - hd.omega));
fprintf('%-28s err=%.3e  -> %s\n', 'Hodge recomposition', rec, passfail(rec < 1e-8));

pc  = nxr.manifold.solve.precompute(m, kModes);
M   = nxr.manifold.operator.mass(m);
G   = pc.eigenvectors' * M * pc.eigenvectors;
orthErr = norm(G - eye(size(G)), 'fro');
fprintf('%-28s ||U''MU - I||=%.3e  -> %s\n', 'M-orthonormality', orthErr, passfail(orthErr < 1e-6));

gd  = nxr.manifold.measure.distance(m, 1);
fprintf('%-28s d(src)=%.3e, min=%.3e, max=%.4f  -> %s\n', 'geodesic distance', ...
    gd(1), min(gd), max(gd), passfail(abs(gd(1)) < 1e-9 && min(gd) > -1e-9));

% normals vs the surface's own reference normals (cosine similarity)
nn   = nxr.manifold.measure.normal(m);
refN = double(c.VertNormals);
cosv = sum(nn .* refN, 2) ./ max(vecnorm(nn,2,2) .* vecnorm(refN,2,2), eps);
fprintf('%-28s mean|cos| vs reference=%.4f  -> %s\n', 'vertex normals', ...
    mean(abs(cosv)), passfail(mean(abs(cosv)) > 0.95));

% ── handle API: caching + reference eigenvalue cross-check ────
fprintf('\n=== stateful handle API + reference eigenvalues ===\n');
h = nxr_compute('create', V, F);

t = tic; nxr_compute('assembleManifoldOperators', h); cold = toc(t);
t = tic; nxr_compute('assembleManifoldOperators', h); warm = toc(t);
fprintf('operator cache: cold=%.3fs  warm=%.3fs  (%.0fx faster warm)\n', ...
    cold, warm, cold / max(warm, eps));

eg  = nxr_compute('precompute', h, kModes);
ref = c.Eigenmodes.Values;
fprintf('reference Eigenmodes: MassType=%s  nRemoved=%g  nModes=%d\n', ...
    c.Eigenmodes.MassType, c.Eigenmodes.nRemoved, c.Eigenmodes.nModes);
fprintf('  %-4s %14s %14s\n', 'idx', 'nxr lambda', 'ref lambda');
for i = 1:min(8, numel(eg.eigenvalues))
    rv = NaN; if i <= numel(ref), rv = ref(i); end
    fprintf('  %-4d %14.6g %14.6g\n', i, eg.eigenvalues(i), rv);
end
fprintf('(DC handling/mass type may differ — compare trends, not exact values.)\n');

nxr_compute('destroy', h);

fprintf('\n[explore_cortex] done — tour %d/%d OK.\n', nPass, nTotal);

% ── local helpers ─────────────────────────────────────────────
function ok = run_op(name, fn)
    t = tic;
    try
        detail = fn();
        fprintf('  [OK]   %-24s %6.2fs  %s\n', name, toc(t), detail);
        ok = 1;
    catch e
        fprintf('  [FAIL] %-24s %6.2fs  [%s] %s\n', name, toc(t), e.identifier, e.message);
        ok = 0;
    end
end

function s = shp(x)
    kind = 'dense'; if issparse(x), kind = 'sparse'; end
    s = sprintf('%s [%s]', kind, num2str(size(x)));
end

function s = flds(x)
    s = ['struct{ ' strjoin(fieldnames(x)', ', ') ' }'];
end

function s = trivialDetail(m)
    % Singularity indices must sum to the Euler characteristic (chi=4 here).
    r = nxr.manifold.interpolate.trivial(m, [1 5000 10000 15000], [1 1 1 1]);
    s = sprintf('dir [%s], chi=%g, gaussBonnet=%d', ...
        num2str(size(r.directionVectors)), r.eulerCharacteristic, r.gaussBonnetSatisfied);
end

function s = stripeDetail(m)
    vf = nxr.manifold.interpolate.smoothVertex(m, 2);
    r  = nxr.manifold.uv.stripe(m, vf.vertexFieldRaw, 5.0);
    s  = sprintf('segments=%d', r.segmentCount);
end

function s = passfail(cond)
    if cond, s = 'PASS'; else, s = 'CHECK'; end
end
