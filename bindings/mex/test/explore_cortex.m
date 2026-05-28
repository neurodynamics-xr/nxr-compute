% explore_cortex.m — run every nxr.manifold.* function on a real Brainstorm
% cortical surface and COLLECT every result into one structure `M` (for
% Manifold) that mirrors the nxr.manifold.* groups, so you can explore,
% analyze, and test the outputs afterwards.
%
%   After running, your workspace holds `M` with, e.g.:
%       M.operator.stiffness            cotan Laplacian (sparse nV x nV)
%       M.operator.mass                 Voronoi mass    (sparse nV x nV)
%       M.operator.d0 / d1              DEC exterior derivatives
%       M.operator.star0 / star1 / star2 / star1Inverse   Hodge stars
%       M.operator.connectionLaplacian  struct
%       M.solve.eigen.eigenvectors      [nV x k]   (raw modes)
%       M.solve.eigen.eigenvalues       [k x 1]
%       M.solve.precompute              normalized + DC-removed modes
%       M.solve.poisson                 [nV x 1]
%       M.solve.hodge                   struct (dAlpha, deltaBeta, gamma, ...)
%       M.solve.heat                    [T x nV] spectral diffusion
%       M.measure.distance              [nV x 1] geodesic distance
%       M.measure.curvature.gaussian/mean/kMin/kMax/principalDirMax
%       M.measure.frame.e1/e2/normals   per-face frames
%       M.measure.normal                [nV x 3] vertex normals
%       M.interpolate.transport/extend/smoothVertex/smoothFace/trivial
%       M.query.isoline                 struct (positions, segmentCount)
%       M.uv.logMap / M.uv.stripe       structs
%       M.reference.eigenvalues/normals/curvature   the surface's own fields
%
% It then validates the results both NUMERICALLY (Gauss-Bonnet, Hodge
% recomposition, M-orthonormality, geodesic d(src)=0, normals vs reference)
% and VISUALLY via nxr.viz (a dashboard of the key results, also saved to
% build/viz/explore_dashboard.png). Set showViz = false to skip the figure.
%
% Run:  edit the two paths below if needed, then in MATLAB:
%         run('bindings/mex/test/explore_cortex.m')

clear; clc;

% ── config ────────────────────────────────────────────────────
repo    = '/Users/diellorbasha/workspace/research/code/nxr-compute';
dataset = '/Users/diellorbasha/workspace/library/datasets/bst_cortex5.mat';
kModes  = 50;     % eigenmodes to solve
showViz = true;   % render the nxr.viz dashboard to validate results visually

addpath(fullfile(repo, 'bindings', 'mex', 'matlab'));
hits = dir(fullfile(repo, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(hits), 'nxr_compute.%s not built under %s/build', mexext, repo);
addpath(hits(1).folder);

% ── load the surface ──────────────────────────────────────────
S = load(dataset);
c = S.bst_cortex;
V = double(c.Vertices);
F = double(c.Faces);
fprintf('nxr_compute %s — exploring "%s"\n', nxr_compute('version'), c.Comment);
fprintf('mesh: nV=%d  nF=%d\n\n', size(V,1), size(F,1));

% ── build the context + seed the results structure M ──────────
ctx = nxr.manifold.context(V, F);

M           = struct();
M.comment   = c.Comment;
M.vertices  = V;
M.faces     = F;
M.nV        = ctx.nV;
M.nE        = ctx.nE;
M.nF        = ctx.nF;
M.totalArea = ctx.totalArea;
M.context   = ctx;                      % the raw nxr.manifold.context struct

% the surface's own reference fields, for comparison/analysis
M.reference.eigenvalues  = c.Eigenmodes.Values;
M.reference.eigenvectors = c.Eigenmodes.Vectors;
M.reference.normals      = double(c.VertNormals);
M.reference.curvature    = double(c.Curvature);

% inputs we generate (kept so results are reproducible / analyzable)
rng(0);
omega = randn(ctx.nE, 1);               % random 1-form for the Hodge decomposition
M.input.omega  = omega;
M.input.kModes = kModes;

fprintf('context: nV=%d nE=%d nF=%d  (totalArea=%.4f)\n\n', M.nV, M.nE, M.nF, M.totalArea);

% ── compute every leaf and store it in M ──────────────────────
% store(M, 'group.name', @() <call>) runs the call, assigns M.group.name,
% times it, and prints OK/FAIL — never aborting.
fprintf('=== computing + storing every nxr.manifold.* result into M ===\n');

% operators
M = store(M, 'operator.stiffness',          @() nxr.manifold.operator.stiffness(ctx));
M = store(M, 'operator.mass',               @() nxr.manifold.operator.mass(ctx));
M = store(M, 'operator.laplacian',          @() nxr.manifold.operator.laplacian(ctx));
M = store(M, 'operator.d0',                 @() nxr.manifold.operator.d0(ctx));
M = store(M, 'operator.d1',                 @() nxr.manifold.operator.d1(ctx));
M = store(M, 'operator.star0',              @() nxr.manifold.operator.star0(ctx));
M = store(M, 'operator.star1',              @() nxr.manifold.operator.star1(ctx));
M = store(M, 'operator.star2',              @() nxr.manifold.operator.star2(ctx));
M = store(M, 'operator.star1Inverse',       @() nxr.manifold.operator.star1Inverse(ctx));
M = store(M, 'operator.connectionLaplacian',@() nxr.manifold.operator.connectionLaplacian(ctx));

% solve
M = store(M, 'solve.eigen',      @() nxr.manifold.solve.eigen(ctx, kModes));
M = store(M, 'solve.precompute', @() nxr.manifold.solve.precompute(ctx, kModes));
M = store(M, 'solve.poisson',    @() nxr.manifold.solve.poisson(ctx, [1 10000], [1 -1]));
M = store(M, 'solve.hodge',      @() nxr.manifold.solve.hodge(ctx, omega));
M = store(M, 'solve.heat',       @() nxr.manifold.solve.heat(ctx, 1, 1.0, [0 0.005 0.01 0.02]));

% measure
M = store(M, 'measure.distance',       @() nxr.manifold.measure.distance(ctx, 1));
M = store(M, 'measure.signedDistance', @() nxr.manifold.measure.signedDistance(ctx, [1 2 3 4 5], false));
M = store(M, 'measure.curvature',      @() nxr.manifold.measure.curvature(ctx));
M = store(M, 'measure.frame',          @() nxr.manifold.measure.frame(ctx));
M = store(M, 'measure.normal',         @() nxr.manifold.measure.normal(ctx));
M = store(M, 'measure.normalArea',     @() nxr.manifold.measure.normal(ctx, 'area'));

% interpolate
M = store(M, 'interpolate.transport',    @() nxr.manifold.interpolate.transport(ctx, 1, [1 0 0]));
M = store(M, 'interpolate.extend',       @() nxr.manifold.interpolate.extend(ctx, [1 10000], [1 0]));
M = store(M, 'interpolate.smoothVertex', @() nxr.manifold.interpolate.smoothVertex(ctx, 2));
M = store(M, 'interpolate.smoothFace',   @() nxr.manifold.interpolate.smoothFace(ctx, 4));
M = store(M, 'interpolate.trivial',      @() nxr.manifold.interpolate.trivial(ctx, [1 5000 10000 15000], [1 1 1 1]));

% query / uv
M = store(M, 'query.isoline', @() nxr.manifold.query.isoline(ctx, M.reference.curvature, 0.0));
M = store(M, 'uv.logMap',     @() nxr.manifold.uv.logMap(ctx, 1));
if isfield(M, 'interpolate') && isfield(M.interpolate, 'smoothVertex')
    vfRaw = M.interpolate.smoothVertex.vertexFieldRaw;   % stripe consumes the raw 2-RoSy field
    M = store(M, 'uv.stripe', @() nxr.manifold.uv.stripe(ctx, vfRaw, 5.0));
end

% ── expected non-results: stored as explanatory strings ───────
% bff needs a boundary; this cortex is closed.
try
    M.uv.bff = nxr.manifold.uv.bff(ctx);
    fprintf('  [OK]   %-26s %s\n', 'uv.bff', describe(M.uv.bff));
catch e
    M.uv.bff = sprintf('N/A — closed surface has no boundary (%s)', e.identifier);
    fprintf('  [NOTE] %-26s %s\n', 'uv.bff', M.uv.bff);
end
% findCenter needs a connected mesh; this cortex is two hemispheres.
try
    M.query.center = nxr.manifold.query.center(ctx, [1 5000 10000]);
    fprintf('  [OK]   %-26s %s\n', 'query.center', describe(M.query.center));
catch e
    M.query.center = sprintf('N/A — needs a connected mesh, this has 2 components (%s)', e.identifier);
    fprintf('  [NOTE] %-26s %s\n', 'query.center', M.query.center);
end

% ── numeric validation (reads the stored results in M) ────────
fprintf('\n=== numeric validation (uses the stored M.* results) ===\n');
vPass = 0; vN = 0;

chi = M.nV - (3*M.nF/2) + M.nF;
gb  = sum(M.measure.curvature.gaussian);
[vPass, vN] = tally(vPass, vN, check('Gauss-Bonnet', abs(gb - 2*pi*chi) < 1e-3, ...
    sprintf('sum(gaussian)=%.4f vs 2*pi*chi=%.4f (chi=%d)', gb, 2*pi*chi, chi)));

hd  = M.solve.hodge;
rec = max(abs(hd.dAlpha + hd.deltaBeta + hd.gamma - hd.omega));
[vPass, vN] = tally(vPass, vN, check('Hodge recomposition', rec < 1e-8, sprintf('err=%.2e', rec)));

U   = M.solve.precompute.eigenvectors;
G   = U' * M.operator.mass * U;
oe  = norm(G - eye(size(G)), 'fro');
[vPass, vN] = tally(vPass, vN, check('M-orthonormality', oe < 1e-6, sprintf('||U''MU - I||=%.2e', oe)));

gd  = M.measure.distance;
[vPass, vN] = tally(vPass, vN, check('geodesic distance', abs(gd(1)) < 1e-9 && min(gd) > -1e-9, ...
    sprintf('d(src)=%.1e min=%.1e max=%.4f', gd(1), min(gd), max(gd))));

nn   = M.measure.normal; rn = M.reference.normals;
cosv = sum(nn .* rn, 2) ./ max(vecnorm(nn,2,2) .* vecnorm(rn,2,2), eps);
[vPass, vN] = tally(vPass, vN, check('normals vs reference', mean(abs(cosv)) > 0.95, ...
    sprintf('mean|cos|=%.4f', mean(abs(cosv)))));

fprintf('validation: %d/%d PASS\n', vPass, vN);

% ── reference eigenvalue cross-check ──────────────────────────
fprintf('\n=== eigenvalues: nxr (precompute) vs reference Eigenmodes ===\n');
fprintf('reference MassType=%s  nRemoved=%g  nModes=%d\n', ...
    c.Eigenmodes.MassType, c.Eigenmodes.nRemoved, c.Eigenmodes.nModes);
ev = M.solve.precompute.eigenvalues; rv = M.reference.eigenvalues;
fprintf('  %-4s %14s %14s\n', 'idx', 'nxr lambda', 'ref lambda');
for i = 1:min(8, numel(ev))
    r = NaN; if i <= numel(rv), r = rv(i); end
    fprintf('  %-4d %14.6g %14.6g\n', i, ev(i), r);
end
fprintf('(nxr keeps 1 residual DC mode — 2 hemispheres; reference removed %g.)\n', c.Eigenmodes.nRemoved);

% ── visual validation via nxr.viz ─────────────────────────────
% One dashboard panel covering the key results (surface, an eigenmode,
% curvature, geodesic distance, normals, spectrum vs reference), so the
% stored M can be eyeballed alongside the numeric checks above.
if showViz
    fprintf('\n=== visual validation (nxr.viz dashboard) ===\n');
    fd = nxr.viz.dashboard(M);
    vizdir = fullfile(repo, 'build', 'viz');
    if ~exist(vizdir, 'dir'), mkdir(vizdir); end
    pngfile = fullfile(vizdir, 'explore_dashboard.png');
    exportgraphics(fd, pngfile, 'Resolution', 130);
    fprintf('  dashboard opened + saved to %s\n', pngfile);
    fprintf('  drill into any result with e.g.  nxr.viz.show(M, ''solve.eigen'', 6)\n');
end

% ── done — M is now in your workspace ─────────────────────────
fprintf('\n[explore_cortex] done. Results are in the struct M:\n');
fprintf('  groups: %s\n', strjoin(fieldnames(M)', ', '));
fprintf('  e.g.  M.operator.stiffness   M.solve.eigen.eigenvectors\n');
fprintf('        M.measure.curvature.gaussian   M.solve.hodge.dAlpha\n');
fprintf('        M.reference.eigenvalues (compare to M.solve.precompute.eigenvalues)\n');
fprintf('  visualize:  nxr.viz.dashboard(M)   nxr.viz.show(M, ''measure.distance'')\n');

% ── local helpers ─────────────────────────────────────────────
function M = store(M, path, fn)
    t = tic;
    parts = strsplit(path, '.');
    try
        val = fn();
        M.(parts{1}).(parts{2}) = val;
        fprintf('  [OK]   %-26s %6.2fs  %s\n', path, toc(t), describe(val));
    catch e
        fprintf('  [FAIL] %-26s %6.2fs  [%s] %s\n', path, toc(t), e.identifier, e.message);
    end
end

function s = describe(v)
    if ischar(v) || isstring(v)
        s = char(v);
    elseif isstruct(v)
        s = ['struct{ ' strjoin(fieldnames(v)', ', ') ' }'];
    elseif issparse(v)
        s = sprintf('sparse [%s]', num2str(size(v)));
    else
        s = sprintf('%s [%s]', class(v), num2str(size(v)));
    end
end

function ok = check(name, cond, detail)
    if cond, tag = 'PASS'; else, tag = 'CHECK'; end
    fprintf('  %-28s %-5s  %s\n', name, tag, detail);
    ok = cond;
end

function [p, n] = tally(p, n, ok)
    p = p + ok;
    n = n + 1;
end
