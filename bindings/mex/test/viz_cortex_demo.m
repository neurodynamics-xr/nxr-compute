% viz_cortex_demo.m — render every nxr.viz primitive + the dashboard on a
% real Brainstorm cortex, exporting a PNG per view to build/viz/ for visual
% validation. Patch-based (validation only; production viz lives in
% nxr-viewer / nxr-design-system charts-manifold).
%
% Run:  run('bindings/mex/test/viz_cortex_demo.m')

clear; clc;
repo    = '/Users/diellorbasha/workspace/research/code/nxr-compute';
dataset = '/Users/diellorbasha/workspace/library/datasets/bst_cortex5.mat';
addpath(fullfile(repo, 'bindings', 'mex', 'matlab'));
mexhit = dir(fullfile(repo, 'build', '**', ['nxr_compute.' mexext]));
assert(~isempty(mexhit), 'nxr_compute.%s not built', mexext);
addpath(mexhit(1).folder);
outdir = fullfile(repo, 'build', 'viz');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% ── load + compute a minimal M (subset of explore_cortex) ─────
S = load(dataset); c = S.bst_cortex;
V = double(c.Vertices); F = double(c.Faces);
ctx = nxr.manifold.context(V, F);

M = struct();
M.comment = c.Comment; M.vertices = V; M.faces = F;
M.nV = ctx.nV; M.nE = ctx.nE; M.nF = ctx.nF;
M.solve.precompute       = nxr.manifold.solve.precompute(ctx, 30);
M.measure.curvature      = nxr.manifold.measure.curvature(ctx);
M.measure.distance       = nxr.manifold.measure.distance(ctx, 1);
M.measure.normal         = nxr.manifold.measure.normal(ctx);
M.interpolate.smoothFace = nxr.manifold.interpolate.smoothFace(ctx, 4);
M.reference.eigenvalues  = c.Eigenmodes.Values;

% ── render each primitive (3-D surface views) ────────────────
snap3(nxr.viz.surface(V, F, 'Title', 'surface'),                 outdir, 'surface.png');
snap3(nxr.viz.show(M, 'solve.precompute', 6),                    outdir, 'eigenmode.png');
snap3(nxr.viz.show(M, 'measure.curvature.mean'),                 outdir, 'curvature.png');
snap3(nxr.viz.show(M, 'measure.distance'),                       outdir, 'distance.png');
snap3(nxr.viz.show(M, 'measure.normal'),                         outdir, 'normals.png');
snap3(nxr.viz.show(M, 'interpolate.smoothFace'),                 outdir, 'smoothface.png');

% ── analysis plot + dashboard (2-D / multi-axes) ─────────────
hs = nxr.viz.spectrum(M.solve.precompute.eigenvalues, 'Reference', M.reference.eigenvalues);
exportgraphics(ancestor(hs, 'figure'), fullfile(outdir, 'spectrum.png'), 'Resolution', 130);
close(ancestor(hs, 'figure'));

fd = nxr.viz.dashboard(M);
exportgraphics(fd, fullfile(outdir, 'dashboard.png'), 'Resolution', 130);
close(fd);

fprintf('[viz_cortex_demo] wrote PNGs to %s\n', outdir);
fprintf('  surface, eigenmode, curvature, distance, normals, smoothface, spectrum, dashboard\n');

% ── local helper ──────────────────────────────────────────────
function snap3(h, outdir, name)
    fig = ancestor(h, 'figure');
    set(fig, 'Position', [100 100 820 680]);
    ax = ancestor(h, 'axes');
    if ~isempty(ax) && isgraphics(ax), view(ax, 120, 20); end
    exportgraphics(fig, fullfile(outdir, name), 'Resolution', 130);
    close(fig);
end
