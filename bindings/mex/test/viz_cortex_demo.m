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

% Unit-length glyphs + magnitude-as-color: these fields are intrinsic
% (tangent), so we show direction everywhere at a fixed short length (no long
% arrows lifting off the folded cortex, no heavy-tail giant/invisible arrows)
% and carry magnitude in the colorbar. MinMag drops near-zero faces (e.g. a
% gradient on flat patches, where the direction is only numerical noise).
vopt = {'Scale', 1.2, 'NMax', 1500, 'MinMag', 0.05, ...
        'Normalize', true, 'ColorByMag', true, 'Colorbar', true};

% ── load + compute a minimal M (subset of explore_cortex) ─────
S = load(dataset); c = S.bst_cortex;
V = double(c.Vertices); F = double(c.Faces);
ctx = nxr.manifold.context(V, F);

% two singularities per hemisphere for the trivial-connection field
% (Σσ = χ_per_hemi = 2 each → Gauss-Bonnet satisfied across both components)
comps = splitComponents(V, F);
singV_trivial = [comps(1).vidx(1); comps(1).vidx(end); ...
                 comps(2).vidx(1); comps(2).vidx(end)];

M = struct();
M.comment = c.Comment; M.vertices = V; M.faces = F;
M.nV = ctx.nV; M.nE = ctx.nE; M.nF = ctx.nF;
M.solve.precompute       = nxr.manifold.solve.precompute(ctx, 30);
% ± dipole in EACH hemisphere — a single global dipole leaves the disconnected
% source-free hemisphere at a meaningless regularization constant (one solid
% color); the per-hemisphere display field below is the well-posed view.
M.solve.poisson          = nxr.manifold.solve.poisson(ctx, ...
    [comps(1).vidx(1); comps(1).vidx(end); comps(2).vidx(1); comps(2).vidx(end)], [1; -1; 1; -1]);
M.solve.hodge            = nxr.manifold.solve.hodge(ctx, randn(ctx.nE, 1));
M.measure.curvature      = nxr.manifold.measure.curvature(ctx);
M.measure.distance       = nxr.manifold.measure.distance(ctx, 1);
M.measure.normal         = nxr.manifold.measure.normal(ctx);
M.interpolate.smoothFace = nxr.manifold.interpolate.smoothFace(ctx, 4);
M.interpolate.smoothVertex = nxr.manifold.interpolate.smoothVertex(ctx, 2);
M.interpolate.trivial    = nxr.manifold.interpolate.trivial(ctx, singV_trivial, [1; 1; 1; 1]);
M.reference.eigenvalues  = c.Eigenmodes.Values;

% gradient + flip-out geodesic path have no functional-API leaf yet — drive
% them through the raw MEX with a transient stateful handle (see explore_cortex).
hctx = nxr_compute('create', V, F);
guard = onCleanup(@() nxr_compute('destroy', hctx)); %#ok<NASGU>
M.field.gradient_of_distance = nxr_compute('gradient', hctx, M.measure.distance);
vA = comps(1).vidx(1);  vB = comps(1).vidx(round(numel(comps(1).vidx)/2));
M.query.tracePath = nxr_compute('tracePath', hctx, int32(vA), int32(vB));
clear guard hctx;

% well-posed Poisson view: solve a ± dipole per hemisphere (each its own
% manifold) and scatter back, so both hemispheres show a real dipole.
poiField = perHemiField(V, comps, @hemiPoisson);

% ── render each primitive (3-D surface views) ────────────────
snap3(nxr.viz.surface(V, F, 'Title', 'surface'),                 outdir, 'surface.png');
snap3(nxr.viz.show(M, 'solve.precompute', 6),                    outdir, 'eigenmode.png');
snap3(nxr.viz.show(M, 'measure.curvature.mean'),                 outdir, 'curvature.png');
snap3(nxr.viz.show(M, 'measure.distance'),                       outdir, 'distance.png');
snap3(nxr.viz.show(M, 'measure.normal'),                         outdir, 'normals.png');
snap3(nxr.viz.show(M, 'interpolate.smoothFace'),                 outdir, 'smoothface.png');

% ── analyses (DEC / Hodge / Poisson / direction fields / path) ─
snap3(nxr.viz.scalar(V, F, poiField, 'Title', 'poisson \phi (dipole per hemisphere)'), ...
      outdir, 'poisson.png');
snap3(nxr.viz.vectorField(V, F, M.field.gradient_of_distance, ...
      'Title', 'gradient of geodesic distance', 'Color', [0.85 0.10 0.10], vopt{:}), ...
      outdir, 'gradient.png');
snap3(nxr.viz.vectorField(V, F, M.solve.hodge.dAlphaVectors, ...
      'Title', 'Hodge d\alpha (exact part)', 'Color', [0.90 0.45 0.10], vopt{:}), ...
      outdir, 'hodge_exact.png');
snap3(nxr.viz.vectorField(V, F, M.solve.hodge.deltaBetaVectors, ...
      'Title', 'Hodge \delta\beta (co-exact part)', 'Color', [0.10 0.60 0.20], vopt{:}), ...
      outdir, 'hodge_coexact.png');
snap3(nxr.viz.vectorField(V, F, M.interpolate.trivial.directionVectors, ...
      'Title', 'trivial connection field (2 sing/hemi)', 'Color', [0.60 0.10 0.70], vopt{:}), ...
      outdir, 'trivial.png');
snap3(nxr.viz.vectorField(V, F, M.interpolate.smoothVertex.vertexVectors, ...
      'Title', 'smoothest NRoSy-2 (vertex line field)', 'Color', [0.10 0.45 0.70], vopt{:}), ...
      outdir, 'smoothvertex.png');
snap3(nxr.viz.segments(polylineToSegments(M.query.tracePath), 'Surface', {V, F}, ...
      'Title', 'flip-out geodesic path', 'LineWidth', 3), ...
      outdir, 'geodesic_path.png');

% ── analysis plot + dashboard (2-D / multi-axes) ─────────────
hs = nxr.viz.spectrum(M.solve.precompute.eigenvalues, 'Reference', M.reference.eigenvalues);
exportgraphics(ancestor(hs, 'figure'), fullfile(outdir, 'spectrum.png'), 'Resolution', 130);
close(ancestor(hs, 'figure'));

fd = nxr.viz.dashboard(M);
exportgraphics(fd, fullfile(outdir, 'dashboard.png'), 'Resolution', 130);
close(fd);

fprintf('[viz_cortex_demo] wrote PNGs to %s\n', outdir);
fprintf('  primitives: surface, eigenmode, curvature, distance, normals, smoothface\n');
fprintf('  analyses:   poisson, gradient, hodge_exact, hodge_coexact, trivial, smoothvertex, geodesic_path\n');
fprintf('  plots:      spectrum, dashboard\n');

% ── local helpers ─────────────────────────────────────────────
function snap3(h, outdir, name)
    fig = ancestor(h, 'figure');
    set(fig, 'Position', [100 100 820 680]);
    ax = ancestor(h, 'axes');
    if ~isempty(ax) && isgraphics(ax), view(ax, 120, 20); end
    exportgraphics(fig, fullfile(outdir, name), 'Resolution', 130);
    close(fig);
end

function comps = splitComponents(V, F)
%SPLITCOMPONENTS  Connected components (hemispheres) of a triangle mesh.
    E   = [F(:,[1 2]); F(:,[2 3]); F(:,[3 1])];
    lbl = conncomp(graph(E(:,1), E(:,2)))';
    comps = struct('vidx', {}, 'V', {}, 'F', {});
    for k = 1:max(lbl)
        vidx  = find(lbl == k);
        remap = zeros(size(V,1), 1); remap(vidx) = 1:numel(vidx);
        fmask = all(ismember(F, vidx), 2);
        comps(k).vidx = vidx;
        comps(k).V    = V(vidx, :);
        comps(k).F    = remap(F(fmask, :));
    end
end

function s = polylineToSegments(pts)
%POLYLINETOSEGMENTS  Convert an [N x 3] polyline into [2*(N-1) x 3] endpoint
%   pairs for nxr.viz.segments (rows 2k-1, 2k are the ends of segment k).
    N = size(pts, 1);
    if N < 2, s = zeros(0, 3); return; end
    s = zeros(2*(N-1), 3);
    s(1:2:end, :) = pts(1:N-1, :);
    s(2:2:end, :) = pts(2:N, :);
end

function field = perHemiField(V, comps, fn)
%PERHEMIFIELD  Scatter a per-component field fn(Vh,Fh) into a full [nV x 1],
%   placing each component's result at its global vertex indices.
    field = nan(size(V,1), 1);
    for k = 1:numel(comps)
        field(comps(k).vidx) = fn(comps(k).V, comps(k).F);
    end
end

function p = hemiPoisson(Vh, Fh)
%HEMIPOISSON  Poisson φ of a ± dipole on a single hemisphere (its own manifold).
%   The per-component additive constant is gauge on a closed surface (only ∇φ
%   is physical), so we remove the mean for a comparable color scale.
    cx = nxr.manifold.context(Vh, Fh);
    p  = nxr.manifold.solve.poisson(cx, [1; size(Vh,1)], [1; -1]);
    p  = p - mean(p);
end
