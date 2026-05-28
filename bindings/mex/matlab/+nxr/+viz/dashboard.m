function f = dashboard(M)
%DASHBOARD  Tiled validation panel for an explore_cortex M struct.
%   f = nxr.viz.dashboard(M)
%
%   Renders several nxr results at once in a tiledlayout — surface, an
%   eigenmode, mean curvature, geodesic distance, vertex normals, and the
%   eigenvalue spectrum — for a one-glance visual sanity check. Panels are
%   drawn only for results present in M. Returns the figure handle.
%
%   See also nxr.viz.show, explore_cortex.
    V = M.vertices; F = M.faces;
    f = figure('Color', 'w', 'Position', [80 80 1200 760]);
    t = tiledlayout(f, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('%s — nxr.viz dashboard', M.comment), 'Interpreter', 'none');

    nxr.viz.surface(V, F, 'Parent', nexttile(t), 'Title', 'surface');

    if isfield(M, 'solve') && isfield(M.solve, 'precompute')
        k = min(6, size(M.solve.precompute.eigenvectors, 2));
        nxr.viz.scalar(V, F, M.solve.precompute.eigenvectors(:,k), ...
            'Parent', nexttile(t), 'Title', sprintf('eigenmode %d', k), 'Colorbar', false);
    end
    if isfield(M, 'measure') && isfield(M.measure, 'curvature')
        nxr.viz.scalar(V, F, M.measure.curvature.mean, ...
            'Parent', nexttile(t), 'Title', 'mean curvature', 'Colorbar', false);
    end
    if isfield(M, 'measure') && isfield(M.measure, 'distance')
        nxr.viz.scalar(V, F, M.measure.distance, ...
            'Parent', nexttile(t), 'Title', 'geodesic distance', 'Colorbar', false);
    end
    if isfield(M, 'measure') && isfield(M.measure, 'normal')
        nxr.viz.vectorField(V, F, M.measure.normal, ...
            'Parent', nexttile(t), 'Title', 'normals', 'Color', [0.1 0.3 0.9]);
    end
    if isfield(M, 'solve') && isfield(M.solve, 'precompute')
        ref = [];
        if isfield(M, 'reference') && isfield(M.reference, 'eigenvalues')
            ref = M.reference.eigenvalues;
        end
        nxr.viz.spectrum(M.solve.precompute.eigenvalues, ...
            'Parent', nexttile(t), 'Reference', ref, 'Title', 'spectrum (nxr vs ref)');
    end
end
