function h = scalar(V, F, values, opts)
%SCALAR  Render a per-vertex scalar field as a colormap on the surface.
%   h = nxr.viz.scalar(V, F, values)
%   h = nxr.viz.scalar(V, F, values, Name, Value, ...)
%
%   Colors the surface by the per-vertex scalar `values` [nV x 1] via
%   FaceVertexCData. Follows the recipe conventions:
%     • SIGNED   data (eigenvectors, signed distance, mean curvature) →
%                diverging blue-white-red colormap, symmetric clim [-a, a].
%     • UNSIGNED data (geodesic distance, magnitudes) → sequential parula,
%                clim [min, max].
%   Signed vs unsigned is auto-detected (any(values < 0)); override with
%   'Signed'.
%
%   Name-value options:
%     'Parent'    target axes (default: new figure)
%     'Title'     title string
%     'Signed'    [] auto (default) | true | false
%     'Colormap'  override colormap (Nx3); default per signed/unsigned
%     'Clim'      override color limits [lo hi]; default per signed/unsigned
%     'EdgeColor' default 'none'
%     'Alpha'     face transparency (default 1)
%     'Colorbar'  show colorbar (default true)
%
%   Returns the patch handle.
%
%   See also nxr.viz.surface, nxr.viz.divergingColormap, nxr.viz.show.
    arguments
        V (:,3) double
        F (:,3) double
        values (:,1) double
        opts.Parent = []
        opts.Title (1,:) char = ''
        opts.Signed = []
        opts.Colormap double = []
        opts.Clim double = []
        opts.EdgeColor = 'none'
        opts.Alpha (1,1) double = 1.0
        opts.Colorbar (1,1) logical = true
    end
    if numel(values) ~= size(V, 1)
        error('nxr:viz:scalar:size', ...
            'values must have nV = %d entries (got %d)', size(V,1), numel(values));
    end

    signed = opts.Signed;
    if isempty(signed)
        signed = any(values < -1e-12);
    end

    ax = resolveAxes(opts.Parent);
    h = patch(ax, 'Faces', F, 'Vertices', V, ...
        'FaceVertexCData', values, 'FaceColor', 'interp', 'EdgeColor', opts.EdgeColor, ...
        'FaceAlpha', opts.Alpha, 'FaceLighting', 'gouraud', ...
        'AmbientStrength', 0.55, 'DiffuseStrength', 0.65, 'SpecularStrength', 0.10);

    if isempty(opts.Colormap)
        if signed, cmap = nxr.viz.divergingColormap(256); else, cmap = parula(256); end
    else
        cmap = opts.Colormap;
    end
    colormap(ax, cmap);

    if ~isempty(opts.Clim)
        clim(ax, opts.Clim);
    elseif signed
        a = max(abs(values)); if a == 0, a = 1; end
        clim(ax, [-a a]);
    else
        lo = min(values); hi = max(values); if lo == hi, hi = lo + 1; end
        clim(ax, [lo hi]);
    end

    axis(ax, 'equal', 'off', 'vis3d');
    view(ax, 3);
    if isempty(findobj(ax, 'Type', 'light'))
        camlight(ax, 'headlight');
        camlight(ax, -80, -10);
    end
    if opts.Colorbar, colorbar(ax); end
    if ~isempty(opts.Title), title(ax, opts.Title, 'Interpreter', 'none'); end
end
