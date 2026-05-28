function h = surface(V, F, opts)
%SURFACE  Render a triangle manifold as a lit patch (validation view).
%   h = nxr.viz.surface(V, F)
%   h = nxr.viz.surface(V, F, Name, Value, ...)
%
%   Draws the surface defined by vertices V [nV x 3] and 1-based faces
%   F [nF x 3] as a shaded, lit patch — the minimal "is the mesh sane"
%   view. Two-component surfaces (e.g. both cortical hemispheres) render
%   as a single patch.
%
%   Name-value options:
%     'Parent'     target axes (default: a new figure)
%     'Title'      title string
%     'FaceColor'  RGB face color (default light gray [0.80 0.80 0.82])
%     'EdgeColor'  edge color (default 'none')
%     'Alpha'      face transparency in [0 1] (default 1)
%
%   Returns the patch handle. Base-MATLAB patch backend — this package is
%   for quick validation; production/interactive viz lives in nxr-viewer /
%   nxr-design-system charts-manifold (three.js + WebGPU).
%
%   See also nxr.viz.scalar, nxr.viz.vectorField, nxr.viz.dashboard.
    arguments
        V (:,3) double
        F (:,3) double
        opts.Parent = []
        opts.Title (1,:) char = ''
        opts.FaceColor (1,3) double = [0.80 0.80 0.82]
        opts.EdgeColor = 'none'
        opts.Alpha (1,1) double = 1.0
    end

    ax = resolveAxes(opts.Parent);
    h = patch(ax, 'Faces', F, 'Vertices', V, ...
        'FaceColor', opts.FaceColor, 'EdgeColor', opts.EdgeColor, ...
        'FaceAlpha', opts.Alpha, 'FaceLighting', 'gouraud', ...
        'AmbientStrength', 0.45, 'DiffuseStrength', 0.75, 'SpecularStrength', 0.15);

    axis(ax, 'equal', 'off', 'vis3d');
    view(ax, 3);
    if isempty(findobj(ax, 'Type', 'light'))
        camlight(ax, 'headlight');
        camlight(ax, -80, -10);
    end
    if ~isempty(opts.Title)
        title(ax, opts.Title, 'Interpreter', 'none');
    end
end
