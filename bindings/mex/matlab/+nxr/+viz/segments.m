function h = segments(positions, opts)
%SEGMENTS  Draw [2N x 3] endpoint-pair line segments.
%   h = nxr.viz.segments(positions)
%   h = nxr.viz.segments(positions, Name, Value, ...)
%
%   `positions` is the [2N x 3] "endpoint pairs" layout produced by the
%   line-type nxr outputs — isolines, stripe patterns, streamlines, and
%   geodesic paths (rows 2k-1 and 2k are the ends of segment k). Drawn as a
%   single NaN-separated polyline for efficiency.
%
%   Name-value options:
%     'Parent'     target axes (default: new figure)
%     'Title'      title string
%     'Color'      line color (default blue)
%     'LineWidth'  default 1
%     'Surface'    {V, F} cell to draw the surface underneath (default none)
%
%   Returns the line handle.
%
%   See also nxr.viz.surface, nxr.viz.vectorField.
    arguments
        positions (:,3) double
        opts.Parent = []
        opts.Title (1,:) char = ''
        opts.Color = [0.10 0.10 0.85]
        opts.LineWidth (1,1) double = 1.0
        opts.Surface cell = {}
    end
    if mod(size(positions,1), 2) ~= 0
        error('nxr:viz:segments:size', ...
            'positions must have an even number of rows ([2N x 3] endpoint pairs)');
    end

    ax = resolveAxes(opts.Parent);
    if numel(opts.Surface) == 2
        nxr.viz.surface(opts.Surface{1}, opts.Surface{2}, 'Parent', ax, 'Alpha', 0.4);
    end

    N = size(positions,1) / 2;
    X = [positions(1:2:end,1)'; positions(2:2:end,1)'; nan(1,N)];
    Y = [positions(1:2:end,2)'; positions(2:2:end,2)'; nan(1,N)];
    Z = [positions(1:2:end,3)'; positions(2:2:end,3)'; nan(1,N)];
    h = plot3(ax, X(:), Y(:), Z(:), '-', 'Color', opts.Color, 'LineWidth', opts.LineWidth);

    axis(ax, 'equal', 'vis3d'); axis(ax, 'off'); view(ax, 3);
    if ~isempty(opts.Title), title(ax, opts.Title, 'Interpreter', 'none'); end
end
