function h = vectorField(V, F, vecs, opts)
%VECTORFIELD  Draw a vertex- or face-based vector field as arrows on the surface.
%   h = nxr.viz.vectorField(V, F, vecs)
%   h = nxr.viz.vectorField(V, F, vecs, Name, Value, ...)
%
%   `vecs` is auto-detected by row count:
%     • [nV x 3]  → per-vertex vectors, anchored at vertices (e.g. normals,
%                   parallel transport).
%     • [nF x 3]  → per-face vectors, anchored at face centroids (e.g.
%                   gradient, DEC / Whitney vectors, direction fields).
%   Arrows are scaled so the longest is ~2 mean edge lengths (relative
%   magnitudes preserved), and the surface is drawn semi-transparent
%   underneath for context.
%
%   Name-value options:
%     'Parent'   target axes (default: new figure)
%     'Title'    title string
%     'Scale'    relative arrow length multiplier (default 1)
%     'Color'    arrow color (default red)
%     'Surface'  draw the gray surface underneath (default true)
%     'NMax'     subsample to at most this many arrows (default 4000)
%
%   Returns the quiver3 handle.
%
%   See also nxr.viz.surface, nxr.viz.scalar, nxr.viz.segments.
    arguments
        V (:,3) double
        F (:,3) double
        vecs (:,3) double
        opts.Parent = []
        opts.Title (1,:) char = ''
        opts.Scale (1,1) double = 1.0
        opts.Color = [0.85 0.10 0.10]
        opts.Surface (1,1) logical = true
        opts.NMax (1,1) double = 4000
    end
    nV = size(V,1); nF = size(F,1); n = size(vecs,1);
    if n == nV
        P = V;
    elseif n == nF
        P = (V(F(:,1),:) + V(F(:,2),:) + V(F(:,3),:)) / 3;   % face centroids
    else
        error('nxr:viz:vectorField:size', ...
            'vecs must have nV=%d (vertex) or nF=%d (face) rows (got %d)', nV, nF, n);
    end

    ax = resolveAxes(opts.Parent);
    if opts.Surface
        nxr.viz.surface(V, F, 'Parent', ax, 'Alpha', 0.5);
    end

    idx = 1:n;
    if n > opts.NMax
        idx = round(linspace(1, n, opts.NMax));
    end
    L  = opts.Scale * 2.0 * meanEdgeLength(V, F);   % target longest-arrow length
    mx = max(vecnorm(vecs(idx,:), 2, 2));
    if mx == 0, mx = 1; end
    s = L / mx;

    h = quiver3(ax, P(idx,1), P(idx,2), P(idx,3), ...
                s*vecs(idx,1), s*vecs(idx,2), s*vecs(idx,3), 0, ...
                'Color', opts.Color, 'LineWidth', 0.6, 'MaxHeadSize', 0.5);

    axis(ax, 'equal', 'off', 'vis3d'); view(ax, 3);
    if isempty(findobj(ax, 'Type', 'light'))
        camlight(ax, 'headlight'); camlight(ax, -80, -10);
    end
    if ~isempty(opts.Title), title(ax, opts.Title, 'Interpreter', 'none'); end
end
