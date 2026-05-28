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
%     'MinMag'   drop arrows whose magnitude is below this fraction of the
%                95th-percentile magnitude (default 0 = draw all). Useful for
%                fields with near-zero regions (e.g. a gradient on flat
%                patches) where the residual direction is numerical noise and
%                a drawn arrow would point in an arbitrary, off-tangent way.
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
        opts.MinMag (1,1) double = 0.0
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
    % Scale so a 95th-percentile-magnitude arrow is ~2 mean edge lengths.
    % Using a robust percentile (not the max) keeps heterogeneous fields
    % (gradient, Hodge components) visible instead of letting a few outlier
    % faces shrink every other arrow. Uniform fields (normals, n-RoSy) are
    % unaffected (their percentile ≈ max). The few magnitudes above the
    % percentile are then clamped to a ceiling so a heavy tail (e.g. a
    % handful of high-gradient faces) shows as a long-but-bounded arrow
    % instead of a giant streak shooting off across the view.
    L    = opts.Scale * 2.0 * meanEdgeLength(V, F);
    vv   = vecs(idx,:);
    mags = vecnorm(vv, 2, 2);
    sm   = sort(mags);
    ref  = sm(max(1, round(0.95 * numel(sm))));
    if ref == 0, ref = max(mags); end
    if ref == 0, ref = 1; end
    sv   = (L / ref) * vv;                       % percentile-normalised: 95th ≈ L
    cap  = 2.0 * L;                              % ceiling on drawn arrow length
    smag = vecnorm(sv, 2, 2);
    over = smag > cap;
    sv(over,:) = sv(over,:) .* (cap ./ smag(over));
    if opts.MinMag > 0                           % suppress noise-direction arrows
        sv(mags < opts.MinMag * ref, :) = NaN;   % quiver3 skips NaN rows
    end

    h = quiver3(ax, P(idx,1), P(idx,2), P(idx,3), ...
                sv(:,1), sv(:,2), sv(:,3), 0, ...
                'Color', opts.Color, 'LineWidth', 0.6, 'MaxHeadSize', 0.5);

    axis(ax, 'equal', 'off', 'vis3d'); view(ax, 3);
    if isempty(findobj(ax, 'Type', 'light'))
        camlight(ax, 'headlight'); camlight(ax, -80, -10);
    end
    if ~isempty(opts.Title), title(ax, opts.Title, 'Interpreter', 'none'); end
end
