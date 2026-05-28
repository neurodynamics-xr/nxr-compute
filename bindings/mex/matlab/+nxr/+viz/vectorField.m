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
%   nxr already sharps 1-forms to face-centred tangent vectors (Whitney
%   interpolation, V(f) = N×(a+b+c)/6A), so face vectors sit at the centroid
%   and lie in the tangent plane; this routine just anchors and scales them.
%
%   Two glyph styles:
%     • default       — arrow length ∝ magnitude (95th-percentile-normalised,
%                       tail clamped). Good when magnitude matters and is fairly
%                       uniform.
%     • Normalize+     — all glyphs the same length (direction only) with
%       ColorByMag       magnitude shown by color + a colorbar. Best for
%                        direction fields with a heavy-tailed magnitude (Hodge
%                        components, gradient) — no giant/invisible arrows.
%
%   Name-value options:
%     'Parent'      target axes (default: new figure)
%     'Title'       title string
%     'Scale'       glyph length multiplier (default 1; length unit ≈ 2 edges)
%     'Color'       arrow color when not coloring by magnitude (default red)
%     'Surface'     draw the gray surface underneath (default true)
%     'NMax'        subsample to at most this many arrows (default 4000)
%     'MinMag'      drop arrows below this fraction of the 95th-percentile
%                   magnitude (default 0 = draw all). Suppresses noise-direction
%                   arrows on near-zero faces (e.g. a gradient on flat patches).
%     'Normalize'   draw every glyph at the same length (default false)
%     'ColorByMag'  color glyphs by magnitude via a colormap (default false)
%     'Colormap'    colormap name for ColorByMag (default 'parula')
%     'Colorbar'    show the magnitude colorbar for ColorByMag (default true)
%     'NBins'       number of color bins for ColorByMag (default 16)
%
%   Returns a representative graphics handle (its ancestor axes is the target).
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
        opts.Normalize (1,1) logical = false
        opts.ColorByMag (1,1) logical = false
        opts.Colormap (1,:) char = 'parula'
        opts.Colorbar (1,1) logical = true
        opts.NBins (1,1) double = 16
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
        % Spatially-uniform subsample: one anchor per coarse grid cell. Mesh
        % index order is NOT spatial, so linear subsampling scatters arrows at
        % random faces and destroys any visible field pattern. Cell sized from
        % the surface area to yield ~NMax samples.
        e21 = V(F(:,2),:)-V(F(:,1),:); e31 = V(F(:,3),:)-V(F(:,1),:);
        area = sum(0.5*vecnorm(cross(e21,e31,2),2,2));
        cell = sqrt(area / opts.NMax);
        [~, ia] = unique(floor(P/max(cell,eps)), 'rows', 'stable');
        idx = ia(:)';
    end
    L    = opts.Scale * 2.0 * meanEdgeLength(V, F);
    P0   = P(idx, :);
    vv   = vecs(idx, :);
    mags = vecnorm(vv, 2, 2);
    sm   = sort(mags);
    ref  = sm(max(1, round(0.95 * numel(sm))));
    if ref == 0, ref = max(mags); end
    if ref == 0, ref = 1; end

    keep = true(size(mags));
    if opts.MinMag > 0
        keep = mags >= opts.MinMag * ref;    % drop noise-direction arrows
    end

    if opts.Normalize
        % uniform-length glyphs (direction only); magnitude carried by color
        g = L * (vv ./ max(mags, eps));
    else
        % length ∝ magnitude: percentile-normalise (95th ≈ L), clamp the tail
        % so a heavy-tailed field doesn't shoot a few giant arrows off the view.
        g  = (L / ref) * vv;
        gm = vecnorm(g, 2, 2); over = gm > 2.0 * L;
        g(over,:) = g(over,:) .* (2.0 * L ./ gm(over));
    end
    g(~keep, :) = NaN;                        % quiver3 skips NaN rows
    B0 = P0 - 0.5*g;                          % center the glyph on its anchor

    if opts.ColorByMag
        h = drawColored(ax, B0, g, mags, keep, opts);
    else
        h = quiver3(ax, B0(:,1), B0(:,2), B0(:,3), g(:,1), g(:,2), g(:,3), 0, ...
                    'Color', opts.Color, 'LineWidth', 0.6, 'MaxHeadSize', 0.5);
    end

    axis(ax, 'equal', 'off', 'vis3d'); view(ax, 3);
    if isempty(findobj(ax, 'Type', 'light'))
        camlight(ax, 'headlight'); camlight(ax, -80, -10);
    end
    if ~isempty(opts.Title), title(ax, opts.Title, 'Interpreter', 'none'); end
end

function h = drawColored(ax, P0, g, mags, keep, opts)
%DRAWCOLORED  Draw the kept glyphs in magnitude bins, one quiver3 per bin,
%   colored from the colormap, with a matching colorbar.
    drawable = keep & all(isfinite(g), 2);
    if ~any(drawable), h = ax; return; end
    lo = min(mags(drawable)); hi = max(mags(drawable));
    % A unit-magnitude direction field (trivial connection, NRoSy) carries no
    % magnitude information — coloring it by magnitude yields a meaningless
    % near-zero colorbar. Fall back to a single color (keeps the field's color
    % identity) when the relative spread is negligible.
    if (hi - lo) <= 1e-2 * max(hi, eps)
        h = quiver3(ax, P0(drawable,1), P0(drawable,2), P0(drawable,3), ...
            g(drawable,1), g(drawable,2), g(drawable,3), 0, ...
            'Color', opts.Color, 'LineWidth', 0.7, 'MaxHeadSize', 0.6);
        return;
    end
    map   = feval(opts.Colormap, opts.NBins);
    edges = linspace(lo, hi, opts.NBins + 1);
    b = discretize(mags, edges);
    b(mags >= hi) = opts.NBins;               % include the top edge

    hold(ax, 'on');
    hs = gobjects(0);
    for bb = 1:opts.NBins
        sel = drawable & (b == bb);
        if ~any(sel), continue; end
        hs(end+1) = quiver3(ax, P0(sel,1), P0(sel,2), P0(sel,3), ...
            g(sel,1), g(sel,2), g(sel,3), 0, ...
            'Color', map(bb,:), 'LineWidth', 0.7, 'MaxHeadSize', 0.6); %#ok<AGROW>
    end
    if isempty(hs), h = ax; else, h = hs(end); end
    colormap(ax, map); clim(ax, [lo hi]);
    if opts.Colorbar
        cb = colorbar(ax); cb.Label.String = 'magnitude';
    end
end
