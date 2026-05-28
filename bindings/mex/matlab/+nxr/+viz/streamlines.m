function h = streamlines(V, F, faceField, opts)
%STREAMLINES  Trace and draw streamlines of a per-FACE tangent field on a surface.
%   h = nxr.viz.streamlines(V, F, faceField)
%   h = nxr.viz.streamlines(V, F, faceField, Name, Value, ...)
%
%   faceField is [nF x 3] (one tangent vector per face, e.g. a direction
%   field, gradient, or Hodge component). Streamlines are integrated ON the
%   surface: at each step the point is snapped onto a real face plane, so a
%   curve physically cannot leave the mesh (unlike a naive plane-projection
%   tracer, which drifts off a folded surface). This is the clearest way to
%   read a direction field's pattern at any zoom.
%
%   Name-value options:
%     'Parent'    target axes (default: new figure)
%     'Title'     title string
%     'Color'     line color (default blue)
%     'LineWidth' streamline width (default 1.0)
%     'NSeeds'    approx. number of seed streamlines, spatially uniform (200)
%     'StepLen'   integration step in world units (default 0.5*mean edge)
%     'MaxSteps'  max steps per direction from each seed (default 400)
%     'Surface'   draw the gray surface underneath (default true)
%
%   Returns the line handle. For an n-RoSy/cross field, pass the
%   representative faceField — streamlines follow the local direction and so
%   avoid the 90-degree branch-flip that plagues a single-arrow glyph.
%
%   See also nxr.viz.vectorField, nxr.viz.segments.
    arguments
        V (:,3) double
        F (:,3) double
        faceField (:,3) double
        opts.Parent = []
        opts.Title (1,:) char = ''
        opts.Color = [0.05 0.15 0.75]
        opts.LineWidth (1,1) double = 1.0
        opts.NSeeds (1,1) double = 200
        opts.StepLen (1,1) double = 0
        opts.MaxSteps (1,1) double = 400
        opts.Surface (1,1) logical = true
    end
    nF = size(F,1);
    if size(faceField,1) ~= nF
        error('nxr:viz:streamlines:size', 'faceField must be [nF=%d x 3] (got %d)', nF, size(faceField,1));
    end
    v1=V(F(:,1),:); v2=V(F(:,2),:); v3=V(F(:,3),:);
    FN = cross(v2-v1, v3-v1, 2); FN = FN ./ max(vecnorm(FN,2,2), eps);
    Cn = (v1+v2+v3)/3;
    mel = mean(vecnorm([v2-v1; v3-v2; v1-v3], 2, 2));
    step = opts.StepLen; if step <= 0, step = 0.4*mel; end
    mdl = KDTreeSearcher(Cn);                     % nearest-face lookup (robust)

    % spatially-uniform seed faces: one per coarse grid cell (~NSeeds total)
    area = sum(0.5*vecnorm(cross(v2-v1, v3-v1, 2), 2, 2));
    cs = sqrt(area / max(opts.NSeeds,1));
    [~, ia] = unique(floor(Cn/max(cs,eps)), 'rows', 'stable');
    seeds = ia(:)';

    % trace every seed forward + backward, join into one NaN-separated polyline
    segs = cell(1, numel(seeds));
    for k = 1:numel(seeds)
        fwd = traceDir(seeds(k), +1, FN, Cn, mdl, faceField, step, opts.MaxSteps);
        bwd = traceDir(seeds(k), -1, FN, Cn, mdl, faceField, step, opts.MaxSteps);
        poly = [flipud(bwd); fwd(2:end,:)];
        if size(poly,1) >= 2, segs{k} = [poly; nan(1,3)]; end
    end
    P = vertcat(segs{:});
    if isempty(P), P = nan(1,3); end

    ax = resolveAxes(opts.Parent);
    if opts.Surface
        nxr.viz.surface(V, F, 'Parent', ax, 'Alpha', 1.0);
    end
    hold(ax, 'on');
    h = plot3(ax, P(:,1), P(:,2), P(:,3), '-', 'Color', opts.Color, 'LineWidth', opts.LineWidth);
    axis(ax, 'equal', 'off', 'vis3d'); view(ax, 3);
    if isempty(findobj(ax, 'Type', 'light'))
        camlight(ax, 'headlight'); camlight(ax, -80, -10);
    end
    if ~isempty(opts.Title), title(ax, opts.Title, 'Interpreter', 'none'); end
end

function pts = traceDir(fc, sgn, FN, Cn, mdl, fld, step, maxSteps)
%TRACEDIR  Integrate one streamline from face fc's centroid in direction sgn.
%   Each step snaps to the nearest face centroid (KD-tree) and projects onto
%   that face's plane, so the curve stays on the surface and follows the local
%   field. Small steps keep the nearest face local (no cross-fold jumps).
    p = Cn(fc, :);
    pts = zeros(maxSteps+1, 3); pts(1,:) = p; n = 1;
    for s = 1:maxSteps
        d = fld(fc, :);
        d = d - (d*FN(fc,:)')*FN(fc,:);          % project field to face plane
        nd = norm(d); if nd < 1e-12, break; end
        d = sgn * d / nd;
        pnew = p + d*step;
        fc = knnsearch(mdl, pnew);               % nearest face = local face
        p  = pnew - ((pnew - Cn(fc,:))*FN(fc,:)')*FN(fc,:);   % snap onto its plane
        n = n + 1; pts(n,:) = p;
    end
    pts = pts(1:n, :);
end
