function h = show(M, fieldpath, varargin)
%SHOW  Visualize an M.<fieldpath> result, auto-choosing the right primitive.
%   nxr.viz.show(M, 'measure.curvature.gaussian')   % scalar field
%   nxr.viz.show(M, 'measure.distance')             % scalar field
%   nxr.viz.show(M, 'solve.eigen', 6)               % eigenvector column 6
%   nxr.viz.show(M, 'interpolate.smoothFace')       % face vectors
%   nxr.viz.show(M, 'query.isoline')                % line segments
%
%   M is the struct produced by explore_cortex. The dotted field path is
%   walked into M, then dispatched by the value's shape:
%     [nV x 1] vector            -> nxr.viz.scalar
%     [nV x 3] / [nF x 3] matrix -> nxr.viz.vectorField
%     [2N x 3] matrix            -> nxr.viz.segments
%     eigen struct (+ mode idx)  -> scalar of that eigenvector
%     {positions,...} struct     -> segments
%   Trailing Name,Value args are forwarded to the chosen primitive.
%
%   See also nxr.viz.scalar, nxr.viz.vectorField, nxr.viz.segments, nxr.viz.dashboard.
    V = M.vertices; F = M.faces; nV = M.nV; nF = M.nF;

    val = M;
    parts = strsplit(fieldpath, '.');
    for i = 1:numel(parts)
        if ~isstruct(val) || ~isfield(val, parts{i})
            error('nxr:viz:show:field', 'M.%s: no field "%s"', fieldpath, parts{i});
        end
        val = val.(parts{i});
    end

    if isstruct(val)
        if isfield(val, 'eigenvectors')                 % solve.eigen / precompute
            k = 2; extra = varargin;
            if ~isempty(extra) && isnumeric(extra{1}) && isscalar(extra{1})
                k = extra{1}; extra = extra(2:end);
            end
            h = nxr.viz.scalar(V, F, val.eigenvectors(:,k), ...
                'Title', sprintf('%s mode %d', fieldpath, k), extra{:});
        elseif isfield(val, 'positions')                % isoline / stripe / streamline
            h = nxr.viz.segments(val.positions, 'Surface', {V, F}, 'Title', fieldpath, varargin{:});
        else
            error('nxr:viz:show:struct', ...
                'M.%s is a struct — give a numeric subfield, e.g. %s.<field>', fieldpath, fieldpath);
        end
        return;
    end

    if isvector(val) && numel(val) == nV
        h = nxr.viz.scalar(V, F, val(:), 'Title', fieldpath, varargin{:});
    elseif ismatrix(val) && size(val,2) == 3 && (size(val,1) == nV || size(val,1) == nF)
        h = nxr.viz.vectorField(V, F, val, 'Title', fieldpath, varargin{:});
    elseif ismatrix(val) && size(val,2) == 3 && mod(size(val,1), 2) == 0
        h = nxr.viz.segments(val, 'Surface', {V, F}, 'Title', fieldpath, varargin{:});
    else
        error('nxr:viz:show:shape', ...
            'M.%s has shape [%s] — no matching primitive', fieldpath, num2str(size(val)));
    end
end
