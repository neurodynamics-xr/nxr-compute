function n = normal(mctx, type)
%NORMAL  Per-vertex normals by estimator.
%   n = nxr.manifold.measure.normal(mctx)         % 'angle' (default, cached)
%   n = nxr.manifold.measure.normal(mctx, type)   % estimator name
%
%   type is one of 'angle' (default) | 'area' | 'equal' | 'sphere' |
%   'mean' | 'gauss'. Omitting type (or the legacy numeric 0) returns the
%   cached angle-weighted normals from nxr.manifold.context; any other
%   estimator is computed on demand. Returns [nV x 3].
    if nargin < 2 || isempty(type) || (isnumeric(type) && isequal(type, 0))
        n = mctx.vertexNormals;   % cached angle-weighted normals from context()
        return;
    end
    if isnumeric(type)
        error('nxr:invalidInput', ...
            ['numeric normal type %g is not supported; pass a name string ' ...
             '(angle|area|equal|sphere|mean|gauss)'], type);
    end
    n = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('normals', h, type));
end
