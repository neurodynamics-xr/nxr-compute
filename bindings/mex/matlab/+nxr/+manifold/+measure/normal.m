function n = normal(mctx, type)
%NORMAL  Per-vertex normals — eagerly computed by `nxr.manifold.context`.
%   n = nxr.manifold.measure.normal(mctx)
%   n = nxr.manifold.measure.normal(mctx, type)   % only `type=0` (angle) supported
%
%   Other estimators (area, equal, sphere-inscribed, mean-curvature,
%   gauss-curvature) are not yet wired in the MEX dispatcher's
%   assembleMeshOperators. The cached `mctx.vertexNormals` is the
%   default angle-weighted variant.
    if nargin >= 2 && type ~= 0
        nxr.manifold.impl.notWired('measure.normal (type ~= 0)');
    end
    n = mctx.vertexNormals;
end
