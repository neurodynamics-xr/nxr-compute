function r = smoothVertexField(mctx, nSym, alignToCurvature)
%SMOOTHVERTEXFIELD  Smoothest vertex-based n-RoSy direction field.
%   r = nxr.manifold.interpolate.smoothVertexField(mctx)
%   r = nxr.manifold.interpolate.smoothVertexField(mctx, nSym)
%   r = nxr.manifold.interpolate.smoothVertexField(mctx, nSym, alignToCurvature)
%
%   Returns a struct with `vertexVectors` (V*3 world-space) and
%   `vertexFieldRaw` (V*2 raw nRoSy field — pass to
%   `nxr.manifold.uv.stripe` as-is).
    if nargin < 2, nSym = 2;             end
    if nargin < 3, alignToCurvature = false; end
    r = nxr_compute('computeSmoothVertexField', mctx.V, mctx.F, ...
                    int32(nSym), logical(alignToCurvature));
end
