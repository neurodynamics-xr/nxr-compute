function v = smoothFace(mctx, nSym, alignToCurvature)
%SMOOTHFACEFIELD  Smoothest face-based n-RoSy direction field (Knöppel-Crane).
%   v = nxr.manifold.interpolate.smoothFace(mctx)
%   v = nxr.manifold.interpolate.smoothFace(mctx, nSym)
%   v = nxr.manifold.interpolate.smoothFace(mctx, nSym, alignToCurvature)
%
%   Returns F*3 world-space vectors (principal nSym-RoSy representative).
    if nargin < 2, nSym = 4;             end
    if nargin < 3, alignToCurvature = false; end
    v = nxr_compute('smoothFace', mctx.V, mctx.F, ...
                    int32(nSym), logical(alignToCurvature));
end
