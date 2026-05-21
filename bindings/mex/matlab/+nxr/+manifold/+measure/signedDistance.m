function d = signedDistance(mctx, curveVerts, isLoop, levelSet)
%SIGNEDDISTANCE  Signed geodesic distance from a curve (Feng & Crane 2024).
%
%   d = nxr.manifold.measure.signedDistance(mctx, curveVerts)
%   d = nxr.manifold.measure.signedDistance(mctx, curveVerts, isLoop)
%   d = nxr.manifold.measure.signedDistance(mctx, curveVerts, isLoop, levelSet)
%
%   curveVerts  ordered vertex indices defining the curve polyline
%   isLoop      default true (last → first edge included)
%   levelSet    0=None, 1=ZeroSet (default), 2=Multiple
%
%   The JS / WASM bindings expose this as `measure.distance.signed(...)`
%   — MATLAB doesn't support method-attached function dispatch, so the
%   sub-leaf becomes a sibling here.
    if nargin < 3, isLoop   = true;  end
    if nargin < 4, levelSet = 1;     end   % ZeroSet
    d = nxr_compute('signedHeat', mctx.V, mctx.F, ...
                    int32(curveVerts), logical(isLoop), int32(levelSet));
end
