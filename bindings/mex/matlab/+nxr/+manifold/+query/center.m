function c = center(mctx, sourceVerts, p)
%CENTER  Karcher mean / surface center of source vertices via vector heat.
%   c = nxr.manifold.query.center(mctx, sourceVerts)
%   c = nxr.manifold.query.center(mctx, sourceVerts, p)   % default p=2
%
%   CAVEAT — connected mesh required. The vector-heat Karcher mean assumes a
%   single connected component. On a disconnected surface (e.g. a both-
%   hemispheres cortex) the log map on components not reached by the sources
%   is non-finite, and the underlying solver raises a checkFinite() / NaN
%   error. Restrict to one connected component, or split the mesh first.
    if nargin < 3
        p = 2;
    end
    c = nxr_compute('findCenter', mctx.V, mctx.F, int32(sourceVerts), p);
end
