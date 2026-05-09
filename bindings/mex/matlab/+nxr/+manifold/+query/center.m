function c = center(mctx, sourceVerts, p)
%CENTER  Karcher mean / surface center of source vertices via vector heat.
%   c = nxr.manifold.query.center(mctx, sourceVerts)
%   c = nxr.manifold.query.center(mctx, sourceVerts, p)   % default p=2
    if nargin < 3
        p = 2;
    end
    c = nxr_compute('vectorHeatFindCenter', mctx.V, mctx.F, int32(sourceVerts), p);
end
