function out = star0(mctx)
%STAR0  Hodge ★₀ — vertex Voronoi areas as a diagonal sparse [nV x nV].
%   out = nxr.manifold.operator.star0(mctx)
    dec = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('assembleDECOperators', h));
    out = dec.hodge0;
end
