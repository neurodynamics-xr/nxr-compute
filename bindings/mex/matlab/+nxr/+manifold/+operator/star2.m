function out = star2(mctx)
%STAR2  Hodge ★₂ — inverse face areas as a diagonal sparse [nF x nF].
%   out = nxr.manifold.operator.star2(mctx)
    dec = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('assembleDECOperators', h));
    out = dec.hodge2;
end
