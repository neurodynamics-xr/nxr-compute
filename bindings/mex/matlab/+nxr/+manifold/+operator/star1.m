function out = star1(mctx)
%STAR1  Hodge ★₁ — edge dual/primal length ratios, diagonal sparse [nE x nE].
%   out = nxr.manifold.operator.star1(mctx)
    dec = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('assembleDECOperators', h));
    out = dec.hodge1;
end
