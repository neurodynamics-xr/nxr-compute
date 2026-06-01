function out = star0(mctx)
%STAR0  Hodge ★₀ — diag(vertexDualAreas) as a sparse [nV x nV] matrix.
%   Each diagonal entry is the sum of A_T/3 over incident triangles
%   (sourced from geometry-central's vertexDualAreas). Same numerical
%   content as the lumped mass matrix.
%   out = nxr.manifold.operator.star0(mctx)
    dec = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('assembleDECOperators', h));
    out = dec.hodge0;
end
