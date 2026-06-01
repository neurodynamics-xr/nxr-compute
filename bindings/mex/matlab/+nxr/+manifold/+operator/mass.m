function M = mass(mctx)
%MASS  V × V mass matrix M (lumped default).
%   M = nxr.manifold.operator.mass(mctx)
%
%   Diagonal lumped mass — sourced from geometry-central's
%   vertexLumpedMassMatrix (== diag(A/3 per vertex)). Eagerly computed
%   by `nxr.manifold.context(V, F)` and held on the struct; returning
%   the field directly so this is O(1).
    M = mctx.M;
end
