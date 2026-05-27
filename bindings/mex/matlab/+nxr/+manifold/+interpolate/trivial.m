function out = trivial(mctx, singVerts, singValues)
%TRIVIAL  Trivial-connection direction field (prescribed singularities).
%   out = nxr.manifold.interpolate.trivial(mctx, singVerts, singValues)
%
%   singVerts   singularity vertex indices (1-based)
%   singValues  singularity indices; must sum to the mesh Euler
%               characteristic (Gauss-Bonnet constraint).
%   Returns a struct {connections, directionVectors [nF x 3],
%   orthogonalVectors [nF x 3], eulerCharacteristic, gaussBonnetSatisfied}.
%
%   Mirrors the WASM `interpolate.trivial`. (Distinct from the smoothest
%   NRoSy fields in interpolate.smoothFace / interpolate.smoothVertex.)
    out = nxr.manifold.impl.withHandle(mctx, @(h) ...
        nxr_compute('trivial', h, int32(singVerts), singValues));
end
