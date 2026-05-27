function phi = poisson(mctx, sourceVerts, sourceValues)
%POISSON  Solve K φ = -M (ρ - ρ̄) for a sparse source map.
%   phi = nxr.manifold.solve.poisson(mctx, sourceVerts, sourceValues)
%
%   sourceVerts   vertex indices (1-based) carrying density
%   sourceValues  matching density values
%   Returns the per-vertex potential φ [nV x 1].
    phi = nxr.manifold.impl.withHandle(mctx, @(h) ...
        nxr_compute('poisson', h, int32(sourceVerts), sourceValues));
end
