function d = distance(mctx, sourceVerts)
%DISTANCE  Heat-method geodesic distance from source vertices.
%   d = nxr.manifold.measure.distance(mctx, sourceVerts)
%
%   sourceVerts  source vertex indices (1-based). Returns [nV x 1]
%   geodesic distances (0 at the sources).
%
%   See also nxr.manifold.measure.signedDistance (signed, from a curve).
    d = nxr.manifold.impl.withHandle(mctx, @(h) ...
        nxr_compute('heat', h, int32(sourceVerts)));
end
