function d = distance(mctx, sourceVerts) %#ok<INUSD>
%DISTANCE  Heat-method geodesic distance from source vertices (TODO — not in MEX).
%
%   To enable, add a `cmdComputeGeodesicDistance` handler to the MEX
%   dispatcher backed by HeatGeodesicSolver.
%
%   See also: nxr.manifold.measure.signedDistance (which IS wired).
    nxr.manifold.impl.notWired('measure.distance');
end
