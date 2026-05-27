function out = curvature(mctx)
%CURVATURE  Per-vertex Gaussian / mean / principal curvatures.
%   out = nxr.manifold.measure.curvature(mctx)
%
%   Returns a struct {gaussian, mean, kMin, kMax, principalDirMax}.
%   The integrated Gaussian curvature obeys discrete Gauss-Bonnet
%   (sum(gaussian) = 2πχ).
    out = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('curvatures', h));
end
