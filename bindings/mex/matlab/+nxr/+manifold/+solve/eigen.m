function eig = eigen(mctx, k, sigma)
%EIGEN  Generalised eigenproblem K φ = λ M φ for the k smallest modes.
%
%   eig = nxr.manifold.solve.eigen(mctx, k)
%   eig = nxr.manifold.solve.eigen(mctx, k, sigma)
%
%   Returns a struct with fields `eigenvectors` (nV × k), `eigenvalues`
%   (k × 1, sorted ascending), and `nV`. The default sigma is -1e-8
%   (shift-invert centred just below zero), targeting the smoothest
%   modes.
    if nargin < 3
        sigma = -1e-8;
    end
    eig = nxr_compute('solve', mctx.K, mctx.M, k, sigma);
end
