function eig = precompute(mctx, k)
%PRECOMPUTE  One-shot assemble + solve + normalize + remove-DC bundle.
%
%   eig = nxr.manifold.solve.precompute(mctx, k)
%
%   Equivalent to:
%       e = nxr.manifold.solve.eigen(mctx, k);
%       e.eigenvectors = nxr_compute('normalizeEigenmodes', e.eigenvectors, mctx.M);
%       e = nxr_compute('removeDC', e);
%
%   The MEX `precompute` dispatcher inlines all four steps in C++, so
%   it's faster than calling them piecewise from MATLAB.
    eig = nxr_compute('precompute', mctx.V, mctx.F, k);
end
