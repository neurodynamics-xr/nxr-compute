function out = connectionLaplacian(mctx, options)
%CONNECTIONLAPLACIAN  Connection Laplacian on vertex / face / edge.
%
%   out = nxr.manifold.operator.connectionLaplacian(mctx)
%   out = nxr.manifold.operator.connectionLaplacian(mctx, options)
%
%   options is a struct with optional fields:
%     domain          'vertex' (default) | 'face' | 'edge'
%     nSym            1 (default) | 2 | 4 | …
%     regularization  default 1e-8
%     format          'real2N' (default) | 'complex'
%
%   Returns a struct with K_real (sparse; plus K_imag for the complex
%   format), baseDim, outputDim, nSym, regularization, domain, format.
    if nargin < 2 || isempty(options), options = struct(); end
    out = nxr.manifold.impl.withHandle(mctx, @(h) ...
        nxr_compute('assembleConnectionLaplacian', h, options));
end
