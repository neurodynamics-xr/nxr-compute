function out = connectionLaplacian(mctx, options) %#ok<INUSD>
%CONNECTIONLAPLACIAN  Connection Laplacian on vertex / face / edge.
%
%   out = nxr.manifold.operator.connectionLaplacian(mctx, options)
%
%   options is a struct with optional fields:
%     domain          'vertex' (default) | 'face' | 'edge'
%     nSym            1 (default) | 2 | 4 | …
%     regularization  default 1e-8
%     format          'real2N' (default) | 'complex'
%
%   Not yet wired in the MEX dispatcher. The C++ free function
%   `assembleConnectionLaplacian` exists in `include/nxr/compute.h`
%   and is exposed by both WASM and the N-API addon. To enable in
%   MEX, add a `cmdAssembleConnectionLaplacian` handler in
%   bindings/mex/src/nxr_compute_mex.cpp that:
%     1. parses the options struct from a MATLAB struct
%     2. calls nxr::compute::assembleConnectionLaplacian(...)
%     3. returns a struct with K (sparse, native MATLAB),
%        baseDim, outputDim, domain, nSym, regularization, format
%   Then replace this body with `nxr_compute('assembleConnectionLaplacian', mctx.V, mctx.F, options)`.
    nxr.manifold.impl.notWired('operator.connectionLaplacian');
end
