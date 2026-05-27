function out = withHandle(mctx, fn)
%WITHHANDLE  Run fn(handle) against a transient stateful MEX context.
%
%   out = nxr.manifold.impl.withHandle(mctx, fn)
%
%   Creates a uint64 context handle from mctx.V / mctx.F via the stateful
%   MEX dispatcher (`nxr_compute('create', ...)`), invokes fn(handle),
%   and destroys the handle afterwards — even if fn errors (onCleanup).
%   Returns whatever fn returns.
%
%   Used by the functional `nxr.manifold.*` leaves whose underlying MEX
%   command is handle-only (poisson, hodge, the DEC operators,
%   connectionLaplacian, curvatures, frames, normals, isoline, bff,
%   trivial, geodesic heat). The transient create/destroy preserves the
%   functional API's stateless, rebuild-per-call semantics; the handle is
%   the MATLAB analogue of WASM's `new Module.Manifold(...)` proxy.
%
%   Example:
%       phi = nxr.manifold.impl.withHandle(mctx, @(h) ...
%           nxr_compute('poisson', h, int32(sourceVerts), sourceValues));
%
%   See also: nxr.manifold.context.
    h = nxr_compute('create', mctx.V, mctx.F);
    cleanup = onCleanup(@() nxr_compute('destroy', h)); %#ok<NASGU>
    out = fn(h);
end
