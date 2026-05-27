function out = heat(mctx, sources, sourceValues, timesteps, alpha, k)
%HEAT  Spectral heat diffusion over a list of timesteps.
%   out = nxr.manifold.solve.heat(mctx, sources, sourceValues, timesteps)
%   out = nxr.manifold.solve.heat(mctx, sources, sourceValues, timesteps, alpha)
%   out = nxr.manifold.solve.heat(mctx, sources, sourceValues, timesteps, alpha, k)
%
%   sources       source vertex indices (1-based)
%   sourceValues  initial values at those vertices
%   timesteps     vector of t values
%   alpha         diffusion coefficient (default 1.0)
%   k             number of eigenmodes for the spectral evolution
%                 (default min(64, nV-1))
%
%   Returns a [T x nV] time series of the diffused field.
%
%   Unlike the WASM binding — where eigenmodes persist on the context —
%   this stateless functional form runs an internal precompute(k) on a
%   transient handle before the spectral diffusion. Pass k to control
%   accuracy. (nxr.manifold.measure.distance is the geodesic sibling.)
    if nargin < 5 || isempty(alpha), alpha = 1.0; end
    if nargin < 6 || isempty(k),     k = min(64, mctx.nV - 1); end
    h = nxr_compute('create', mctx.V, mctx.F);
    cleanup = onCleanup(@() nxr_compute('destroy', h)); %#ok<NASGU>
    nxr_compute('precompute', h, k);
    out = nxr_compute('heatDiffusion', h, int32(sources), sourceValues, timesteps, alpha);
end
