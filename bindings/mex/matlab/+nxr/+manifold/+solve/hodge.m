function h = hodge(mctx, omega)
%HODGE  Hodge / Helmholtz decomposition of a 1-form ω = dα + δβ + γ.
%   h = nxr.manifold.solve.hodge(mctx, omega)
%
%   omega  the input 1-form on edges [nE x 1].
%   Returns a struct with exactPotential (α), coExactPotentialV (β),
%   dAlpha, deltaBeta, gamma (each [nE x 1]) and the Whitney-interpolated
%   face vector fields. By construction dAlpha + deltaBeta + gamma = omega.
    h = nxr.manifold.impl.withHandle(mctx, @(hCtx) nxr_compute('hodge', hCtx, omega));
end
