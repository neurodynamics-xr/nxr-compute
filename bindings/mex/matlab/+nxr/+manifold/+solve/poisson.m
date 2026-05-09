function phi = poisson(mctx, sourceVerts, sourceValues) %#ok<INUSD>
%POISSON  Solve K φ = -M (ρ - ρ̄) for a sparse source map.
%   nxr.manifold.solve.poisson(mctx, sourceVerts, sourceValues)
%
%   Not yet wired in the MEX dispatcher — the WASM and addon
%   bindings expose this. To enable, add a `cmdSolvePoisson` handler
%   in bindings/mex/src/nxr_compute_mex.cpp.
    nxr.manifold.impl.notWired('solve.poisson');
end
