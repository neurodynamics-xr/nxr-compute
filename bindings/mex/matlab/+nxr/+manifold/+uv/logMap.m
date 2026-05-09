function lm = logMap(mctx, sourceVertex, strategy)
%LOGMAP  Logarithmic map at a source vertex via the vector heat method.
%   lm = nxr.manifold.uv.logMap(mctx, sourceVertex)
%   lm = nxr.manifold.uv.logMap(mctx, sourceVertex, strategy)
%
%   strategy: 0=VectorHeat, 1=AffineLocal (default), 2=AffineAdaptive
    if nargin < 3
        strategy = 1;   % AffineLocal
    end
    lm = nxr_compute('vectorHeatLogMap', mctx.V, mctx.F, ...
                     int32(sourceVertex), int32(strategy));
end
