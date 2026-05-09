function uv = bff(mctx) %#ok<INUSD>
%BFF  Boundary First Flattening (TODO — not in MEX dispatcher).
%
%   To enable, add a `cmdComputeUVCoordinates` handler.
    nxr.manifold.impl.notWired('uv.bff');
end
