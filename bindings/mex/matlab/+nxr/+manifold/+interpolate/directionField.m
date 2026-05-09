function out = directionField(mctx, singVerts, singValues) %#ok<INUSD>
%DIRECTIONFIELD  Trivial-connection direction field (TODO — not in MEX).
%
%   To enable, add a `cmdComputeDirectionField` handler in the MEX
%   dispatcher.
    nxr.manifold.impl.notWired('interpolate.directionField');
end
