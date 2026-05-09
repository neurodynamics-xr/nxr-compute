function out = d0(mctx) %#ok<INUSD>
%D0  DEC vertex→edge derivative (TODO — assembleDECOperators not exposed by MEX dispatcher).
%
%   To enable, add a `cmdAssembleDECOperators` handler in the MEX
%   dispatcher and a corresponding lazy field on the
%   nxr.manifold.context return struct.
    nxr.manifold.impl.notWired('operator.d0');
end
