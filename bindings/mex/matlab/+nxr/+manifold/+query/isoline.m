function out = isoline(mctx, field, level) %#ok<INUSD>
%ISOLINE  Single-level isoline of a per-vertex scalar field (TODO — not in MEX).
%
%   To enable, add a `cmdComputeIsolines` handler to the MEX dispatcher.
    nxr.manifold.impl.notWired('query.isoline');
end
