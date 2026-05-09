function K = stiffness(mctx)
%STIFFNESS  V × V cotangent stiffness K (PSD, symmetrised).
%   K = nxr.manifold.operator.stiffness(mctx)
%
%   Eagerly computed by `nxr.manifold.context(V, F)` and held on the
%   struct. Returning the field directly so this is O(1).
    K = mctx.K;
end
