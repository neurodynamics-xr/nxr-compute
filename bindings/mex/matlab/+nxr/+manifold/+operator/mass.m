function M = mass(mctx)
%MASS  V × V mass matrix M (Voronoi default).
%   M = nxr.manifold.operator.mass(mctx)
%
%   Eagerly computed by `nxr.manifold.context(V, F)` and held on the
%   struct. Returning the field directly so this is O(1).
    M = mctx.M;
end
