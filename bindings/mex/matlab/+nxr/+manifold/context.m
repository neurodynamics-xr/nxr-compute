function mctx = context(V, F)
%CONTEXT  Build a manifold context for the nxr.manifold.* surface.
%
%   mctx = nxr.manifold.context(V, F)
%
%   V    nV-by-3 double, vertex positions
%   F    nF-by-3 integer (zero-based), triangle indices
%
%   Returns a struct holding the mesh, the assembled cotangent stiffness
%   `K`, the Voronoi mass `M`, and the per-axis sizes `nV / nE / nF`.
%   Each nxr.manifold.* leaf takes mctx as its first argument, mirroring
%   the JS / WASM / N-API binding's six-group surface:
%
%       mctx = nxr.manifold.context(V, F);
%       eig  = nxr.manifold.solve.eigen(mctx, 6);
%       d    = nxr.manifold.measure.signedDistance(mctx, [0,1,2], false);
%
%   Setup:
%     addpath('bindings/mex/matlab')                   % this package tree
%     addpath('build/bindings/mex/Release')            % the MEX binary
%
%   Build the MEX with `bash scripts/build.sh Release`.
%
%   Lifetime: MATLAB is value-typed, so the struct is copied at every
%   call boundary. The MEX dispatcher itself is stateless — each call
%   rebuilds the C++ ComputeContext internally. Eager assembly here
%   pays the cotangent / Voronoi cost once per `nxr.manifold.context`
%   call; downstream leaves operate on the resulting K and M directly,
%   matching the WASM shim's lazy-cache semantics on first access.
%
%   See also: nxr.manifold.solve.eigen, nxr.manifold.operator.stiffness.

    validateattributes(V, {'double'}, {'2d', 'ncols', 3, 'finite'}, 'context', 'V');
    if ~isa(F, 'int32')
        F = int32(F);
    end
    validateattributes(F, {'int32'}, {'2d', 'ncols', 3}, 'context', 'F');

    ops = nxr_compute('assembleMeshOperators', V, F);

    mctx        = struct();
    mctx.V      = V;
    mctx.F      = F;
    mctx.K      = ops.stiffness;
    mctx.M      = ops.mass;
    mctx.nV     = ops.nV;
    mctx.nE     = ops.nE;
    mctx.nF     = ops.nF;
    mctx.totalArea  = ops.totalArea;
    mctx.vertexAreas = ops.vertexAreas;
    mctx.normals     = ops.normals;
end
