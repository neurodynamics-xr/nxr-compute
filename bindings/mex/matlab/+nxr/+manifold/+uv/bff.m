function uv = bff(mctx)
%BFF  Boundary First Flattening — conformal planar UV [nV x 2].
%   uv = nxr.manifold.uv.bff(mctx)
%
%   Requires a mesh with at least one boundary loop; raises a structured
%   nxr:* error on a closed mesh (cut it first).
    uv = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('bff', h));
end
