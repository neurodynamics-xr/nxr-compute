function out = extend(mctx, sourceVerts, sourceValues)
%EXTEND  Smoothly extend a sparse scalar to the entire mesh (vector heat).
%   out = nxr.manifold.interpolate.extend(mctx, sourceVerts, sourceValues)
    out = nxr_compute('vectorHeatExtendScalar', mctx.V, mctx.F, ...
                      int32(sourceVerts), sourceValues);
end
