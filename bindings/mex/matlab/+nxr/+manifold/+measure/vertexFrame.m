function out = vertexFrame(mctx)
%VERTEXFRAME  Per-vertex orthonormal tangent frame (the connection-Laplacian gauge).
%   out = nxr.manifold.measure.vertexFrame(mctx)
%
%   Returns a struct {e1, e2, normals}, each [nV x 3] — the 3D realization of the
%   per-vertex tangent space in which the vertex connection-Laplacian's complex
%   eigenvector coordinates are expressed: a coordinate z=(a,b) at vertex i is the
%   3D tangent vector a*e1(i,:) + b*e2(i,:). e1 ⟂ e2, both unit, e2 = n × e1.
%
%   See also: nxr.manifold.measure.frame (the per-FACE frame).
    out = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('vertexFrames', h));
end
