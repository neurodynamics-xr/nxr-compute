function s = vertex(mctx, v) %#ok<INUSL>
%VERTEX  Identity wrapper — single-vertex selection.
%   s = nxr.manifold.query.vertex(mctx, v) → struct('vertexIndex', v)
    s = struct('vertexIndex', v);
end
