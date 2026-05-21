function s = stripe(mctx, vertexFieldRaw, uniformFrequency, connectOnSingularities)
%STRIPE  Sinusoidal stripes aligned to a 2-RoSy field (Knöppel-Crane 2015).
%   s = nxr.manifold.uv.stripe(mctx, vertexFieldRaw, uniformFrequency)
%   s = nxr.manifold.uv.stripe(mctx, vertexFieldRaw, uniformFrequency, connectOnSingularities)
%
%   `vertexFieldRaw` is the V*2 raw nRoSy field returned by
%   `nxr.manifold.interpolate.smoothVertex` in its
%   `vertexFieldRaw` field.
    if nargin < 4
        connectOnSingularities = true;
    end
    s = nxr_compute('compute', mctx.V, mctx.F, ...
                    vertexFieldRaw, uniformFrequency, ...
                    logical(connectOnSingularities));
end
