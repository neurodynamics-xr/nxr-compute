function cmap = divergingColormap(n)
%DIVERGINGCOLORMAP  Blue-white-red diverging colormap for signed fields.
%   cmap = nxr.viz.divergingColormap(n) returns an n-by-3 colormap that
%   runs blue → light gray → red (a coolwarm-style map), for signed scalar
%   fields shown with a symmetric color range. MATLAB ships no good
%   diverging map by default, hence this helper. Default n = 256.
    if nargin < 1 || isempty(n)
        n = 256;
    end
    anchors = [0.230 0.299 0.754;     % blue (low)
               0.865 0.865 0.865;     % near-white (zero)
               0.706 0.016 0.150];    % red (high)
    x  = [0; 0.5; 1];
    xi = linspace(0, 1, n)';
    cmap = [interp1(x, anchors(:,1), xi), ...
            interp1(x, anchors(:,2), xi), ...
            interp1(x, anchors(:,3), xi)];
    cmap = min(max(cmap, 0), 1);
end
