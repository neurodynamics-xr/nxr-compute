function ax = resolveAxes(parent)
%RESOLVEAXES  Resolve a target axes for an nxr.viz primitive.
%   ax = resolveAxes(parent) returns `parent` if it is a usable graphics
%   container (axes / tiledlayout tile), or a fresh axes in a new white
%   figure when `parent` is empty. Leaves the axes in `hold on` so callers
%   can overlay. Private helper for the +nxr/+viz package.
    if isempty(parent)
        ax = axes('Parent', figure('Color', 'w'));
    else
        ax = parent;
    end
    hold(ax, 'on');
end
