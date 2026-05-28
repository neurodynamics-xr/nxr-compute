function h = spectrum(eigenvalues, opts)
%SPECTRUM  Plot an eigenvalue spectrum, optionally against a reference.
%   h = nxr.viz.spectrum(eigenvalues)
%   h = nxr.viz.spectrum(eigenvalues, 'Reference', refValues, ...)
%
%   A 2-D analysis plot (not a surface map): eigenvalue vs mode index, with
%   an optional reference series (e.g. M.reference.eigenvalues) overlaid.
%
%   Name-value options:
%     'Parent'     target axes (default: new figure)
%     'Reference'  reference eigenvalues to overlay (default none)
%     'Title'      default 'eigenvalue spectrum'
%
%   Returns the line handle for the nxr series.
    arguments
        eigenvalues (:,1) double
        opts.Parent = []
        opts.Reference (:,1) double = []
        opts.Title (1,:) char = 'eigenvalue spectrum'
    end
    if isempty(opts.Parent)
        ax = axes('Parent', figure('Color', 'w'));
    else
        ax = opts.Parent;
    end
    hold(ax, 'on');
    h = plot(ax, eigenvalues, 'o-', 'MarkerSize', 4, 'DisplayName', 'nxr');
    if ~isempty(opts.Reference)
        plot(ax, opts.Reference, 's--', 'MarkerSize', 4, 'DisplayName', 'reference');
        legend(ax, 'Location', 'northwest');
    end
    grid(ax, 'on');
    xlabel(ax, 'mode index');
    ylabel(ax, '\lambda');
    title(ax, opts.Title, 'Interpreter', 'none');
end
