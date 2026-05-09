function out = stubMarker(name)
%STUBMARKER  One-shot warning + TODO marker for round-1 stub leaves.
%
%   Mirrors `stubWarn` in the WASM/JS shim. Used for query/measure
%   leaves that are placeholders in every binding (line, circle,
%   region, area, density) — no MEX command is missing here, the
%   primitive simply hasn't been designed yet across the stack.

    persistent warned
    if isempty(warned)
        warned = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    end
    if ~isKey(warned, name)
        warning('nxr:stub', ...
            '[nxr.manifold.%s] is a round-1 stub — returns a TODO marker. Wire when needed.', name);
        warned(name) = true;
    end
    out = struct('method', 'todo', 'name', name);
end
