function notWired(name)
%NOTWIRED  Throw a stable MException for not-yet-wired MEX leaves.
%
%   Used by every nxr.manifold.* leaf whose underlying MEX command
%   doesn't exist yet in bindings/mex/src/nxr_compute_mex.cpp. Mirrors
%   the JS shim's `NOT_WIRED_IN_ADDON` error code so consumers can
%   pattern-match the same way across all bindings.
%
%   The way to "wire" a leaf is to add a `cmdXxx` handler in the MEX
%   dispatcher and a corresponding `else if (cmd == "...")` line in
%   mexFunction(). Once that's done, replace the body of the .m leaf
%   with the actual `nxr_compute('...', ...)` call.

    err = MException( ...
        'nxr:notWiredInMex', ...
        ['[NOT_WIRED_IN_MEX] %s is exposed by the WASM binding but ' ...
         'not yet by the MEX dispatcher. Add it to ' ...
         'bindings/mex/src/nxr_compute_mex.cpp to enable.'], ...
        name);
    throw(err);
end
