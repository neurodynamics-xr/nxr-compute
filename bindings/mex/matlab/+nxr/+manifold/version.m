function v = version()
%VERSION  nxr-compute version string from the loaded MEX dispatcher.
%
%   Returns a string like "nxr-compute 0.1.0" — the same value
%   `nxr.version()` returns in the JS / WASM bindings.
    v = nxr_compute('version');
end
