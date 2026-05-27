# nxr-compute: stateful MEX binding

> **Superseded.** This rough draft was refined into a full design spec on
> 2026-05-27. See the canonical document:
>
> **[`docs/superpowers/specs/2026-05-27-stateful-mex-binding-design.md`](superpowers/specs/2026-05-27-stateful-mex-binding-design.md)**
>
> Key changes from the original draft after review:
>
> - The `nxr.Manifold` MATLAB handle class is **shelved**. Statefulness is
>   a property of the binding shell (the compiled MEX), not the host-language
>   wrapper — exactly as in WASM, where the state lives in the C++
>   `ContextWrapper` and JS holds only a proxy. MATLAB `.m` wrappers over the
>   handle are an application-side concern.
> - The work is scoped to making `nxr_compute_mex.cpp` stateful and bringing
>   it to **full parity with the WASM `ContextWrapper` surface**.
> - Dispatch uses an **additive handle layer**: `create`/`destroy` + a
>   `uint64`-handle branch in each op, leaving the existing stateless
>   commands (and the `nxr.manifold.*` functional API) untouched.
