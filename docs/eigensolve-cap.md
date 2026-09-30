# Eigensolve K cap

In the WASM build, `solveEigenmodes` throws `EigensolveInvalidK` when `k > 1000`.

## Why

Spectra's `SymGEigsShiftSolver` allocates a Krylov basis of size
`ncv = max(2k+1, 20)` clamped at `n` (the matrix size). Each Krylov
vector is `n` doubles. So:

| k    | ncv  | basis size on n=10242 (fsaverage5) |
|-----:|-----:|-----------------------------------:|
|   10 |   21 | 1.7 MB                              |
|  100 |  201 | 16 MB                               |
| 1000 | 2001 | 164 MB                              |
| 2000 | 4001 | 328 MB                              |
| 5000 | 10001 ≈ n | **820 MB**                  |

Add to that `n × k × 8` bytes for the eigenvectors return value (e.g.
410 MB at k=5000) plus internal residuals, temporaries, and the
shift-invert factor. WASM's linear-memory cap is 2 GB on 32-bit;
realistic browser builds hit memory pressure well before that.

The bench probe `bench-eigensolve.html` confirmed empirically:

```
fsaverage5 (10 242 V):
  k=10:     75 ms
  k=100:   468 ms
  k=1000:  33.4 s     (warm-ops, IRAM-dominated)
  k=5000:  ERROR after 384 s — heap grew to 1976 MB
```

The k=5000 cell ran for 6.4 minutes, grew the WASM heap to 1976 MB
(near the 2 GB cap), then errored mid-iteration.

## Where the cap applies — the WASM build only (2026-09-29)

The check is compiled under `__EMSCRIPTEN__`. Native consumers (the
N-API addon, the MEX bindings, the CLI) take any `k < n`: they have the
memory headroom, and cortical-flow's Dirac basis asks for 1600 modes
plus a 4-fold guard (1632) — a native solve. Until 2026-09-29 the cap
applied to every build despite this document's intent, and the desktop
app failed at K = 1600 the same way the browser would.

## Lifting the cap

Three paths in roughly increasing effort:

1. **Tighter Krylov basis** — expose Spectra's `ncv` as a parameter
   on `solveEigenmodes`. Callers can pick `1.5k+1` instead of `2k+1`,
   buying ~25 % memory at large k for a small iteration-count cost.
   Buys roughly 25 % more headroom; cap could move to ~k=1300.

2. **memory64 WASM** — emscripten supports `MEMORY64=1` for >4 GB
   heaps. Modern browsers ship this behind flags. Once standard
   support lands, the cap can be removed for browser builds.

3. **Server-side fallback** — run the addon binding in a Node.js
   process or remote endpoint for k > 1000 cases. The addon already
   exists and has no memory cap of its own.

Until one of those lands, the cap stands.
