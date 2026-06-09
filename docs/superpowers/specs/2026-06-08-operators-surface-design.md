# Operators Surface — on-demand `.operators` sub-struct per surface

**Date:** 2026-06-08
**Status:** Design — approved (Option 1), pending spec review
**Builds on:** `2026-06-08-topology-geometry-gauge-bundle-design.md` (un-defers part of §10 "Operators")

---

## 1. Goal

Add an `.operators` sub-struct to each of `Topology` / `Geometry` / `Gauge`,
holding the sparse operators whose *highest ingredient* is that surface, as
**live native MATLAB sparse matrices**, assembled on demand and cached for the
handle's lifetime. Opt-in via an `operators` flag so the base structs stay light
by default.

The operator natural to each level is `operators.laplacian`, forming a
through-line: **graph → cotan → connection**.

Non-goal (deferred): the lazy MATLAB `mesh.topology.operators.laplacian`-computes-
on-read ergonomics. That requires an application-side handle class (overloaded
property access), which is the Brainstorm `MeshData` layer — out of nxr-compute's
scope, exactly as the JS wrappers are out of WASM's. nxr-compute provides the
cached engine; MeshData will wrap this flag/cache and add the lazy sugar later.

---

## 2. Layout

```
Topology.operators.laplacian       V×V real sparse    graph Laplacian  L = D − A  = d0ᵀ d0
Topology.operators.dec.d0          E×V real sparse    signed incidence (exterior derivative on 0-forms)
Topology.operators.dec.d1          F×E real sparse    signed incidence (exterior derivative on 1-forms)

Geometry.operators.laplacian       V×V real sparse    cotan Laplace–Beltrami
Geometry.operators.mass.lumped     V×V diag sparse    barycentric lumped mass (= ★0)
Geometry.operators.mass.galerkin   V×V sparse         FEM Galerkin mass
Geometry.operators.hodge.h0        V×V diag sparse    ★0
Geometry.operators.hodge.h1        E×E diag sparse    ★1
Geometry.operators.hodge.h2        F×F diag sparse    ★2
Geometry.operators.hodge.h1inv     E×E diag sparse    ★1⁻¹

Gauge.operators.laplacian          V×V complex sparse connection Laplacian IN THE CURRENT GAUGE (nSym=1)
```

**Surface assignment rationale:** `d0`/`d1` are metric-free ±1 incidence →
Topology. The Hodge stars and mass are the metric weights → Geometry. The
"full DEC Laplacian" is `d0ᵀ·hodge.h1·d0` = the cotan Laplacian — assembled
*across* Topology (d) and Geometry (★), and surfaced once as
`Geometry.operators.laplacian`. The connection Laplacian additionally needs the
frame/transport → Gauge.

**`Gauge.operators.laplacian` follows the gauge:**
- `levi-civita` and `euclidean` → the Levi-Civita vertex connection Laplacian
  (`assembleConnectionLaplacian`, domain=Vertex, nSym=1, format=Complex). For
  `euclidean` the intrinsic operator is still the Levi-Civita one — euclidean is
  a *global* frame, so there is no distinct intrinsic connection; we surface the
  canonical Levi-Civita connection Laplacian and document this.
- `trivial` → the trivial-connection Laplacian built from the gauge's
  singularities (`assembleTrivialConnectionLaplacian`, domain=Vertex, nSym=1,
  format=Complex).

Native **complex** sparse (not Real2N) — MATLAB has native complex sparse, and
`K` is Hermitian. Higher `nSym` (line/cross fields) stays available through the
existing `assembleConnectionLaplacian` command.

---

## 3. Invocation — the `operators` flag

The base calls are unchanged and light by default. An `operators` field on the
options struct opts in:

```matlab
% light (today's behavior) — no .operators field present:
G = nxr_compute('geometry', h);

% with operators (cached) — .operators populated:
G = nxr_compute('geometry', h, struct('operators', true));
G.operators.laplacian        % cotan, native sparse
G.operators.hodge.h1         % ★1

B = nxr_compute('bundle', h, 'trivial', ...
      struct('singVerts', sv, 'singValues', si, 'operators', true));
B.Topology.operators.laplacian   % graph L
B.Geometry.operators.laplacian   % cotan L
B.Gauge.operators.laplacian      % trivial connection L (complex)
```

Signature changes:
- `nxr_compute('topology', h[, opts])` — gains an optional opts struct; only
  `opts.operators` is read.
- `nxr_compute('geometry', h[, opts])` — same.
- `nxr_compute('gauge', h, type[, opts])` — `opts.operators` added to the
  existing opts (alongside `singVerts`/`singValues`/`source`).
- `nxr_compute('bundle', h, gaugeType[, opts])` — `opts.operators` propagates to
  all three sub-structs.

When `opts.operators` is absent or false, behavior is byte-identical to today
(no `.operators` field). Element-grouped struct shapes and the §3 index-base /
§5 complex-frame contracts from the base spec are unchanged.

---

## 4. Caching

All assembly is cached on the `ContextHolder` for the handle's lifetime — first
request assembles, repeats are instant ("if not already computed and available"):

| Operator | Source | Cache |
|---|---|---|
| graph Laplacian | `d0ᵀ d0` (new lib fn `graphLaplacian`) | new `ContextHolder` slot |
| `d0`, `d1`, `hodge.*` | `ensureDec(h)` → `DECOperators` | existing `dec` |
| cotan, `mass.lumped` (=★0) | `ensureOps(h)` → `ManifoldOperators` | existing `ops` |
| `mass.galerkin` | Galerkin variant assembly | new `ContextHolder` slot (or `ensureOps` galerkin) |
| connection (LC / trivial) | `assembleConnectionLaplacian` / `assembleTrivialConnectionLaplacian` | existing `clCache` (keyed by domain/nSym/reg/format) |

The marshaling cost (sparse → mxArray) is paid each call the field is present;
the assembly cost is paid once.

---

## 5. New code

**Library (`nxr::manifold::ops`):**
```cpp
// Graph (combinatorial) Laplacian L = D − A = d0ᵀ d0, from the metric-free
// exterior derivative. Pure topology — independent of vertex positions.
Eigen::SparseMatrix<double> graphLaplacian(Manifold& m);
```
Implemented as `d0ᵀ d0` using the passthrough `d0(m)` accessor. Native test:
symmetric, zero row sums, PSD, diagonal = vertex degree, off-diagonal −1 iff
adjacent.

**Marshaling (`bindings/mex/src/marshal.h`):**
```cpp
// Eigen complex sparse → native MATLAB complex sparse (mxCOMPLEX, interleaved).
inline mxArray* eigenComplexSparseToMx(const Eigen::SparseMatrix<std::complex<double>>& src);
```
(The existing `eigenSparseToMx` handles the real operators.)

**MEX builders** (one per surface, take `ContextHolder&`, return `mxArray*`):
`buildTopologyOperators(h)`, `buildGeometryOperators(h)`,
`buildGaugeOperators(h, gaugeType, opts)`. Called by the surface `buildXxxStruct`
helpers when `opts.operators` is set, attaching the `.operators` field.

---

## 6. Test plan

- **Native** (`test/test_geometry_bundle.cpp`): `graphLaplacian` properties
  (symmetric, zero row sums, degree diagonal, −1 adjacency off-diagonal, PSD via
  smallest eigenvalue ≈ 0).
- **MATLAB** (new `bindings/mex/test/test_operators.m`): with `operators=true`,
  - `Topology.operators.laplacian` is V×V real sparse, symmetric, zero row sums;
    `dec.d0` is E×V, `dec.d1` is F×E; `d1*d0 == 0` (dδ identity, exact).
  - `Geometry.operators.laplacian` is V×V cotan (symmetric, zero row sums);
    `mass.lumped` diagonal & positive; `hodge.h1` diagonal E×E.
  - `Geometry.operators.laplacian ≈ d0ᵀ·hodge.h1·d0` (cross-surface assembly identity).
  - `Gauge.operators.laplacian` is V×V complex sparse, Hermitian; differs between
    `levi-civita` and `trivial` gauges.
  - Without the flag, `~isfield(G, 'operators')` (light-by-default preserved).
  - bundle-with-operators sub-structs equal the standalone-with-operators calls.

---

## 7. Decisions log

| # | Decision | Resolution |
|---|---|---|
| 1 | Content of `.operators.*` | live native MATLAB sparse (real; complex for connection) |
| 2 | Delivery | opt-in `operators` flag on the existing calls; light by default |
| 3 | Lazy field-read ergonomics | deferred to app-side MeshData handle class (out of scope) |
| 4 | Surface assignment | d0/d1 → Topology; cotan/mass/hodge → Geometry; connection → Gauge |
| 5 | `operators.laplacian` meaning | graph (Topology) / cotan (Geometry) / connection (Gauge) |
| 6 | Gauge connection variant | follows gauge type: LC for euclidean+levi-civita, trivial for trivial |
| 7 | Connection format | native complex sparse, nSym=1 (higher nSym via existing command) |
| 8 | Graph Laplacian | `d0ᵀ d0`, new `nxr::manifold::ops::graphLaplacian` |
| 9 | Caching | ContextHolder (existing ops/dec/clCache + new graph/galerkin slots) |
