# nxr.viz — patch-based validation visualization (MATLAB)

**Status:** approved 2026-05-27
**Author:** Diellor Basha (with Claude as scribe)
**Repository:** `nxr-compute`

---

## 1. Goal

A small MATLAB package, `nxr.viz.*`, for **visually validating** nxr-compute
results on a real surface from inside MATLAB — render the manifold, scalar
fields (eigenvectors, poisson, heat, curvature, signed distance), vector
fields (normals, gradient/DEC, direction fields), line segments (paths,
isolines, stripes), and a small suite of analysis plots.

The purpose is **validation / sanity-checking**, not production rendering.

## 2. Non-goals

- **No production or interactive 3-D viz.** Sophisticated, interactive
  visualization is handled downstream by `nxr-viewer` and the
  `nxr-design-system` charts-manifold (three.js + WebGPU). `nxr.viz` is the
  quick MATLAB validation counterpart only.
- **No `surfaceMesh`/`Viewer3D`.** Although the surface could render in
  `surfaceMeshShow`, its `Viewer3D` is a closed viewer — no arrow/line
  overlays, no `subplot`/`tiledlayout` tiling, no data-linked `colorbar`.
  Validation needs overlays, colorbars, and dashboards, so we use classic
  `axes` + `patch` for everything (one consistent, composable backend).

## 3. Backend & conventions

- **Backend:** classic `axes`. `patch` (with `FaceVertexCData`) for the
  surface + scalar colormaps; `quiver3` for vector arrows; `line`/`plot3`
  for segments; `tiledlayout` for the suite.
- **Colormap conventions** (mirroring `docs/visualization-recipes.md`):
  - **Signed** scalar (eigenvectors, signed distance, mean curvature) →
    diverging blue-white-red colormap, **symmetric** clim `[-a, +a]`,
    `a = max(|values|)`.
  - **Unsigned** scalar (geodesic distance, |gradient|) → sequential
    (`parula`), clim `[min, max]`.
  - Signed/unsigned is auto-detected (`any(values < 0)`), overridable.
  - **Vectors** → `quiver3` arrows; per-**vertex** (rows == nV, e.g.
    normals) anchored at vertices, per-**face** (rows == nF, e.g. gradient)
    anchored at face centroids; auto-detected by row count.
  - **Segments** → `[2N × 3]` endpoint pairs drawn as `line` pairs.
- Two-hemisphere surfaces render as a single `patch` object; default two
  `camlight`s + `axis equal vis3d` + a sensible default view.

## 4. Components

Package `bindings/mex/matlab/+nxr/+viz/`. Every function accepts an optional
`'Parent', ax` (default: a fresh figure/axes) so views compose into
dashboards, and returns the primary graphics handle.

**Primitives** (raw `(V, F, data)`, decoupled from the `M` struct):

| Function | Purpose |
|---|---|
| `nxr.viz.surface(V, F, opts)` | Lit gray manifold (`patch` + camlights + equal axes). The minimal starting point. |
| `nxr.viz.scalar(V, F, values, opts)` | Per-vertex scalar field on the surface (`FaceVertexCData` + colormap + colorbar; signed/unsigned conventions above). |
| `nxr.viz.vectorField(V, F, vecs, opts)` | `quiver3` arrows over the surface; per-vertex or per-face auto-detected by row count. |
| `nxr.viz.segments(positions, opts)` | `[2N × 3]` endpoint pairs → line segments (paths, isolines, stripes). |

**Helpers:**

| Function | Purpose |
|---|---|
| `nxr.viz.divergingColormap(n)` | Blue-white-red diverging map (MATLAB has no good built-in one) for signed scalars. |

**M-aware layer + suite** (sugar over the primitives, for the explore `M`):

| Function | Purpose |
|---|---|
| `nxr.viz.show(M, 'field.path', ...)` | Look up `M.<path>`, choose the right primitive by shape, apply smart defaults (e.g. `'solve.eigen', k` → signed scalar of eigenvector k; `'measure.distance'` → unsigned scalar; `'interpolate.smoothFace'` → face vectors). |
| `nxr.viz.spectrum(eigenvalues, opts)` | Eigenvalue plot, optionally overlaid against `M.reference.eigenvalues`. |
| `nxr.viz.dashboard(M)` | `tiledlayout` validating several results at once (surface, an eigenmode, curvature, geodesic distance, normals, spectrum). |

## 5. Options (common)

`scalar`/`surface`/`vectorField`/`segments` share name-value options where
meaningful: `Parent` (axes), `Title`, `Colormap`, `Clim`, `Signed`
(true/false/auto), `EdgeColor` (default `'none'`), `Alpha`, and for vectors
`Scale`/`Color`, for segments `Color`/`LineWidth`.

## 6. Data shapes consumed

`V` `[nV×3]`, `F` `[nF×3]` (1-based); scalar `values` `[nV×1]`; vectors
`[nV×3]` (vertex) or `[nF×3]` (face); segments `[2N×3]`. These match the
nxr-compute / `M`-struct outputs directly.

## 7. Testing / validation

Because the output is figures, "testing" is visual:

- A demo script (`bindings/mex/test/viz_cortex_demo.m`) renders each function
  on the Brainstorm cortex (`M` from `explore_cortex`), then
  `exportgraphics` writes a PNG per view into a scratch dir
  (`build/viz/`, git-ignored).
- Validation = the PNGs render the expected thing (surface shaded correctly,
  scalar field colored sensibly, arrows on the surface, segments as lines)
  and no function errors. The PNGs are inspected visually.
- Each function also guards inputs (shape checks → clear `error`).

## 8. Build order

`surface` → `scalar` (+ `divergingColormap`) → `vectorField` → `segments`
→ `show` → `spectrum` → `dashboard`. Validate each on the cortex (PNG) before
moving on.

## 9. Files

| File | Action |
|---|---|
| `bindings/mex/matlab/+nxr/+viz/surface.m` | new |
| `bindings/mex/matlab/+nxr/+viz/scalar.m` | new |
| `bindings/mex/matlab/+nxr/+viz/vectorField.m` | new |
| `bindings/mex/matlab/+nxr/+viz/segments.m` | new |
| `bindings/mex/matlab/+nxr/+viz/divergingColormap.m` | new |
| `bindings/mex/matlab/+nxr/+viz/show.m` | new |
| `bindings/mex/matlab/+nxr/+viz/spectrum.m` | new |
| `bindings/mex/matlab/+nxr/+viz/dashboard.m` | new |
| `bindings/mex/test/viz_cortex_demo.m` | new — render-everything demo + PNG export |
