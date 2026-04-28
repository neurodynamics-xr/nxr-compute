# nxr-compute architecture

Read [README.md](README.md) first. This document covers the architectural
model in depth — the layering, the boundaries between layers, how data
flows through the system, and what the internal abstractions are.

---

## Mental model

nxr-compute is a pure C++ math library. **Inputs are vertices and faces; outputs
are math results.** Sparse matrices, dense vectors, structured records —
everything is values, not objects holding hidden state, not protocols, not
RPCs. Anyone using nxr-compute builds a thin "binding shell" that translates their
data type to/from nxr-compute's pointer-and-shape interface.

The design is borrowed directly from `geometry-processing-js`'s
`linear-algebra` package: a small math core wrapped by per-language
bindings.

---

## The four-layer architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CONSUMERS                                                       │
│  Your three.js app    Brainstorm pipeline    cortical-flow app   │
│         │                     │                       │          │
└─────────┼─────────────────────┼───────────────────────┼──────────┘
          │                     │                       │
┌─────────▼─────────────────────▼───────────────────────▼──────────┐
│  BINDINGS  (each is a small marshalling shell)                   │
│  nxr_compute.wasm           nxr_compute.mexw64       nxr_compute_addon.node    nxr-compute       │
│  (browser/JS)       (MATLAB)         (Electron N-API)  (CLI)     │
└─────────┬─────────────────────┬───────────────────────┬──────────┘
          │                     │                       │
          └──────────┬──────────┴───────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│  cxf-io  (optional, file format adapters)                        │
│  FreeSurfer parser, Zarr writer                                  │
│  Linked by the CLI and addon. NOT linked by MEX or WASM.         │
└────────────────────┬─────────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│  nxr-compute  (pure compute library)                                     │
│  Depends on: Eigen + Spectra + geometry-central                  │
│  Public API: vertices/faces in, math results out                 │
└──────────────────────────────────────────────────────────────────┘
```

Four layers, hard seams between them. Your three.js app can swap WASM for
N-API addon (or vice versa) without changing what nxr-compute computes underneath
— only the binding-specific marshalling differs.

---

## What `nxr-compute` (the math library) provides

A header (`nxr/compute.h`) declares the public C++ API. Operations are grouped
into:

- **Mesh operators** — cotangent Laplacian, mass matrix, vertex/face/edge
  counts, vertex normals (six estimator variants), vertex dual areas.
- **DEC operators** — `d0`, `d1`, `★0`, `★1`, `★2`, `★1⁻¹`. The discrete
  exterior calculus stack: gradient, curl, Hodge stars.
- **Spectral basis** — sparse generalized eigensolver (Spectra IRAM),
  M-orthonormalization, DC-mode removal.
- **Solvers** — Cholesky cache (factor once, reuse many), Poisson solve
  (`L φ = -M (ρ - ρ̄)`), heat-method geodesic distance, Hodge
  decomposition (full α + β + γ split of a 1-form).
- **Visualization plumbing** — per-face orthonormal tangent frames,
  geodesic paths via the flip-out algorithm, BFF parametrization (UV
  coordinates).
- **Field generators** — heat diffusion, damped wave (both spectral, very
  fast), random scalar / face / 1-form generators, Whitney interpolation
  (edge 1-form → face vectors), scalar gradient on faces.
- **Curvatures** — Gaussian, mean, principal κ_min / κ_max, principal
  directions.
- **Streamlines** — particle integration through a face vector field.
- **Isolines** — contour line segments from a vertex scalar.

Every operation operates on raw value buffers and returns Eigen types
(internally) or value buffers (across the binding boundary).

---

## What `cxf-io` provides (and why it's separate)

`cxf-io` is an *optional sibling library* containing file format adapters:

- FreeSurfer surface (`.pial`, `.white`, `.curv`) reader
- Zarr v2 writer (matches the cortical-flow store schema)

It exists separately from `nxr-compute` so that:

- **WASM bundles stay small** — browser apps don't ship a FreeSurfer
  parser they don't need.
- **MATLAB MEX stays clean** — MATLAB has native FreeSurfer / Zarr
  support; we don't want symbol collisions or duplicate code.
- **The math API stays values-only** — nxr-compute doesn't take file paths; the
  contract is "vertices in, math out."

The CLI links both. The Electron addon links both (the renderer's
"Import FreeSurfer subject" feature uses it). MEX and WASM link only
`nxr-compute`.

---

## Bindings — what each one looks like

A binding is a thin shell that:

1. Translates the host's data types into `(double*, int)` pointer/shape
   pairs.
2. Calls into `nxr::compute::*` functions.
3. Translates the returned `Eigen::*` types back into the host's
   container type.

Each binding has its own "language" but they all wrap the same math.

| Binding | Style | Stateful? | Data marshalling |
|---|---|---|---|
| **CLI** | Subcommand dispatch (`nxr-compute precompute …`) | Stateless per invocation | File path → buffer → nxr-compute → file |
| **MEX** | Single dispatcher: `nxr_compute('command', args)` | Stateless per call | mxArray ↔ pointer/shape (column-major matches Eigen) |
| **N-API addon** | `addon.someMethod(handle, args)` | **Stateful** — context handle caches operators / Cholesky factors | TypedArray ↔ pointer/shape |
| **WASM** | `ctx.someMethod(args)` (Embind class) | **Stateful** — same handle pattern | TypedArray (heap-aliased) ↔ pointer/shape |

Stateful vs stateless matters for performance: stateful bindings (addon,
WASM) keep the cotangent Laplacian and Cholesky factors alive between
calls, so a Hodge solve after a Poisson solve doesn't re-factor anything.
Stateless bindings (MEX, CLI) build everything per call but match the
host's natural usage pattern.

---

## What is `ManifoldSurfaceMesh` — internal or abstraction?

**Internal.** It's `geometry-central`'s data structure for "a triangle
mesh that satisfies the manifold property" (no non-manifold edges, no
non-manifold vertices, halfedges either have a twin or sit on a single
boundary loop). nxr-compute uses it to back its `ComputeContext`:

```cpp
class nxr::compute::ComputeContext {
public:
    ComputeContext(const double* vertices, int nV,
                   const int32_t* faces, int nF);
    // …
private:
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> mesh_;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry_;
};
```

When you (from MATLAB, JS, the CLI, anywhere) call
`nxr::compute::ComputeContext(verts, faces)`, nxr-compute builds the `ManifoldSurfaceMesh`
internally. **Consumers never touch the mesh class directly.** They pass
arrays in and get arrays back.

What consumers DO see is:

- A `ComputeContext` opaque handle (in WASM and addon — kept alive across
  calls so cached state isn't lost).
- A transient context (in MEX and CLI — built per call, discarded).

The only consumer-facing constraint is "your input mesh must be
manifold." If you pass a non-manifold mesh (duplicate edges, T-junctions,
multiply-connected vertices), `ComputeContext` construction throws.
Cortical surfaces from FreeSurfer always satisfy this.

This is the same pattern as `geometry-processing-js`'s linear-algebra
package: the JS side has `Vector` and `SparseMatrix` handles wrapping
internal Eigen objects. Consumers don't manipulate Eigen storage; they
work with the JS-facing API.

---

## Data flow at the API boundary

Every binding expresses the same shape, in its language:

**Inputs:**
- `vertices`: a buffer of `V × 3` float64 values (xyz triples, row-major)
- `faces`: a buffer of `F × 3` int32 values (vertex indices, **0-based**)

**Outputs depend on the operation:**
- Dense vector: 1D float64 buffer of length N
- Dense matrix: 2D float64 buffer (rows × cols, row-major flat)
- Sparse matrix: COO triplets `{row, col, value, rows, cols}` or CSC
  `{indptr, indices, data}` depending on binding
- Struct of multiple results: object/struct with named fields
- Time-series: `[T × V]` float32 (for memory efficiency on large meshes)

In WASM specifically, all bulk data crosses the boundary as **typed
arrays in WASM heap memory** (`Float64Array`, `Int32Array`, `Float32Array`).
The JS side allocates, fills, calls nxr-compute, and reads results. There's a
small per-call marshalling overhead but no copying for objects you keep
around — nxr-compute's `ComputeContext` lives in WASM linear memory, you hold a
JS handle.

### Indexing conventions

- Faces are **0-based** at the nxr-compute boundary in all bindings except MEX
  (where the binding shell automatically converts MATLAB's 1-based
  indexing to 0-based on entry).
- Vertices: V vertices indexed `0..V-1`.
- Faces: F faces indexed `0..F-1`.
- Edges: E edges, indexed by halfedge convention. Most consumers don't
  need to materialize edge indices — they're implicit in the operators.

### Storage conventions

- Vertices: row-major xyz triples (consumer-friendly, matches three.js
  `BufferAttribute('position', 3)`).
- Faces: row-major triangle triples.
- Eigenvectors: column-major `[V × K]` matrix (each eigenvector is a
  column). To extract the k-th mode in JS:
  ```javascript
  const mode_k = eigenvectors.subarray(k * nV, (k + 1) * nV)
  ```
- Time-series activity: row-major `[T × V]` (each row is a frame).
- Sparse matrices: CSC by default (matches Eigen and MATLAB); some
  bindings also expose COO for ergonomic JS construction.

---

## The dependency policy

nxr-compute has **three** allowed third-party dependencies, all C++ template /
header-only or vendored:

1. **Eigen** (header-only) — foundational linear algebra
2. **Spectra** (header-only, depends only on Eigen) — sparse eigensolver
   via IRAM. No equivalent in Eigen itself; densification is unviable
   (40k × 40k double would be 12 GB).
3. **`geometry-central`** (vendored as submodule, Eigen-only) — halfedge
   mesh, cotan operators, DEC, heat-method geodesics, flip-out geodesic
   paths, BFF parametrization, direction fields.

**Excluded by policy:** SuiteSparse / CHOLMOD, MKL, PARDISO, CUDA /
cuSPARSE, system BLAS, MPI, anything that requires a separate install or
breaks the WASM target. The performance gains from these libraries
evaporate in WASM (no threading, no BLAS3) and risk symbol collisions in
MEX (MATLAB ships its own bundled SuiteSparse).

The rule keeps nxr-compute shippable as a single static archive that compiles
unchanged for native, MATLAB MEX, and WebAssembly.

---

## The "library, not framework" philosophy

Frameworks have opinions about how you build your app. nxr-compute is a library:
it does math, you do everything else.

- **You** load your mesh data however you want — three.js loaders, REST
  fetches, Zarr stores via `@zarrita/core`, `@react-three/fiber`,
  whatever. nxr-compute takes the vertices and faces.
- **You** display results however you want — three.js shaders, plain
  GLSL, TSL, fixed-function arrow glyphs, line meshes. nxr-compute gives you the
  values to feed into your renderer.
- **You** run nxr-compute at the rate you need — once on app load, on every
  Timeline frame change, in a worker, in the main thread. nxr-compute doesn't
  schedule itself.

If a nxr-compute operation isn't fast enough for your use case (e.g., heat
diffusion needs to run at 60 fps on a 100k-vertex mesh), the answer is
*usually* "use the spectral generators" or "factor your matrices once and
reuse" — both already supported by the library. If that's still not
enough, it's likely a job for the GPU layer of your renderer, not for
nxr-compute to absorb new responsibilities.

---

## What `nxr-compute` does NOT do — explicitly

- **No file I/O.** That's `cxf-io`'s job, and it's optional.
- **No rendering.** No three.js, no WebGL, no Canvas. nxr-compute is values.
- **No threading abstractions.** Single-threaded. If you need
  parallelism, run multiple WASM instances in workers, or call nxr-compute from
  a thread you manage in your binding shell.
- **No GPU compute.** nxr-compute runs on the CPU. The GPU is for the
  visualization layer (three.js TSL, WebGPU compute shaders, etc.).
- **No format conversions besides `cxf-io`'s.** No OBJ / PLY / NIFTI /
  GIFTI parsers in nxr-compute itself. Add them to `cxf-io` (and link from the
  CLI / addon) if you need them, or handle parsing in your host language
  (browsers have lots of mesh parsers; MATLAB has built-ins for many
  formats).
- **No persistence.** nxr-compute has no concept of a "save this" or "load
  that." If you want to cache eigenmodes across app sessions, write
  them to your store from JS (or from MATLAB, whatever) — nxr-compute returned
  them as a buffer; how you persist them is your concern.

This is what keeps the math library small, fast to compile, and portable.
