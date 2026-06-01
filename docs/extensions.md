# nxr-compute API extensions — C++-first design proposal

> **Status (2026-04-28): Phase A landed, with deviations.**
> The implementation diverges from this proposal in several places —
> see `docs/nxr-compute-usage.md` for the actual delivered API and
> `native/CLAUDE.md §11–§12` for the binding contract.
>
> Notable deviations from the proposal:
> - **C++17 floor**, not C++20. Eigen 3.4 + Emscripten Clang + C++20
>   trips a template-deduction bug; deferred to Phase B with an Eigen
>   upgrade.
> - **`nxr::compute::CancellationToken`** instead of `std::stop_token` — the
>   `stop_token_from_atomic` bridge in §5.2 had no spec-compliant
>   mechanism. The custom token cleanly accepts an atomic-flag pointer
>   *or* a `std::function<bool()>` poller (the latter for MATLAB Ctrl-C
>   via `utIsInterruptPending`).
> - **Type-aliased indices** (`using VertexIndex = int32_t;`) instead
>   of scoped enums — the bug class is rare and the binding cost is
>   real per-call-site.
> - **vMajor (row-major) eigenvector layout** instead of column-major.
>   Matches the cortical-flow Zarr schema and the GPU spectral-synthesis
>   access pattern; column-major is cheaper for "extract one mode" but
>   that's not the hot path.
> - **Namespace reorg / split headers / RAII solver classes** are
>   Phase B+, not Phase A. Phase A is purely additive cross-cutting
>   features (errors, cancel, progress) plus the storage-convention fix.
>
> The proposal is preserved as design rationale.

Status: **proposal** — for review by the nxr-compute author (Diellor) before
implementation.

This document specifies API additions and reorganizations for nxr-compute, the
C++ math/compute library underpinning cortical-flow and the NXR design
system's surface charts. **nxr-compute is designed as a modern C++ library
first.** Bindings (Emscripten/WASM, MATLAB MEX, Node.js N-API, CLI) are
downstream consumers that wrap a stable C++ surface — they do not drive
the C++ design.

The design follows mainstream C++20 idioms: namespaces, free functions
for math, RAII classes for state, `std::span` / `std::stop_token` from
the standard library, strong-typed indices, exception-based errors with
codes, and PIMPL for ABI stability.

The downstream JS/React adapter (`@nxr/nxr-compute-react` in the NXR
gallery-surface project) is the proximate motivator for several of
these additions, but the public C++ surface is what gets shipped — the
JS layer just consumes it.

---

## 1. Guiding principles

### 1.1 nxr-compute does math; consumers do everything else

nxr-compute is a header-light, dependency-light C++ static library. The
boundary is firm — anything that can live in a binding shell or
consumer should live there:

| Concern | Owner |
|---|---|
| Sparse linear algebra, halfedge queries, eigensolvers | **nxr-compute (C++)** |
| Numerical correctness, cache lifecycle, std::stop_token semantics | **nxr-compute (C++)** |
| Mesh I/O (OBJ, GLTF, FreeSurfer parsing) | `cxf-io` (sibling lib) or consumer |
| State machines, UI feedback, progress bars | consumer (e.g. `@nxr/nxr-compute-react`) |
| Cross-process / cross-language marshalling | binding shell (Embind, MEX, N-API) |
| Caching computed results across sessions | consumer (e.g. IndexedDB) |
| Web Worker orchestration, message-passing | consumer |
| Throttling, debouncing, scheduling | consumer |

nxr-compute exposes the **minimum primitives** needed for consumers to do their
job. Where a consumer would otherwise be unable to deliver a feature
(e.g., showing real progress during a long solve), nxr-compute adds a
*low-level* hook (an atomic-counter observer pointer) — never a
high-level abstraction.

### 1.2 No serialization headers in result types

nxr-compute result types contain **only the numerical/structural information
the math produced**. They do not carry provenance metadata or
schema-versioned `Header` structs. Provenance is the binding's
responsibility: any consumer that wants to log, serialize (JSON, Zarr,
HDF5), or display construction parameters can build that from:

1. The arguments the consumer passed in to the call (which the consumer
   already knows).
2. Diagnostic fields nxr-compute returns on the result (e.g.,
   `EigenResult::nIterations`, `EigenResult::finalResidualNorm`,
   `EigenResult::info`).

This keeps nxr-compute result structs flat, cheap to construct, and free of
string-encoded equations or version strings that don't belong in a math
library.

### 1.3 C++20 baseline

nxr-compute targets C++20. This unlocks `std::span`, `std::stop_token`,
`std::atomic_ref`, designated initializers for option structs, and
concepts where useful. Emscripten's Clang and modern GCC/MSVC all
support these cleanly; geometry-central compiles at C++17 but builds
fine at C++20.

### 1.4 Namespace = subpackage

The library is organized into a small set of namespaces, each owning
one mathematical concern. One header per namespace, plus an umbrella
`nxr/compute.h`. Namespaces nest one level (`nxr::compute::eigen`, `nxr::compute::geometry`)
with `nxr::compute::detail` reserved for non-public implementation helpers.

### 1.5 Free functions for math, classes only for state

Math operations are **free functions** that take their inputs (matrices,
mesh handles, options) explicitly. Classes are reserved for entities
that own state — the mesh handle (`ComputeContext`), cached solver
factors (`PoissonSolver`), etc.

This is consistent with `<algorithm>`, `<numeric>`, Eigen's solver
free functions, and the broader modern C++ idiom: classes earn their
keep only when they own resources or invariants.

---

## 2. Module organization

```
include/nxr-compute/
  compute.h                    ← umbrella; pulls in all public headers
  types.h              ← shared structs & strong-typed indices
  errors.h             ← Error, ErrorCode
  cancellation.h       ← interop helpers around std::stop_token
  progress.h           ← ProgressObserver
  context.h            ← ComputeContext (mesh handle)
  topology.h           ← namespace nxr::compute::topology
  geometry.h           ← namespace nxr::compute::geometry
  operators.h          ← namespace nxr::compute::operators   (note: 'operator' is a keyword)
  eigen.h              ← namespace nxr::compute::eigen
  solve.h              ← namespace nxr::compute::solve       (PoissonSolver, HeatSolver, …)
  query.h              ← namespace nxr::compute::query
  transform.h          ← namespace nxr::compute::transform
  connection.h         ← namespace nxr::compute::connection
  health.h             ← namespace nxr::compute::health
src/
  context.cpp          ← ComputeContext::Impl pimpl
  topology.cpp
  …                    ← one .cpp per header
```

### 2.1 Naming hazard: `operator`

`operator` is a C++ reserved keyword. The MATLAB toolbox's `+operator`
subpackage cannot be named identically in C++. Adopting the plural
**`operators`** matches the standard library's plural-noun convention
(`<algorithm>`, `<numeric>`, `<concepts>`) and reads cleanly:

```cpp
auto L = nxr::compute::operators::laplaceBeltrami(ctx);
auto M = nxr::compute::operators::mass(ctx);
auto g = nxr::compute::operators::gradient(ctx, scalar);
```

### 2.2 Subpackage namespaces

| Namespace | Purpose |
|---|---|
| `nxr::compute::topology`     | Halfedge structure, adjacency, neighbors, edges, twins, face/vertex/edge enumeration |
| `nxr::compute::geometry`     | Per-element properties: face frames, areas, centroids, normals; vertex normals, dual areas, curvatures; edge lengths, cotan weights |
| `nxr::compute::operators`    | Discrete operators: Laplace–Beltrami, mass, stiffness, gradient, divergence, curl, DEC stack, Hodge Laplacian, MFT/IMFT, connection Laplacian |
| `nxr::compute::eigen`        | Generalized eigenproblem: solve, M-orthonormalization, DC-mode removal, warm-load |
| `nxr::compute::solve`        | Cached solvers as RAII classes: `PoissonSolver`, `HeatSolver`, `HeatDistanceSolver` |
| `nxr::compute::query`        | Geodesic distance / path, shortest paths, neighbors, BFS/DFS, dual-mesh queries, isolines, streamlines |
| `nxr::compute::transform`    | Field transforms: scalar gradient on faces, BFF parametrization, scalar→edge1form, edge1form→face vectors (Whitney) |
| `nxr::compute::connection`   | Trivial connections, parallel transport, direction fields with prescribed singularities |
| `nxr::compute::health`       | Validate / report / repair mesh health (no compute — separate concern) |

Internal helpers live under `nxr::compute::detail` or per-subpackage
`nxr::compute::eigen::detail`, etc., and are not part of the public ABI.

---

## 3. Core types

### 3.1 Strong-typed indices

Raw `int` for indexing into vertices vs. faces vs. edges is a class of
bugs C++ can statically prevent. nxr-compute uses scoped enums over `int32_t`:

```cpp
namespace nxr-compute {
  enum class VertexIndex   : int32_t {};
  enum class FaceIndex     : int32_t {};
  enum class EdgeIndex     : int32_t {};
  enum class HalfedgeIndex : int32_t {};

  // Free helpers — explicit conversions, no implicit widening
  constexpr int32_t  to_int(VertexIndex v) noexcept { return static_cast<int32_t>(v); }
  constexpr VertexIndex make_vertex(int32_t i) noexcept { return VertexIndex{i}; }
  // …
}
```

Zero runtime cost. Embind binds these as plain JS numbers; MEX/N-API
binders likewise convert transparently.

### 3.2 Buffer parameters with `std::span`

All buffer inputs use `std::span<const T>` (read) or `std::span<T>`
(write). Replaces `(const T* data, size_t size)` pairs everywhere.

```cpp
ComputeContext::ComputeContext(std::span<const double>  vertices,    // V × 3 row-major
                               std::span<const int32_t> faces);      // F × 3 row-major
```

`std::span` works equally with `std::vector`, raw arrays, Eigen `Map`,
and the WASM-heap-backed views the Embind layer constructs.

### 3.3 Result structs are flat

Every result struct contains **only** what the computation produced
plus the diagnostics needed to interpret it. No nested `Header`. No
encoded equations. No version strings. Consumers that want serializable
provenance build it from their own call-site context.

```cpp
struct EigenResult {
  Eigen::MatrixXd eigenvectors;     // V × K, column-major (Eigen default)
  Eigen::VectorXd eigenvalues;      // K
  int nConverged       = 0;
  int nIterations      = 0;
  double finalResidualNorm = 0.0;
  enum class Info { Converged, MaxIter, Partial, Cancelled } info
                       = Info::Converged;
};

struct FaceFrames {
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> e1;     // F × 3
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> e2;     // F × 3
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> normals; // F × 3
};
```

Result structs are move-only when they wrap large buffers (no
accidental deep copies); copy-constructible only if their members are
cheap.

### 3.4 Option structs with designated initializers

Each operation accepts a single `Options` struct so call sites stay
self-documenting:

```cpp
namespace nxr::compute::eigen {
  struct SolveOptions {
    double sigma     = -1e-8;
    double tol       = 1e-10;
    int    maxIter   = 0;          // 0 → use Spectra's default
    int    ncv       = 0;          // 0 → 2k + 1
    bool   symmetric = true;       // FEM Laplacian symmetry
  };
}

// Call site uses C++20 designated initializers:
auto eig = nxr::compute::eigen::solve(L, M, /*k=*/1000, {
  .tol     = 1e-6,
  .maxIter = 5000,
}, stop, progress);
```

---

## 4. Errors

### 4.1 Single class with `enum class` code

```cpp
namespace nxr-compute {
  enum class ErrorCode {
    InvalidInput,
    NonManifold,
    OpenMeshRequired,
    ClosedMeshRequired,
    GeometryDegenerate,
    EigensolveInvalidK,
    EigensolveNotConverged,
    NotPrecomputed,
    Cancelled,
    OutOfMemory,
    InternalError,
  };

  class Error : public std::runtime_error {
  public:
    Error(ErrorCode code,
             std::string message,
             std::string hint = {});

    ErrorCode               code()    const noexcept;
    std::string_view        hint()    const noexcept;

  private:
    ErrorCode   code_;
    std::string hint_;
  };
}
```

C++ consumers `catch (const nxr::compute::Error& e)` and switch on `e.code()`.
Embind translates the exception to a JS `Error` subclass with `.code`
exposed as the enumerator name (e.g., `'NON_MANIFOLD'`); MATLAB MEX
maps `code()` to `MException` identifier (`nxr-compute:nonManifold`); the CLI
can map to exit codes.

### 4.2 No `std::expected`

`std::expected<T, Error>` (C++23) was considered. Rejected because:

- Exceptions integrate cleanly with Embind, MEX, and N-API — all three
  binders translate thrown C++ exceptions automatically. `expected`
  would require manual unwrapping in every binding shell.
- Most nxr-compute operations have a single failure mode along the happy path;
  exceptions match the cost profile.
- C++23 availability is still patchy in WebAssembly toolchains.

If a *non-throwing* validation API is needed (it is — see
`nxr::compute::health::check` in §6.2), it returns a value type, not
`std::expected`.

---

## 5. Cancellation and progress

### 5.1 Cancellation via `std::stop_token`

`std::stop_token` is the C++20 standard cancellation primitive. nxr-compute
uses it directly — no custom token type.

```cpp
namespace nxr::compute::eigen {
  EigenResult solve(const Eigen::SparseMatrix<double>& K,
                    const Eigen::SparseMatrix<double>& M,
                    int k,
                    const SolveOptions& opts = {},
                    std::stop_token stop = {});
}
```

C++ users:

```cpp
std::stop_source src;
auto fut = std::async([&] {
  return nxr::compute::eigen::solve(L, M, 1000, {}, src.get_token());
});
// elsewhere, on user click:
src.request_stop();
// fut.get() throws Error(ErrorCode::Cancelled, ...)
```

The eigensolver checks `stop.stop_requested()` once per outer Spectra
IRAM iteration (~100 ms apart on cortical-sized meshes). On `true`,
throws `Error(ErrorCode::Cancelled, ...)`.

### 5.2 Bridging external atomic flags

WASM hosts (and any consumer that drives cancellation from another
thread / process) need to flip the stop state without owning a
`std::stop_source`. nxr-compute provides a bridge in `cancellation.h`:

```cpp
namespace nxr-compute {
  // Returns a stop_token whose stop state mirrors the value at *flag.
  // The flag's storage may live anywhere — stack, heap, WASM linear
  // memory backed by a SharedArrayBuffer. Caller guarantees it
  // outlives the token. A zero value means "not stopped".
  std::stop_token stop_token_from_atomic(const std::atomic<int32_t>* flag);
}
```

JS adapters allocate an `Int32Array` over a `SharedArrayBuffer`, get
its WASM heap pointer, and pass it through this helper. From the
solver's perspective, it's still just a `std::stop_token`.

**Implementation note.** A small bookkeeping object inside
`stop_token_from_atomic` polls the atomic and forwards to a
`std::stop_source`. The polling cost is amortized across the existing
once-per-outer-iteration check in the solver — no extra threads, no
busy loops.

### 5.3 Progress via `ProgressObserver`

```cpp
namespace nxr-compute {
  struct ProgressObserver {
    std::atomic<int32_t>* iteration       = nullptr;
    std::atomic<int32_t>* totalIterations = nullptr;
    std::atomic<int32_t>* residualMicro   = nullptr;  // residual × 1e6, fits int32

    void update(int iter, int total, double residual) const noexcept {
      if (iteration)       iteration      ->store(iter,                                std::memory_order_relaxed);
      if (totalIterations) totalIterations->store(total,                               std::memory_order_relaxed);
      if (residualMicro)   residualMicro  ->store(static_cast<int32_t>(residual * 1e6),
                                                  std::memory_order_relaxed);
    }
  };
}
```

Allocation-free, lock-free, bridges naturally to a SharedArrayBuffer
on the JS side.

`std::function<void(int, double)>` was the obvious alternative.
Rejected because (a) it allocates, (b) calling JS functions from WASM
requires Asyncify which would balloon the bundle ~60%, and (c) the
atomic-counter approach is more flexible — C++ users can poll the same
atomics from any thread without the indirection.

Operations that accept progress observers:

```cpp
namespace nxr::compute::eigen {
  EigenResult solve(..., std::stop_token stop = {},
                    ProgressObserver progress = {});
}

namespace nxr::compute::transform {
  ParametrizationResult bff(..., std::stop_token stop = {},
                            ProgressObserver progress = {});
}

namespace nxr::compute::query {
  StreamlinesResult traceStreamlines(..., std::stop_token stop = {},
                                     ProgressObserver progress = {});
}
```

Symmetric API across all long-running operations. Same primitives, same
parameters at the end of every signature.

---

## 6. ComputeContext (mesh handle)

```cpp
namespace nxr-compute {
  class ComputeContext {
  public:
    // Constructor performs manifold validation; throws Error on failure.
    ComputeContext(std::span<const double>  vertices,    // V × 3 row-major
                   std::span<const int32_t> faces);      // F × 3 row-major, 0-based

    ~ComputeContext();
    ComputeContext(ComputeContext&&) noexcept;
    ComputeContext& operator=(ComputeContext&&) noexcept;
    ComputeContext(const ComputeContext&)            = delete;
    ComputeContext& operator=(const ComputeContext&) = delete;

    // O(1) introspection. None throw.
    int  nV() const noexcept;
    int  nE() const noexcept;
    int  nF() const noexcept;
    int  eulerCharacteristic() const noexcept;
    bool isClosed() const noexcept;
    int  numConnectedComponents() const noexcept;
    int  numBoundaryLoops() const noexcept;
    // genus() throws Error(OpenMeshRequired) if the mesh is not closed
    int  genus() const;

    // Stable per-context identity for caching (e.g., IndexedDB keys).
    // Computed once on construction, cached. SHA-256 of canonicalized
    // vertex+face bytes.
    std::string fingerprint() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Friend namespace functions need access to impl_
    friend class detail::ContextAccess;
  };
}
```

PIMPL hides geometry-central types and cached factor storage from the
public header. Move-only ownership; consumers can shuffle contexts
through containers but never accidentally deep-copy.

Introspection is on the class because it's *about* the state the
class owns, and its implementation is O(1) reads against pre-built
halfedge data. Compute (operators, eigensolves, geodesics) is in
namespace functions taking `const ComputeContext&` — see §7+.

---

## 7. Eigen subpackage

### 7.1 Solve

```cpp
namespace nxr::compute::eigen {
  struct SolveOptions {
    double sigma     = -1e-8;
    double tol       = 1e-10;
    int    maxIter   = 0;          // 0 → use Spectra default
    int    ncv       = 0;          // 0 → 2k + 1
    bool   symmetric = true;
  };

  // Generalized eigenproblem K u = λ M u. Smallest-magnitude k modes,
  // M-orthonormalized, DC mode removed (consistent with nxr-compute's existing
  // precompute behavior).
  //
  // Throws Error on:
  //   - InvalidInput              (K, M dimension mismatch; k < 1)
  //   - EigensolveInvalidK        (k > N - 1)
  //   - EigensolveNotConverged    (Spectra reports failure with no partial result)
  //   - Cancelled                 (stop_token requested)
  EigenResult solve(const Eigen::SparseMatrix<double>& K,
                    const Eigen::SparseMatrix<double>& M,
                    int k,
                    const SolveOptions& opts = {},
                    std::stop_token stop = {},
                    ProgressObserver progress = {});

  // M-orthonormalize columns of U in place against M.
  void normalize(Eigen::Ref<Eigen::MatrixXd> U,
                 const Eigen::SparseMatrix<double>& M);

  // Drop the (numerically-determined) DC mode if present.
  EigenResult removeDC(EigenResult&& r,
                       double dcEigenvalueTol = 1e-12);
}
```

`solve` takes K and M as plain `Eigen::SparseMatrix<double>` — it does
not require a `ComputeContext`. Callers that compute a Laplace–Beltrami
operator from a mesh will have built it via
`nxr::compute::operators::laplaceBeltrami(ctx)` (see §8.1) and pass the
resulting matrices to `nxr::compute::eigen::solve`. This decouples *eigensolving*
from *being a manifold*.

### 7.2 Warm-load (cross-session caching)

```cpp
namespace nxr::compute::eigen {
  // Validate that an externally-supplied eigenpair set is consistent
  // with the supplied K, M (size + M-orthonormality within tolerance).
  // Use case: consumer persists eigenvectors/eigenvalues to disk via
  // its own storage layer; on reload, validates and uses them with
  // downstream spectral generators.
  //
  // Throws Error(InvalidInput) on dimension or orthonormality
  // violation.
  EigenResult validateLoaded(Eigen::MatrixXd eigenvectors,
                             Eigen::VectorXd eigenvalues,
                             const Eigen::SparseMatrix<double>& M,
                             double orthoTol = 1e-6);
}
```

nxr-compute does not own the persistence layer — it just ratifies what the
consumer hands back, returning a normal `EigenResult`. Consumers that
trust their cache fully can skip validation and construct an
`EigenResult` directly.

---

## 8. Operators subpackage

### 8.1 Building blocks

Each operator is a free function taking `const ComputeContext&` plus
its own `Options` struct. All return Eigen matrices/vectors directly —
no nested wrapper types.

```cpp
namespace nxr::compute::operators {
  enum class MassVariant       { Lumped, Galerkin };
  enum class StiffnessVariant  { Cotan };
  enum class StiffnessSign     { Positive, Negative };

  struct LaplaceBeltramiOptions {
    MassVariant      mass       = MassVariant::Lumped;
    StiffnessVariant stiffness  = StiffnessVariant::Cotan;
    StiffnessSign    sign       = StiffnessSign::Positive;
    bool             symmetrize = true;
  };

  struct LaplaceBeltramiResult {
    Eigen::SparseMatrix<double> S;   // stiffness
    Eigen::SparseMatrix<double> M;   // mass
  };

  LaplaceBeltramiResult laplaceBeltrami(const ComputeContext& ctx,
                                        const LaplaceBeltramiOptions& opts = {});

  Eigen::SparseMatrix<double> mass     (const ComputeContext& ctx, MassVariant      = MassVariant::Lumped);
  Eigen::SparseMatrix<double> stiffness(const ComputeContext& ctx, StiffnessVariant = StiffnessVariant::Cotan,
                                                                   StiffnessSign    = StiffnessSign::Positive);

  struct DECOperators {
    Eigen::SparseMatrix<double> d0, d1;
    Eigen::SparseMatrix<double> hodge0, hodge1, hodge1Inverse, hodge2;
  };
  DECOperators dec(const ComputeContext& ctx);

  // Per-vertex / per-face / per-edge derived ops
  Eigen::VectorXd                                              gradient (const ComputeContext& ctx,
                                                                         Eigen::Ref<const Eigen::VectorXd> scalar);
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>    faceGradient(const ComputeContext& ctx,
                                                                         Eigen::Ref<const Eigen::VectorXd> scalar);
  Eigen::SparseMatrix<double>                                  hodgeLaplacian(const ComputeContext& ctx,
                                                                              int kForm);
}
```

### 8.2 Splitting the old `precompute()`

The current `ctx.precompute({ k })` aggregator is **removed**. Consumers
compose what they need:

```cpp
ComputeContext ctx{verts, faces};

// Cheap, run on mesh load:
auto frames  = nxr::compute::geometry::face::frames(ctx);            // ~3 ms
auto vNorms  = nxr::compute::geometry::vertex::normals(ctx);         // ~1 ms
auto curv    = nxr::compute::geometry::vertex::curvatures(ctx);      // ~10 ms

// Build operators only when needed:
auto LB      = nxr::compute::operators::laplaceBeltrami(ctx);        // ~50 ms

// Expensive, opt-in:
auto eig     = nxr::compute::eigen::solve(LB.S, LB.M, 1000);         // seconds–minutes
```

The aggregator was a convenience that obscured cost; with namespaced
free functions, every line names its own cost.

---

## 9. Geometry subpackage

Per-element scalar/vector properties of the embedded mesh. Free
functions; ComputeContext caches what it needs internally.

```cpp
namespace nxr::compute::geometry::face {
  Eigen::VectorXd                                             areas    (const ComputeContext& ctx);  // F
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>   centroids(const ComputeContext& ctx);  // F × 3
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>   normals  (const ComputeContext& ctx);  // F × 3
  FaceFrames                                                  frames   (const ComputeContext& ctx);
}

namespace nxr::compute::geometry::vertex {
  enum class NormalEstimator { Uniform, AreaWeighted, AngleWeighted, MeanCurvatureWeighted };

  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
                  normals    (const ComputeContext& ctx,
                              NormalEstimator estimator = NormalEstimator::AngleWeighted);
  Eigen::VectorXd dualAreas  (const ComputeContext& ctx);          // V — Voronoi by default

  struct Curvatures {
    Eigen::VectorXd                                             gaussian, mean, kMin, kMax;     // V each
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>   principalDir;                   // V × 3
  };
  Curvatures      curvatures (const ComputeContext& ctx);
}

namespace nxr::compute::geometry::edge {
  Eigen::VectorXd lengths     (const ComputeContext& ctx);   // E
  Eigen::VectorXd cotanWeights(const ComputeContext& ctx);   // E
}
```

---

## 10. Topology subpackage

Halfedge structure and adjacency, all derived from the
`ComputeContext`'s internal halfedge mesh.

```cpp
namespace nxr::compute::topology {
  struct FaceAdjacency {
    Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>   neighbors;          // F × 3, -1 on boundary
    Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>   oppositeVertex;     // F × 3
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor>   neighborLocalEdge;  // F × 3 (0/1/2)
  };
  FaceAdjacency adjacency(const ComputeContext& ctx);

  struct Halfedge {
    int nHalfedges;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>   tailVertex, headVertex;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>   face, next, prev, twin;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>   edge;                    // undirected edge index
    Eigen::Matrix<uint8_t,  Eigen::Dynamic, 1>   isBoundary;
  };
  Halfedge halfedge(const ComputeContext& ctx);

  // 1-ring neighbors of a vertex (variable-size; CSR layout)
  struct VertexNeighbors {
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1>  offsets;   // V + 1
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1>  neighbors; // total degree sum
  };
  VertexNeighbors vertexNeighbors(const ComputeContext& ctx);
}
```

---

## 11. Solve subpackage — RAII solvers with cached factors

Long-lived solvers as classes; one cached factorization, many right-
hand sides.

```cpp
namespace nxr::compute::solve {
  class PoissonSolver {
  public:
    enum class Method { Pinned, Screened };
    enum class Input  { RHS, Density };

    struct Options {
      Method      method        = Method::Pinned;
      Input       input         = Input::RHS;
      VertexIndex pinnedVertex { 0 };       // Method::Pinned
      double      alpha         = 0.0;      // Method::Screened
    };

    PoissonSolver(const ComputeContext& ctx, const Options& opts = {});
    ~PoissonSolver();

    PoissonSolver(PoissonSolver&&)            noexcept;
    PoissonSolver& operator=(PoissonSolver&&) noexcept;
    PoissonSolver(const PoissonSolver&)            = delete;
    PoissonSolver& operator=(const PoissonSolver&) = delete;

    Eigen::VectorXd solve(Eigen::Ref<const Eigen::VectorXd> rhs) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };

  class HeatSolver {
    /* analogous; constructor takes time step dt */
  };

  class HeatDistanceSolver {
    /* analogous; geodesic distance via heat method */
  };
}
```

Move-only, RAII, const-correct `solve()`. Embind exposes them as JS
classes with `.delete()` — same pattern as `ComputeContext`.

`mutable` members guarded by appropriate locking are avoided —
factors live in the solver class, not in `ComputeContext`.

---

## 12. Query subpackage

```cpp
namespace nxr::compute::query {
  // Geodesic distance from one or more sources via heat method.
  // Reuses any cached Cholesky factor of the cotan Laplacian.
  Eigen::VectorXd geodesicDistance(const ComputeContext& ctx,
                                   std::span<const VertexIndex> sources);

  // Geodesic path between two vertices via flip-out.
  // Throws Error on disconnected vertices.
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
                  geodesicPath(const ComputeContext& ctx,
                               VertexIndex vA, VertexIndex vB);

  // Streamline tracing through a face vector field.
  struct StreamlinesOptions {
    int    numSeeds = 50;
    double stepCoef = 0.15;     // step length, in units of mean edge length
    int    maxSteps = 1000;
    uint64_t seed   = 0;        // RNG seed for seed placement
  };
  struct StreamlinesResult {
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>   positions;     // 2N × 3 endpoint pairs
    int segmentCount;
  };
  StreamlinesResult traceStreamlines(
      const ComputeContext& ctx,
      Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> faceField,
      const StreamlinesOptions& opts = {},
      std::stop_token stop = {},
      ProgressObserver progress = {});

  // Isoline extraction at numLevels evenly-spaced contour values.
  struct IsolinesOptions { int numLevels = 20; double minValue = 0.0; double maxValue = 0.0; };
  struct IsolinesResult  {
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>   positions;
    int segmentCount;
  };
  IsolinesResult isolines(const ComputeContext& ctx,
                          Eigen::Ref<const Eigen::VectorXd> scalar,
                          const IsolinesOptions& opts = {});
}
```

---

## 13. Transform subpackage

```cpp
namespace nxr::compute::transform {
  // BFF parametrization (open meshes only).
  // Throws Error(OpenMeshRequired) on closed meshes.
  Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>
                  bff(const ComputeContext& ctx,
                      std::stop_token stop = {},
                      ProgressObserver progress = {});

  // Whitney interpolation: edge 1-form → per-face 3D vector.
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
                  whitney(const ComputeContext& ctx,
                          Eigen::Ref<const Eigen::VectorXd> oneForm);

  // Hodge decomposition of a 1-form ω on edges.
  struct HodgeDecomposition {
    Eigen::VectorXd                                              exactPotential;       // V (α)
    Eigen::VectorXd                                              coExactPotentialV;    // V (β, vertex-averaged)
    Eigen::VectorXd                                              gamma;                // E (harmonic)
    Eigen::VectorXd                                              dAlpha, deltaBeta;    // E
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>    omegaVectors;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>    dAlphaVectors;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>    deltaBetaVectors;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>    gammaVectors;
  };
  HodgeDecomposition hodgeDecompose(const ComputeContext& ctx,
                                    Eigen::Ref<const Eigen::VectorXd> omega);
}
```

---

## 14. Connection subpackage

```cpp
namespace nxr::compute::connection {
  // Trivial connection: a globally consistent tangent vector field on
  // the surface, with prescribed singularities. Singularity values must
  // satisfy Gauss-Bonnet (Σ σ = χ) for a smooth field.
  struct DirectionField {
    Eigen::VectorXd                                              connections;          // E (per-edge angles)
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>    directionVectors;     // F × 3
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>    orthogonalVectors;    // F × 3
    int    eulerCharacteristic;
    bool   gaussBonnetSatisfied;
  };
  DirectionField trivialConnection(const ComputeContext& ctx,
                                   std::span<const VertexIndex> singularityVertices,
                                   std::span<const double>      singularityValues);
}
```

---

## 15. Health subpackage — validation and repair

### 15.1 Pre-construction validation

```cpp
namespace nxr::compute::health {
  enum class CheckLevel { Quick, Standard, Full };

  enum class IssueCode {
    NonManifoldEdge,
    NonManifoldVertex,
    DuplicateFace,
    DegenerateTriangle,
    UnreferencedVertex,
    WrongWinding,
    BoundaryEdge,
  };

  enum class Severity { Warn, Error };

  struct Issue {
    IssueCode code;
    Severity  severity;
    int       count;
    int32_t   sampleIndex = -1;   // -1 means "no specific element"
  };

  struct Summary {
    int  nV, nE, nF;
    int  eulerCharacteristic;
    bool isClosed;
  };

  struct CheckOptions {
    CheckLevel level             = CheckLevel::Standard;
    bool       requireManifold   = true;
    bool       requireOriented   = true;
    bool       requireOutward    = false;
    bool       requireClosed     = false;
  };

  struct CheckReport {
    bool                ok;
    Severity            severity;     // worst issue severity, or Warn if ok
    Summary             summary;
    std::vector<Issue>  issues;
  };

  // Static / non-throwing — validate before attempting to build a
  // ComputeContext. Same internal validator, just doesn't throw.
  CheckReport check(std::span<const double>  vertices,
                    std::span<const int32_t> faces,
                    const CheckOptions& opts = {});
}
```

### 15.2 Repair operations

```cpp
namespace nxr::compute::health {
  enum class RepairOp {
    WeldDuplicateVertices,
    OrientConsistently,
    OrientOutward,
    RemoveDegenerateFaces,
    RemoveUnreferencedVertices,
    RemoveDuplicateFaces,
    SplitNonManifoldEdges,
  };

  struct RepairOptions {
    std::vector<RepairOp> ops;
    double weldTolerance = 1e-6;
  };

  struct RepairResult {
    std::vector<double>  vertices;   // V' × 3 row-major
    std::vector<int32_t> faces;      // F' × 3 row-major
    int  weldedVertices = 0;
    int  flippedFaces   = 0;
    int  removedFaces   = 0;
    int  removedVertices = 0;
  };

  RepairResult repair(std::span<const double>  vertices,
                      std::span<const int32_t> faces,
                      const RepairOptions& opts);
}
```

Returns plain `std::vector` rather than Eigen matrices — repair is a
mesh-edit operation, not a numerical one, and the result is destined
to be fed back into the consumer's mesh pipeline.

---

## 16. Tier summary — implementation priority

| Tier | C++ feature | Subpackage | Notes |
|---|---|---|---|
| 1 | `std::stop_token` cancellation               | cross-cutting   | §5.1, §5.2 |
| 1 | `ProgressObserver`                           | cross-cutting   | §5.3 |
| 1 | Granular operations (split `precompute`)     | various         | §8.2 |
| 1 | `nxr::compute::eigen::solve` decoupled from ctx       | `eigen`         | §7.1 |
| 1 | Subpackage namespace reorganization          | all             | §2 |
| 2 | `ComputeContext` introspection methods       | `nxr-compute`           | §6 |
| 2 | `nxr::compute::health::check` non-throwing validation | `health`        | §15.1 |
| 2 | `Error` / `ErrorCode` structured errors   | cross-cutting   | §4 |
| 2 | Eigensolve hyperparameters + diagnostics     | `eigen`         | §7.1 |
| 3 | `nxr::compute::eigen::validateLoaded` (warm-load)     | `eigen`         | §7.2 |
| 3 | `nxr::compute::topology::adjacency`                   | `topology`      | §10 |
| 3 | `nxr::compute::health::repair`                        | `health`        | §15.2 |
| 3 | `nxr::compute::solve::PoissonSolver` etc.             | `solve`         | §11 |

Tier 1 unblocks the React Eigensolve composite component. Everything
else can follow incrementally without forcing JS-side rewrites.

---

## 17. Explicitly NOT requested (would be bloat)

- **Per-context memory tracking.** Consumers can read total WASM heap
  size themselves and decide eviction policy.
- **Asyncify-based callbacks from C++ to JS.** Would balloon the WASM
  bundle ~60%. The atomic-pointer `ProgressObserver` and
  `stop_token_from_atomic` give us cross-language signalling at zero
  size cost.
- **Full ComputeContext serialization.** Restoring a halfedge mesh +
  Cholesky factors from a blob is hairy. The cheaper proxy
  (`nxr::compute::eigen::validateLoaded` + `ComputeContext::fingerprint`)
  covers the high-value case.
- **State machines, observable patterns, event emitters.** Consumer
  concern.
- **Caching policy / LRU / IndexedDB integration.** Consumer concern.
- **Web Worker boilerplate / RPC framework.** Consumer concern.
- **Mesh I/O (OBJ, GLTF, FreeSurfer parsers).** `cxf-io` already
  exists for this and should not bloat the nxr-compute compute build.
- **`std::function`-based callbacks.** Allocate, cross FFI poorly,
  require Asyncify under WASM. The atomic-pointer alternative is
  strictly better for our use case.
- **Provenance metadata / serialization headers in result types.**
  Bindings that need this build it from call-site context plus the
  diagnostic fields nxr-compute returns.

---

## 18. JS / Embind binding chapter

This chapter documents how the C++ surface above maps through Embind to
the JavaScript consumers (`@nxr-compute/wasm`). It's a *binding contract*, not
a separate API design. C++ is the source of truth; JS receives a
faithful translation.

### 18.1 Namespaces → nested objects

```js
import { initNxrCompute } from '@nxr-compute/wasm'
const nxrCompute = await initNxrCompute()

// nxr-compute.ComputeContext         ← class
// nxr-compute.eigen.solve(...)        ← namespaced function
// nxr-compute.operators.laplaceBeltrami(...)
// nxr-compute.solve.PoissonSolver     ← class
// nxr-compute.health.check(...)
```

Embind doesn't natively support nested namespaces, so the binding
shell builds the nesting at module-init: `nxrCompute.eigen = { solve, … }`.
Cost: a few lines per namespace.

### 18.2 Strong indices → plain numbers

`VertexIndex`, `FaceIndex` etc. lose their type tag at the binding —
JS sees plain numbers. This is acceptable: the JS adapter
(`@nxr/nxr-compute-react`) can layer a TypeScript branded-type back on if
needed.

### 18.3 `std::span<const T>` → typed arrays

JS passes `Float64Array` / `Int32Array` / etc. Embind binds them as
heap-aliased views — zero copy on the way in.

### 18.4 Eigen matrices/vectors → typed arrays + shape metadata

Result struct fields of type `Eigen::MatrixXd` are exposed as
`{ data: Float64Array, rows, cols }` (for column-major) or as flat
`Float64Array` with documented shape (for `RowMajor` matrices).
Sparse matrices expose CSC triplets `{ indptr, indices, data }`.

### 18.5 `std::stop_token` → SAB-backed `Int32Array`

```js
const cancel = new Int32Array(new SharedArrayBuffer(4))
nxr-compute.eigen.solve(L, M, k, opts, cancel /* stop_token */, progress)

// Trigger from anywhere:
Atomics.store(cancel, 0, 1)
// → solve throws Error({ code: 'CANCELLED' }) within ~100ms
```

The binding accepts the typed array directly and bridges via
`nxr::compute::stop_token_from_atomic(...)` internally. Same SAB can host both
cancellation flag and progress slots — distinct offsets.

Requires the page to be served with COOP/COEP headers (the consumer's
responsibility).

### 18.6 `ProgressObserver` → SAB-backed `Int32Array` slots

```js
const progress = new Int32Array(new SharedArrayBuffer(16)) // [iter, total, residual×1e6, _pad]
nxr-compute.eigen.solve(L, M, k, opts, cancel, progress)

// Main thread polls:
const iter = Atomics.load(progress, 0)
const tot  = Atomics.load(progress, 1)
const res  = Atomics.load(progress, 2) / 1e6
```

### 18.7 `Error` → JS Error subclass with `.code`

```js
try {
  nxr-compute.eigen.solve(L, M, k)
} catch (e) {
  // e instanceof nxr-compute.Error
  // e.code === 'CANCELLED' | 'NON_MANIFOLD' | …  (string-named enumerator)
  // e.message, e.hint
}
```

Embind's `EMBIND_REGISTER_EXCEPTIONS` plus a small wrapper handles the
translation from `nxr::compute::Error` to a JS class.

### 18.8 RAII solvers → JS classes with `.delete()`

```js
const poisson = new nxrCompute.solve.PoissonSolver(ctx, { method: 'pinned' })
const phi1 = poisson.solve(rhs1)
const phi2 = poisson.solve(rhs2)
poisson.delete()
```

---

## 19. Open questions for the C++ side

1. **`ComputeContext::fingerprint()` algorithm.** SHA-256 of
   canonicalized vertex+face bytes is the safest. Faster alternatives
   (xxHash, BLAKE3) are fine if the consumer just wants a cache key
   and doesn't need cryptographic strength. Pick one and document.

2. **Eigen storage order.** Most nxr-compute result matrices currently use
   row-major to match three.js / NumPy expectations. Eigen's default
   is column-major. Either choose a consistent convention across all
   matrix-returning functions and document it, or expose both via
   templated `Storage` parameter. The current sketch uses row-major
   for `[F × 3]` / `[V × 3]` typed buffers (matches three.js
   `BufferAttribute` storage) and column-major for `[V × K]`
   eigenvectors (matches Eigen's natural layout for tall matrices —
   contiguous columns let `eigenvectors.col(k)` be a zero-copy view).

3. **`std::stop_token` polling cadence.** Once per outer Spectra
   iteration (~100 ms) is the cheapest. Faster polling (every 10 ms)
   improves cancel-latency at negligible cost. Pick a default
   constant; expose as a build-time tunable if needed.

4. **Error message stability.** `Error::what()` strings should be
   considered diagnostic-only (English, may change). `code()` is the
   stable contract. Document this so consumers don't pattern-match
   on `what()`.

5. **C++17 fallback.** If any deployment target stalls on C++17,
   `std::stop_token` and `std::span` need backports (gsl::span;
   custom stop_token shim). Confirm C++20 is the floor or specify
   the fallback path before implementation begins.
