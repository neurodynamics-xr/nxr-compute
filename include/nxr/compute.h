#pragma once

// nxr-compute — portable C++ math/compute library for cortical-flow,
// the NXR design system, and any other consumer that needs
// halfedge-mesh-based scientific computing. Targets one source,
// many shells (N-API addon, WASM/Embind, MATLAB MEX, CLI).
//
// Layout convention for matrix returns (hard rule, applies to
// every JS-facing binding shell — addon, WASM):
//
//   • [V × 3] / [F × 3] / [N × 3]  →  row-major flat: xyz xyz xyz …
//                                     matches three.js BufferAttribute
//                                     and NumPy `.reshape(-1, 3)`.
//   • [V × K] eigenvectors         →  row-major flat (vMajor): U[v*K+k]
//                                     matches the Zarr schema
//                                     (manifold/eigenmodes/eigenvectors
//                                     stored as [V, K]) and the GPU
//                                     spectral-synthesis access pattern
//                                     ("for each vertex, sum across modes").
//                                     C++ side keeps Eigen default
//                                     column-major storage; bindings
//                                     transpose at the flatten step.
//   • [T × V] activity time series →  row-major flat (frame-major):
//                                     frame t at data[t*V .. (t+1)*V].
//                                     matches the Zarr `recordings/.../
//                                     activity` schema.
//
// MEX is exempt from the JS flatten rule: a V×K Eigen matrix
// becomes a V×K MATLAB matrix (column-major in MATLAB's native
// storage) so MATLAB users get U(:,k) for mode k contiguously.
// See native/CLAUDE.md §11 for the binding contract.

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "nxr/errors.h"
#include "nxr/cancellation.h"
#include "nxr/progress.h"

// C++17 floor today. C++20 was the planned baseline (see Phase A
// design doc) but a vendored-Eigen 3.4 + Emscripten Clang + C++20
// template-deduction bug in geometry-central's BFF source forces
// us to stay at 17 until Phase B can address Eigen separately.
// Phase A's new primitives (Error, CancellationToken,
// ProgressObserver) intentionally avoid std::span and designated
// initializers so this floor is sufficient.

// Forward declarations for geometry-central types
namespace geometrycentral {
namespace surface {
class ManifoldSurfaceMesh;
class VertexPositionGeometry;
}
}

namespace nxr::compute {

// ── Compute Context ──────────────────────────────────────────
// Owns the geometry-central mesh and geometry objects.
// All compute functions operate on a context, which can be created
// once and reused across multiple calls.

class ComputeContext {
public:
    ComputeContext(const double* vertices, int nV,
                   const int32_t* faces, int nF);
    ~ComputeContext();

    // Non-copyable
    ComputeContext(const ComputeContext&) = delete;
    ComputeContext& operator=(const ComputeContext&) = delete;

    int nV() const;
    int nE() const;
    int nF() const;

    // Access to underlying geometry-central objects (for advanced use).
    // The mesh is a ManifoldSurfaceMesh — required for tangent-space-aware
    // features (curvature principal directions, BFF parametrization,
    // flip-out geodesics). Construction throws if the input is non-manifold;
    // cortical surfaces from FreeSurfer satisfy the constraint.
    geometrycentral::surface::ManifoldSurfaceMesh& mesh();
    geometrycentral::surface::VertexPositionGeometry& geometry();

private:
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> mesh_;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry_;
};

// ── Operator Assembly ────────────────────────────────────────

/**
 * Mass-matrix variant for the FEM L² inner product ⟨u, v⟩ = uᵀ M v.
 *
 * All three variants conserve total surface area (Σᵢⱼ Mᵢⱼ == totalArea)
 * and are symmetric. They differ in M's structure and consequently in
 * the eigenvalues / eigenvectors of the generalized problem K φ = λ M φ.
 *
 * - Voronoi (default): diagonal, M_ii = vertex's mixed Voronoi-barycentric
 *   dual area (Meyer et al. 2003). Robust on obtuse triangles. Matches
 *   geometry-central's vertexDualAreas. The default for backward-compat.
 *
 * - Barycentric: diagonal, M_ii = (Σ adjacent triangle areas) / 3. Pure
 *   lumped mass, no Voronoi correction. Cheaper to assemble; less
 *   geometrically accurate than Voronoi on irregular triangulations.
 *
 * - ConsistentFEM: full sparse, off-diagonal couplings between adjacent
 *   vertices. Per-triangle element matrix is (A_T / 12) · [[2 1 1][1 2 1][1 1 2]],
 *   from L² integrals of piecewise-linear hat functions. This is the
 *   canonical FEM mass — used by lapy and gptoolbox's `'full'` variant.
 *   Eigenvalues converge to the continuous Laplace–Beltrami spectrum
 *   faster than diagonal lumped variants on coarse meshes.
 */
enum class MassMatrixVariant {
    Voronoi,        // diagonal, mixed Voronoi-barycentric (Meyer)
    Barycentric,    // diagonal, area/3 per vertex
    ConsistentFEM   // sparse, off-diagonal couplings (full FEM mass)
};

// MeshOperators / DECOperators mix view and value semantics. Field
// names follow geometry-central's canonical naming for the underlying
// cached matrices (cotanLaplacian, vertexDualAreas, vertexNormals,
// d0/d1/hodge*), so the surface here is a 1-to-1 viewer over GC's
// cache where the concept exists. Variant-aware quantities (mass)
// keep a generic name because GC has no single matrix covering them.
//
//   * View fields (const-references into geometry-central's cached matrices):
//       MeshOperators::cotanLaplacian, all DECOperators fields. Bound by
//       assembleMeshOperators / assembleDECOperators after the relevant
//       require* call pins them on the geometry.
//
//   * Owned field:
//       MeshOperators::mass — variant-dependent (Voronoi / Barycentric /
//       ConsistentFEM); non-Voronoi variants don't have a single
//       geometry-central matrix to view, so mass is materialised by
//       value regardless of variant for uniform semantics.
//
// Lifetime contract: bindings must keep the owning ComputeContext alive
// for the entire lifetime of any MeshOperators / DECOperators they hold,
// and must NOT call unrequire* on the underlying geometry while the
// structs are alive. ContextHolder / ContextWrapper enforce this by
// keeping the operator structs and the context together.
//
// Because of the reference fields, both structs are copy-constructible
// (refs rebind to the same targets) but not copy-assignable. The
// bindings hold them via shared_ptr / unique_ptr and never reassign.

struct MeshOperators {
    MeshOperators(const Eigen::SparseMatrix<double>& cotanLaplacian_,
                  Eigen::SparseMatrix<double> mass_,
                  MassMatrixVariant massVariant_,
                  Eigen::VectorXd vertexDualAreas_,
                  Eigen::MatrixXd vertexNormals_,
                  double totalArea_)
      : cotanLaplacian(cotanLaplacian_),
        mass(std::move(mass_)),
        massVariant(massVariant_),
        vertexDualAreas(std::move(vertexDualAreas_)),
        vertexNormals(std::move(vertexNormals_)),
        totalArea(totalArea_) {}

    const Eigen::SparseMatrix<double>& cotanLaplacian;  // view: GC's cotangent Laplacian
    Eigen::SparseMatrix<double> mass;                   // owned: depends on massVariant
    MassMatrixVariant massVariant;                      // which variant produced `mass`
    Eigen::VectorXd vertexDualAreas;                    // [nV] mixed Voronoi (variant-independent)
    Eigen::MatrixXd vertexNormals;                      // [nV, 3] vertex normals
    double totalArea;
    // nV / nE / nF intentionally omitted — read from the owning
    // ComputeContext (ctx.nV(), ctx.nE(), ctx.nF()) to avoid duplicating
    // mesh metadata that's already canonical on the context.
};

/** Assemble operators from a context. Default mass variant is Voronoi
 *  (matches the previous behavior and the cortical-flow / mesh-tests
 *  consumers). */
MeshOperators assembleMeshOperators(
    ComputeContext& ctx,
    MassMatrixVariant variant = MassMatrixVariant::Voronoi);

/** Convenience overload: creates context and assembles operators in one call. */
MeshOperators assembleMeshOperators(
    const double* vertices, int nV,
    const int32_t* faces, int nF,
    MassMatrixVariant variant = MassMatrixVariant::Voronoi
);

/** String → enum helper for binding shells (WASM, addon).
 *  Accepts "voronoi", "barycentric", "full". Throws Error(InvalidInput)
 *  on unknown. Case-sensitive (matches the MATLAB +bct convention). */
MassMatrixVariant parseMassMatrixVariant(const std::string& name);

// ── Connection Laplacian ─────────────────────────────────────
//
// Connection Laplacian on a chosen domain (vertex / face / edge).
// The operator that drives smoothest n-RoSy direction fields,
// parallel-transport energies, and tangent-bundle eigendecompositions.
//
// Mathematically the operator is complex Hermitian on the chosen
// domain (V, F, or E entries), built from cotangent weights and
// Levi-Civita transport rotations raised to the nSym power. The
// `Real2N` format (default) lowers each complex entry a + bi to the
// 2×2 real block [[a, -b], [b, a]], producing a 2N×2N symmetric real
// matrix that drops directly into nxr-compute's existing real-only
// eigensolver (`solveEigenmodes`) when paired with a block-diagonal
// real mass matrix `blkdiag(M, M)`. The smallest eigenpair then
// reproduces the smoothest n-direction field that
// `computeSmoothest{Vertex,Face}DirectionField` returns internally.
//
// `Complex` format leaves the matrix as complex Hermitian; bindings
// surface real and imaginary triplet arrays separately. Useful when
// callers want to apply a complex-Hermitian eigensolver themselves.
//
// MATLAB reference: none in `+bct` today (the existing direction-field
// reference uses trivial connections; connection-Laplacian-based
// smoothest fields delegate to geometry-central). When a MATLAB
// oracle harness lands, mirror it under
// `+bct.+manifold.+operator.connectionLaplacian`.

enum class ConnectionDomain {
    Vertex,                  // V × V (intrinsic vertex CL)
    Face,                    // F × F (intrinsic face CL — geometry-central
                             // has FIXME on face weights, propagated as a
                             // doc caveat below)
    EdgeCrouzeixRaviart      // E × E (intrinsic Crouzeix-Raviart CL)
};

enum class ConnectionLaplacianFormat {
    Real2N,                  // 2N × 2N symmetric real (default)
    Complex                  // N × N complex Hermitian, surfaced as
                             // parallel real/imag triplet arrays at
                             // the binding edge
};

/** Options bag for `assembleConnectionLaplacian`. Each field is
 *  defaulted; a value-initialized struct produces the default
 *  vertex / nSym=1 / regularization=1e-8 / Real2N variant. */
struct ConnectionLaplacianOptions {
    ConnectionDomain domain          = ConnectionDomain::Vertex;
    int    nSym                      = 1;       // 1=vector, 2=line, 4=cross
    double regularization            = 1e-8;    // additive ε·I shift
    ConnectionLaplacianFormat format = ConnectionLaplacianFormat::Real2N;
};

/** Result of `assembleConnectionLaplacian`. Exactly one of `K_real`
 *  / `K_complex` is populated, selected by `options.format`. */
struct ConnectionLaplacian {
    Eigen::SparseMatrix<double>                K_real;     // populated iff format == Real2N
    Eigen::SparseMatrix<std::complex<double>>  K_complex;  // populated iff format == Complex
    int  baseDim   = 0;                                    // V / F / E
    int  outputDim = 0;                                    // 2 * baseDim (Real2N) or baseDim (Complex)
    ConnectionDomain domain          = ConnectionDomain::Vertex;
    int  nSym                        = 1;
    double regularization            = 1e-8;
    ConnectionLaplacianFormat format = ConnectionLaplacianFormat::Real2N;
};

/** Assemble the connection Laplacian on the chosen domain.
 *
 *  Walks halfedges / faces / edges directly using
 *  `geometry.edgeCotanWeights`, `geometry.transportVectorsAlongHalfedge`,
 *  `geometry.transportVectorsAcrossHalfedge`, and applies `.pow(nSym)` to
 *  each transport rotation (mirrors the file-static helpers in
 *  `geometry-central/src/surface/direction_fields.cpp` that drive
 *  `computeSmoothest{Vertex,Face}DirectionField`).
 *
 *  Throws `Error(InvalidInput)` for `nSym <= 0`. The Crouzeix-Raviart
 *  variant requires triangle meshes (geometry-central's own assertion).
 *
 *  Note: the face variant inherits geometry-central's FIXME on face
 *  weights (uniform 1.0). The numerical answer is still well-defined
 *  for nRoSy energy minimization; revisit if a more principled face
 *  weight emerges upstream. */
ConnectionLaplacian assembleConnectionLaplacian(
    ComputeContext& ctx,
    const ConnectionLaplacianOptions& opts = {}
);

/** String → enum helpers for binding shells (WASM, addon).
 *  Domain accepts "vertex", "face", "edge". Format accepts "real2N"
 *  / "real" / "complex". Throws Error(InvalidInput) on unknown. */
ConnectionDomain          parseConnectionDomain(const std::string& name);
ConnectionLaplacianFormat parseConnectionLaplacianFormat(const std::string& name);

// ── DEC Operators ────────────────────────────────────────────

struct DECOperators {
    DECOperators(const Eigen::SparseMatrix<double>& d0_,
                 const Eigen::SparseMatrix<double>& d1_,
                 const Eigen::SparseMatrix<double>& hodge0_,
                 const Eigen::SparseMatrix<double>& hodge1_,
                 const Eigen::SparseMatrix<double>& hodge2_,
                 const Eigen::SparseMatrix<double>& hodge1Inverse_)
      : d0(d0_), d1(d1_), hodge0(hodge0_), hodge1(hodge1_),
        hodge2(hodge2_), hodge1Inverse(hodge1Inverse_) {}

    // All fields are views into geometry-central's cached matrices —
    // see the comment block above MeshOperators for the lifetime
    // contract. dec.hodge0 is the same matrix as ops.mass (both
    // are the lumped Voronoi mass).
    const Eigen::SparseMatrix<double>& d0;      // vertex → edge
    const Eigen::SparseMatrix<double>& d1;      // edge → face
    const Eigen::SparseMatrix<double>& hodge0;  // vertex → vertex (diagonal)
    const Eigen::SparseMatrix<double>& hodge1;  // edge → edge (diagonal)
    const Eigen::SparseMatrix<double>& hodge2;  // face → face (diagonal)
    const Eigen::SparseMatrix<double>& hodge1Inverse;
};

DECOperators assembleDECOperators(ComputeContext& ctx);

// ── Factor Cache ─────────────────────────────────────────────
//
// Pre-factored Cholesky / LU decompositions of mesh-derived
// matrices, cached across compute calls. Each factor is built
// lazily on first request and reused on every subsequent call
// for the lifetime of the owning ComputeContext.
//
// Why this exists: mesh-derived sparse factorizations cost
// O(n^1.5) to build but only O(nnz) per back-substitution. Each
// solver call (Poisson, Hodge α-solve, direction-field) hits
// the same matrix; without caching we redo that O(n^1.5) work
// on every click and on every frame of any future per-frame
// pipeline. Both geometry-processing-js and geometry-central
// pre-factor and reuse for this reason.
//
// Bit-for-bit contract: each cached factor is built by the same
// matrix-assembly + factorization sequence the previous inline
// solvers used (same regularization ε, same coefficient order,
// same Eigen solver type). Refactored callers must produce
// numerically identical output to the pre-cache implementation.
//
// Lifetime: owned by ContextHolder in the addon; destroyed when
// the JavaScript handle releases (i.e. when the mesh changes).
//
// Not cached: the eigensolver's (K - σM) factor lives inside
// Spectra::SymShiftInvert and is discarded per call. Caching
// it would require replacing Spectra's ShiftInvertOp with a
// custom one — out of scope for v1.
//
// Perf options compatible with the modularity rule (CLAUDE.md §10):
// AVX2/AVX-512 compile flags (the build enables these on x86_64),
// Eigen's separate analyzePattern() + factorize() for matrices
// that share sparsity pattern, multi-RHS batched solves, and GPU
// triangular back-substitution via TSL for hot per-frame paths.
// Not on the table: CHOLMOD / SuiteSparse / MKL / PARDISO — they
// break the WASM and MEX targets.

class CholeskyCache {
public:
    CholeskyCache() = default;
    CholeskyCache(const CholeskyCache&) = delete;
    CholeskyCache& operator=(const CholeskyCache&) = delete;

    /** Diagonal regularization added to every factored matrix.
     *  Identical to the constant the pre-cache inline solvers used. */
    static constexpr double kRegularization = 1e-8;

    /** LLT of (K + ε·I) for any symmetric PSD K. The cache is keyed
     *  by lifetime, not content — the *first* matrix passed in is the
     *  one factored, and subsequent calls return that factor regardless
     *  of what's passed. Use a fresh CholeskyCache per matrix. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& laplacian(
        const Eigen::SparseMatrix<double>& K);

    /** Convenience overload: factors ops.cotanLaplacian. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& laplacian(
        const MeshOperators& ops) {
        return laplacian(ops.cotanLaplacian);
    }

    /** LLT of (d0ᵀ ★₁ d0 + ε·I).
     *  Used by both Hodge α-solve and direction-field. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& hodgeExact(
        const DECOperators& dec);

    /** LU of (d1 ★₁⁻¹ d1ᵀ + ε·I), used by Hodge β-solve.
     *  B is symmetric PSD after regularization, so SimplicialLLT
     *  would also work; SparseLU preserved for bit-for-bit
     *  equivalence with the pre-cache implementation. */
    const Eigen::SparseLU<Eigen::SparseMatrix<double>>& hodgeCoExact(
        const DECOperators& dec);

private:
    std::unique_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>> laplacian_;
    std::unique_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>> hodgeExact_;
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> hodgeCoExact_;
};

// ── Eigensolver ──────────────────────────────────────────────

struct EigenResult {
    Eigen::MatrixXd eigenvectors;  // [nV, k] columns are eigenvectors
    Eigen::VectorXd eigenvalues;   // [k] sorted ascending
    int k;
    int nConverged;
};

/**
 * Solve generalized eigenvalue problem: K * φ = λ * M * φ
 * for the k smallest eigenvalues.
 * Matches MATLAB: eigs(K, M, k, 'SM')
 *
 * @param sigma     Shift for shift-invert mode. Values near 0
 *                  (e.g. -1e-8) target the lowest eigenmodes.
 *                  The solver pre-factors (K - σM) once, then
 *                  each IRAM iteration is a cheap forward/back-
 *                  substitution. This is dramatically faster than
 *                  standard mode for the smallest eigenvalues of
 *                  large sparse systems.
 * @param cancel    Optional cancellation token. The solver polls
 *                  cancel.requested() once per Spectra perform_op
 *                  call; on true, throws Error(Cancelled). Cancel
 *                  latency is roughly one SpMV ≈ a few ms on
 *                  cortical-sized meshes. Default: never cancelled.
 * @param progress  Optional observer of solver progress. The
 *                  `iteration` slot is incremented once per Spectra
 *                  perform_op (proxy for solver work units);
 *                  `totalIterations` is set to a high-water bound.
 *                  Default: no-op (no slots set).
 *
 * Throws Error(EigensolveInvalidK) on k < 1 or k > N - 1.
 * Throws Error(Cancelled)          on cancel.requested().
 * Throws Error(EigensolveNotConverged) on Spectra failure with
 *                                          no partial result.
 */
EigenResult solveEigenmodes(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int k,
    double sigma = -1e-8,
    const CancellationToken& cancel = {},
    const ProgressObserver& progress = {}
);

// ── Normalization ────────────────────────────────────────────

/** M-orthonormalize: enforce U' * M * U = I via Cholesky whitening. */
Eigen::MatrixXd normalizeEigenmodes(
    const Eigen::MatrixXd& U,
    const Eigen::SparseMatrix<double>& M
);

/** Remove the DC mode (eigenvalue closest to zero). */
EigenResult removeDC(const EigenResult& result);

// ── Poisson Solver ───────────────────────────────────────────

/**
 * Solve the scalar Poisson equation: K * φ = -M * (ρ - ρ̄)
 * where ρ̄ is the M-weighted mean of ρ.
 *
 * Uses cache.laplacian(K) for the LLT factor of (K + ε·I), so
 * repeated calls with the same cache skip the factorization step.
 *
 * Agnostic in K and M: works on cotangent Laplacians + Voronoi mass
 * (mesh case), graph Laplacians + node-weight diagonals (graph case),
 * or any SPD pair. The total mass `Σᵢⱼ Mᵢⱼ` plays the role of the
 * "totalArea" normalization for the weighted mean.
 *
 * @param K           SPD stiffness matrix (Laplacian)
 * @param M           SPD mass / inner-product matrix (typically diagonal)
 * @param cache       factor cache (one per K)
 * @param densityMap  sparse density: index → density value
 * @return            harmonic potential φ
 */
Eigen::VectorXd solvePoisson(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    CholeskyCache& cache,
    const std::map<int, double>& densityMap
);

/** Convenience overload: forwards to (K, M) using ops.cotanLaplacian, ops.mass. */
inline Eigen::VectorXd solvePoisson(
    const MeshOperators& ops,
    CholeskyCache& cache,
    const std::map<int, double>& densityMap
) {
    return solvePoisson(ops.cotanLaplacian, ops.mass, cache, densityMap);
}

// ── Geodesic Distance (Heat Method) ──────────────────────────
//
// Stateful solver wrapping geometry-central's HeatMethodDistanceSolver,
// matching the lifetime model of VectorHeatSolver / SignedHeatSolver.
// The inner solver pre-factors both Cholesky systems (mean-curvature
// flow `M + tA` and Laplacian `A`) at construction; subsequent
// computeDistance() calls only do back-substitution.
//
// Without this caching we redo two factorizations on every call —
// see commit history for the perf regression discovered when the
// pre-cache version was constructing the solver inline. Bench
// numbers in bench/REPORT.md document the warm/cold gap before the
// fix landed.

class HeatGeodesicSolverImpl;
class HeatGeodesicSolver {
public:
    explicit HeatGeodesicSolver(ComputeContext& ctx, double tCoef = 1.0);
    ~HeatGeodesicSolver();
    HeatGeodesicSolver(const HeatGeodesicSolver&) = delete;
    HeatGeodesicSolver& operator=(const HeatGeodesicSolver&) = delete;

    HeatGeodesicSolverImpl& impl();

private:
    std::unique_ptr<HeatGeodesicSolverImpl> impl_;
};

/**
 * Compute geodesic distances from source vertices via the heat method.
 * Reuses the pre-factored solver — first call pays the factor, subsequent
 * calls are back-substitution only.
 *
 * @param solver          stateful heat-geodesic solver
 * @param sourceVertices  list of source vertex indices
 * @return                per-vertex geodesic distances
 */
Eigen::VectorXd computeGeodesicDistance(
    HeatGeodesicSolver& solver,
    const std::vector<int>& sourceVertices
);

// ── Hodge Decomposition ──────────────────────────────────────

struct HodgeResult {
    // Scalar potentials (per vertex)
    Eigen::VectorXd exactPotential;      // α
    Eigen::VectorXd coExactPotentialF;   // β (per face)
    Eigen::VectorXd coExactPotentialV;   // β averaged to vertices
    Eigen::VectorXd combinedPotential;   // α + β
    Eigen::VectorXd omega;               // input 1-form on edges

    // 1-form components of the decomposition (all [nE])
    Eigen::VectorXd dAlpha;              // exact 1-form: d(α)
    Eigen::VectorXd deltaBeta;           // co-exact 1-form: δ(β)
    Eigen::VectorXd gamma;               // harmonic 1-form: ω - dα - δβ

    // Face-centered vector fields via Whitney interpolation, each [nF, 3]
    Eigen::MatrixXd omegaVectors;
    Eigen::MatrixXd dAlphaVectors;
    Eigen::MatrixXd deltaBetaVectors;
    Eigen::MatrixXd gammaVectors;
};

/**
 * Hodge/Helmholtz decomposition of a 1-form ω:
 *   ω = dα + δβ + γ
 *
 * Given random ω, returns:
 *   - α: scalar potential (vertex-based) — gradient part
 *   - β: scalar potential (face-based, averaged to vertices) — curl part
 *
 * @param dec    DEC operators (d0, d1, hodge1, hodge1Inverse)
 * @param omega  input 1-form on edges (random or user-supplied)
 */
HodgeResult hodgeDecompose(
    ComputeContext& ctx,
    const DECOperators& dec,
    CholeskyCache& cache,
    const Eigen::VectorXd& omega
);

/** Generate a random 1-form on edges. */
Eigen::VectorXd generateRandomOmega(int nE, unsigned int seed = 42);

// ── Field Generators ─────────────────────────────────────────
//
// Analytic field synthesis. Mirrors the +bct.+field.+generate.*
// family from the MATLAB reference toolbox; the cortical-flow
// solvers (Hodge, Poisson, gradient flow) are source-agnostic and
// consume fields produced here interchangeably with real activity
// data loaded from Zarr.
//
// The validation pattern (see test_field_generators.cpp) is the
// geometry-processing-js demo pattern: synthesize ω from known
// α / β / γ contributions, run hodgeDecompose, verify the
// algorithm recovers what was put in.

/** Sparse delta vector: zeros everywhere except the listed source
 *  vertices, which take the supplied values. Useful as an initial
 *  condition for heat-method / Poisson workflows. */
Eigen::VectorXd generateDelta(int nV, const std::map<int, double>& sources);

/** Random vertex scalar field, uniform in [-1, 1]. Reproducible by
 *  seed. Used as the α (gradient potential) in decomposed-1-form
 *  synthesis. */
Eigen::VectorXd generateRandomVertexScalar(int nV, unsigned int seed = 42);

/** Random face scalar field, uniform in [-1, 1]. Reproducible by
 *  seed. Used as the β (curl potential) in decomposed-1-form
 *  synthesis. */
Eigen::VectorXd generateRandomFaceScalar(int nF, unsigned int seed = 42);

/** Extract the k-th eigenmode from a precomputed EigenResult as a
 *  vertex scalar field. */
Eigen::VectorXd generateEigenmodeField(const EigenResult& eig, int k_index);

/** Synthesize a 1-form on edges by combining independently-random
 *  exact and co-exact contributions:
 *
 *      ω = α_strength · (d0 · α_rand) + β_strength · (★₁⁻¹ d1ᵀ · β_rand)
 *
 *  where α_rand is a random vertex scalar (seed) and β_rand is a
 *  random face scalar (seed + 1). The Hodge decomposition of ω
 *  must recover the same dα and δβ within solver tolerance — this
 *  is the validation contract for hodgeDecompose.
 *
 *  γ_strength is reserved for the harmonic component but ignored in
 *  v1; harmonic basis computation is deferred (cortical meshes are
 *  genus 3-5 due to ventricle topology, so the harmonic space is
 *  non-trivial in principle, but its construction needs its own
 *  primitive and isn't required for the gpjs-style validation). */
Eigen::VectorXd generateRandomDecomposed1Form(
    const DECOperators& dec,
    int nV, int nE, int nF,
    double alphaStrength,
    double betaStrength,
    double gammaStrength,
    unsigned int seed = 42
);

// ── Time-Varying Field Generators ────────────────────────────
//
// Spectral-evolution generators that produce [T, V] float32 arrays
// suitable for the Timeline / activity slot. Both reuse precomputed
// eigenmodes — no per-frame Cholesky needed, which is the spectral
// method's defining feature: O(K) per timestep regardless of nV.
//
// Storage layout: Eigen::MatrixXf with shape (T, V), column-major
// (Eigen default). Frame ti is `output.row(ti)`. When flattened
// row-major to a Float32Array, this matches the [T, V] convention
// used by the Zarr `recordings/.../activity` schema.
//
// Both functions assume the eigenmodes are M-orthonormalized — i.e.
// the caller has already run normalizeEigenmodes() on solveEigenmodes
// output. Failure to do so silently produces wrong amplitudes.

/** Heat diffusion via spectral evolution.
 *
 *  Solves  ∂u/∂t = -α K u  on the domain, returning the full time
 *  series. M-weighted spatial mean is preserved exactly across all
 *  frames (heat equation invariant); only the zero-mean part evolves
 *  spectrally:
 *
 *      u(t) = u_mean + Σₖ ⟨u₀ - u_mean, φₖ⟩_M · exp(-α·t·λₖ) · φₖ
 *
 *  The DC mode (if present in `eig`) is harmless — its λ ≈ 0 means
 *  exp(-αtλ) ≈ 1 and it just reproduces the mean offset.
 *
 *  Agnostic in M: works on Voronoi mass (mesh) or node-weight
 *  diagonals (graph). M-orthonormality of `eig.eigenvectors`
 *  is the caller's responsibility regardless.
 *
 *  @param M          mass / inner-product matrix used for projection
 *  @param eig        precomputed eigenmodes (M-orthonormal, sorted ascending)
 *  @param u0         initial condition [n]
 *  @param timesteps  T values of t (any non-negative reals)
 *  @param alpha      diffusion coefficient
 *  @return           [T, n] float32 time series, frame-major
 */
Eigen::MatrixXf generateHeatDiffusion(
    const Eigen::SparseMatrix<double>& M,
    const EigenResult& eig,
    const Eigen::VectorXd& u0,
    const std::vector<double>& timesteps,
    double alpha = 1.0
);

/** Convenience overload: forwards to the (M, …) form using ops.mass. */
inline Eigen::MatrixXf generateHeatDiffusion(
    const MeshOperators& ops,
    const EigenResult& eig,
    const Eigen::VectorXd& u0,
    const std::vector<double>& timesteps,
    double alpha = 1.0
) {
    return generateHeatDiffusion(ops.mass, eig, u0, timesteps, alpha);
}

/** Damped wave field via per-mode oscillation.
 *
 *  Synthesizes a superposition of independently-damped harmonic
 *  oscillators on the mesh:
 *
 *      u(t,v) = Σₘ aₘ · exp(-γₘ·t) · cos(√λ_{kₘ}·t + φₘ) · φ_{kₘ}(v)
 *
 *  The angular frequency of each mode is √λ_{kₘ} (acoustic relation
 *  between Laplace eigenvalues and wave speeds). Useful for
 *  illustrative time-varying fields and for studying mode-mixing
 *  visualizations.
 *
 *  @param eig          precomputed eigenmodes (M-orthonormal)
 *  @param modeIndices  which eigenmodes to excite (size M)
 *  @param amplitudes   per-mode amplitude aₘ (size M)
 *  @param dampings     per-mode damping rate γₘ ≥ 0 (size M)
 *  @param phases       per-mode phase φₘ in radians (size M)
 *  @param timesteps    T values of t
 *  @return             [T, nV] float32 time series, frame-major
 */
Eigen::MatrixXf generateDampedWave(
    const EigenResult& eig,
    const std::vector<int>& modeIndices,
    const std::vector<double>& amplitudes,
    const std::vector<double>& dampings,
    const std::vector<double>& phases,
    const std::vector<double>& timesteps
);

// ── Curvatures ───────────────────────────────────────────────

struct CurvatureResult {
    Eigen::VectorXd gaussian;        // K per vertex (angle defect)
    Eigen::VectorXd mean;            // H per vertex
    Eigen::VectorXd kMin;            // κ_min per vertex
    Eigen::VectorXd kMax;            // κ_max per vertex
    Eigen::MatrixXd principalDirMax; // [nV, 3] max principal direction lifted to 3D
};

CurvatureResult computeCurvatures(ComputeContext& ctx);

// ── Normal Estimators ────────────────────────────────────────

enum class NormalType {
    AngleWeighted,    // default (matches gpjs "Tip Angle Weighted")
    AreaWeighted,     // weighted by face area
    EqualWeighted,    // equal weighting (normalized sum)
    SphereInscribed,  // (u × v) / (|u|² |v|²) per corner
    MeanCurvature,    // cotangent-weighted (Laplace-Beltrami)
    GaussCurvature    // dihedral angle / edge length weighted
};

/** Compute per-vertex normals using a given estimator.
 *  Returns [nV * 3] flat array (row-major). */
Eigen::MatrixXd computeVertexNormals(ComputeContext& ctx, NormalType type);

// ── Face Frames ──────────────────────────────────────────────

/** Per-face orthonormal tangent frame.
 *
 *  Each face f has an in-plane orthonormal basis (e1, e2) and a
 *  normal n, all 3D vectors in world coordinates. Used by:
 *    - GPU-side particle advection (frames define each face's
 *      local 2D coordinate system)
 *    - Tangent vector field visualization (face vectors expressed
 *      as 2D coords in (e1, e2) reconstruct in JS / shaders)
 *    - Texture mapping where a per-face frame is needed
 *
 *  Convention matches direction_field.cpp::faceOrthonormalBasis:
 *  e1 is the unit edge vector along f.halfedge(), e2 = n × e1.
 *  This is the same frame the trivial-connection direction field
 *  uses, so tangent vectors transported between faces are
 *  interpretable in the same coordinate system. */
struct FaceFrames {
    Eigen::MatrixXd e1;       // [nF, 3] — first tangent vector
    Eigen::MatrixXd e2;       // [nF, 3] — second tangent (n × e1)
    Eigen::MatrixXd normals;  // [nF, 3] — face normals
};

FaceFrames computeFaceFrames(ComputeContext& ctx);

// ── Geodesic Paths ───────────────────────────────────────────

/** Geodesic path between two vertices via the flip-out algorithm
 *  (Sharp & Crane, SIGGRAPH Asia 2020). Initialised with a
 *  Dijkstra path between vStart and vEnd, then iteratively
 *  shortened by edge flips until locally straight.
 *
 *  Returns a [N, 3] matrix where each row is a 3D point along
 *  the geodesic. N depends on the path complexity (typically
 *  O(diameter / mean-edge-length) on cortical surfaces).
 *
 *  Differs from `computeGeodesicDistance` (heat method) which
 *  returns a scalar field on all vertices, not a path. */
Eigen::MatrixXd tracePath(
    ComputeContext& ctx,
    int vStart,
    int vEnd
);

// ── Surface Parametrization ──────────────────────────────────

/** Boundary First Flattening (Sawhney & Crane 2017) — produces a
 *  conformal-by-default planar parametrization of the surface.
 *
 *  Returns a [V, 2] matrix of UV coordinates (u, v) per vertex.
 *  Mesh must have at least one boundary loop; for closed meshes
 *  the user must cut the surface first. The mesh in nxr-compute is
 *  ManifoldSurfaceMesh so boundary detection works automatically.
 *
 *  Used by texture-based visualizations (LIC, isoline rendering
 *  in UV space, anisotropic shaders). Lower angular distortion
 *  than LSCM, especially near complex boundaries. Uses
 *  geometry-central's bundled BFF implementation — no extra
 *  dependency.
 *
 *  Throws if the mesh has no boundary. */
Eigen::MatrixXd computeUVCoordinates(ComputeContext& ctx);

// ── Vector Field Operations ──────────────────────────────────

/** Whitney interpolation: edge 1-form → face-centered 3D vectors.
 *  Formula: V(f) = (N × (a + b + c)) / (6A)
 *  Returns [nF, 3] matrix (row-major vectors). */
Eigen::MatrixXd whitneyInterpolate(
    ComputeContext& ctx,
    const DECOperators& dec,
    const Eigen::VectorXd& oneForm
);

/** Gradient of a vertex scalar field → face-centered 3D vectors.
 *  For triangle (vi, vj, vk): ∇u = (1 / 2A) Σ (uₖ - uᵢ) (N × eⱼᵢ)
 *  Returns [nF, 3] matrix. */
Eigen::MatrixXd scalarGradient(
    ComputeContext& ctx,
    const Eigen::VectorXd& scalarField
);

// ── Isolines ─────────────────────────────────────────────────

struct IsolineResult {
    Eigen::MatrixXd positions;  // [segmentCount * 2, 3] endpoints of line segments
    int segmentCount;
};

/** Extract isolines (contours) from a per-vertex scalar field.
 *  numLevels evenly-spaced contour values in [minValue, maxValue].
 *  If minValue == maxValue, auto-detects range from the data. */
IsolineResult computeIsolines(
    ComputeContext& ctx,
    const Eigen::VectorXd& scalarField,
    int numLevels = 20,
    double minValue = 0.0,
    double maxValue = 0.0
);

// ── Vector Heat Method (Sharp, Soliman, Crane 2019) ──────────
//
// Stateful solver around a single (Laplacian, vector heat, Poisson)
// factor triple. Owned by the binding shell's per-mesh holder
// (ContextHolder / ContextWrapper) and lazily constructed on first
// use, mirroring CholeskyCache's lifetime model.
//
// The solver is opaque to JS — three free functions below operate
// on a long-lived solver passed by reference. The shells expose
// these as methods on their stateful context wrapper so the
// internal `VectorHeatMethodSolver*` never crosses the boundary.
//
// All tangent-vector inputs/outputs are V × 3 world-space vectors.
// Internally we project to / lift from each vertex's intrinsic
// tangent basis (geometry-central's `vertexTangentBasis`). This
// is a §11 layout extension: V × 2 tangent-frame data is lifted
// to V × 3 at the C++/JS boundary so callers never see Vector2.

class VectorHeatSolverImpl;
class VectorHeatSolver {
public:
    explicit VectorHeatSolver(ComputeContext& ctx, double tCoef = 1.0);
    ~VectorHeatSolver();
    VectorHeatSolver(const VectorHeatSolver&) = delete;
    VectorHeatSolver& operator=(const VectorHeatSolver&) = delete;

    // Internal — bindings should not call these directly.
    VectorHeatSolverImpl& impl();
    ComputeContext& ctx();

private:
    std::unique_ptr<VectorHeatSolverImpl> impl_;
};

/** Parallel-transport tangent vectors from a sparse set of sources
 *  to every vertex via the vector heat method.
 *
 *  Each source is (vertexIdx, worldVec3). The solver projects
 *  worldVec3 onto the source vertex's tangent basis, runs the
 *  vector heat solve, then lifts each destination vertex's
 *  Vector2 result to a world-space V × 3 vector using its own
 *  vertex tangent basis. Vectors at vertices very near a source
 *  reproduce the source vector almost exactly; vectors far away
 *  are the parallel transport of those sources along geodesics.
 *
 *  @return [nV, 3] row-major world-space tangent vectors. */
Eigen::MatrixXd vectorHeatTransport(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::MatrixXd& sourceVectors  // [nSources, 3]
);

/** Extend a sparse scalar field from the given source vertices to
 *  the entire mesh via the scalar heat method. Behaves like a
 *  geodesic Voronoi: each destination vertex takes the value of
 *  its nearest source, smoothly interpolated.
 *
 *  @return [nV] per-vertex scalars. */
Eigen::VectorXd vectorHeatExtendScalar(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::VectorXd& sourceValues   // [nSources]
);

/** Logarithmic map at a source vertex: per-vertex 2D coordinates
 *  (logX, logY) in the *source* vertex's tangent frame.
 *
 *  The norm √(logX² + logY²) is the geodesic distance from source
 *  to that vertex; the angle atan2(logY, logX) is the direction
 *  in source's frame. Equivalent to the exponential-map inverse
 *  flattened around the source, useful for picking, brushing, and
 *  any "distance + bearing from p" UI.
 *
 *  Strategy defaults to AffineLocal (Affine Heat Method, faster
 *  and more accurate near source than the original VectorHeat
 *  approach; the inner solver pre-factors and caches its
 *  connection-Laplacian so repeated calls on different sources
 *  amortize the factorization).
 *
 *  Returns:
 *    - logCoords: [nV, 2] row-major (logX, logY)
 *    - sourceE1, sourceE2: 3D world vectors of the source frame
 *      so consumers can reconstruct world positions if needed
 *      via  `position(v) ≈ source + logX·e1 + logY·e2`. */
struct LogMapResult {
    Eigen::MatrixXd logCoords;   // [nV, 2]
    Eigen::Vector3d sourceE1;    // source vertex's tangent basis e1 in world
    Eigen::Vector3d sourceE2;    // source vertex's tangent basis e2 in world
};

enum class LogMapStrategy {
    VectorHeat = 0,      // original VHM paper (no prefactoring)
    AffineLocal = 1,     // Affine Heat Method, prefactored, default
    AffineAdaptive = 2,  // Affine Heat Method, no prefactor, most accurate
};

LogMapResult vectorHeatLogMap(
    VectorHeatSolver& solver,
    int sourceVertex,
    LogMapStrategy strategy = LogMapStrategy::AffineLocal
);

/** Karcher mean / surface center of a set of source vertices.
 *  Returns the 3D world position of the SurfacePoint that
 *  minimizes ∑ d(p, source_i)^p (default p=2 → Karcher mean). */
Eigen::Vector3d vectorHeatFindCenter(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    int p = 2
);

// ── Signed Heat Method (Feng & Crane 2024) ───────────────────
//
// Signed geodesic distance from a curve on the surface. Stateful
// solver, same lifetime model as VectorHeatSolver. The curve is
// expressed as a polyline of vertex indices; `isLoop` controls
// whether the curve closes (last → first).
//
// Useful for region selection, flood fills, and morphological
// operations on the mesh: positive distance one side of the
// curve, negative the other. The level-set constraint keeps the
// zero-set pinned exactly at the curve.

class SignedHeatSolverImpl;
class SignedHeatSolver {
public:
    explicit SignedHeatSolver(ComputeContext& ctx, double tCoef = 1.0);
    ~SignedHeatSolver();
    SignedHeatSolver(const SignedHeatSolver&) = delete;
    SignedHeatSolver& operator=(const SignedHeatSolver&) = delete;

    SignedHeatSolverImpl& impl();

private:
    std::unique_ptr<SignedHeatSolverImpl> impl_;
};

enum class SignedHeatLevelSet {
    None = 0,        // unconstrained — curve may not lie exactly on the zero set
    ZeroSet = 1,     // pin the zero level set to the curve (default)
    Multiple = 2,    // pin each curve to its own level set
};

/** Signed geodesic distance from a curve to every vertex.
 *
 *  @param curveVertices  ordered vertex indices defining the curve polyline
 *  @param isLoop         if true, edge from last → first is included
 *  @param levelSet       which level-set constraint to enforce
 *  @return               [nV] signed distances (sign convention follows
 *                        the curve's orientation; flip the input order
 *                        to flip the sign) */
Eigen::VectorXd signedHeatDistance(
    SignedHeatSolver& solver,
    const std::vector<int>& curveVertices,
    bool isLoop = true,
    SignedHeatLevelSet levelSet = SignedHeatLevelSet::ZeroSet
);

// ── Smooth Direction Fields (NRoSy, Knöppel-Crane) ───────────
//
// Smoothest unit-norm direction field of order n via the
// Knöppel-Crane formulation — minimizes Dirichlet energy on
// the connection Laplacian, no prescribed singularities (they
// emerge automatically). Distinct from `computeDirectionField`
// above (which solves trivial connections with user-prescribed
// singularities).
//
// nSym selects the symmetry order:
//    1 → vector field (one direction per face)
//    2 → line field (two opposite directions per face)
//    4 → cross field (four directions, useful for quad meshing
//                     and stripe patterns aligned to two axes)
//
// `alignToCurvature` returns the principal-curvature-aligned
// field instead of pure smoothest (only meaningful for nSym ≥ 2).

/** Result of `computeSmoothVertexField`. The face-based variant
 *  returns a plain Eigen::MatrixXd directly, since face stripes
 *  aren't a downstream consumer of its raw form. */
struct SmoothVertexFieldResult {
    Eigen::MatrixXd vertexVectors;   // [nV, 3] world-space, per vertex
    Eigen::VectorXd vertexFieldRaw;  // [nV * 2] raw Vector2 in vertex tangent basis
    int nSym;
};

/** Compute the smoothest face-based direction field.
 *  Returns [nF, 3] world-space vectors (principal nSym-RoSy
 *  representative; the other nSym-1 directions are obtained by
 *  rotating in the face tangent plane). */
Eigen::MatrixXd computeSmoothFaceField(
    ComputeContext& ctx,
    int nSym = 4,
    bool alignToCurvature = false
);

/** Compute the smoothest vertex-based direction field (needed
 *  as input to stripe patterns).
 *  Returns the raw [nV * 2] Vector2 form alongside [nV, 3] world
 *  vectors, since stripes consume the Vector2 form internally. */
SmoothVertexFieldResult computeSmoothVertexField(
    ComputeContext& ctx,
    int nSym = 2,
    bool alignToCurvature = false
);

// ── Stripe Patterns (Knöppel-Crane SIGGRAPH 2015) ────────────
//
// Procedural sinusoidal stripes aligned to a 2-RoSy direction
// field. Output is a list of 3D polyline segments (the zero
// level set of the per-corner phase function), suitable for
// drawing as three.js LineSegments.
//
// Frequencies control stripe spacing in oscillations per unit
// edge length. A constant uniformFrequency is the common case;
// pass per-vertex frequencies if you want curvature-modulated
// striping.

struct StripePatternResult {
    Eigen::MatrixXd positions;   // [2 * segmentCount, 3] endpoint pairs
    int segmentCount;
};

/** Stripe pattern with a uniform target frequency. The vertex
 *  field must be 2-RoSy (use `computeSmoothVertexField(ctx, 2)`
 *  or pass an existing line field). */
StripePatternResult computeStripePattern(
    ComputeContext& ctx,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2] in vertex tangent basis
    double uniformFrequency,
    bool connectOnSingularities = true
);

/** Stripe pattern with per-vertex target frequencies. */
StripePatternResult computeStripePatternFreq(
    ComputeContext& ctx,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2]
    const Eigen::VectorXd& frequencies,     // [nV]
    bool connectOnSingularities = true
);

// ── Direction Field Design (Trivial Connections) ─────────────

struct DirectionFieldResult {
    Eigen::VectorXd connections;         // φ per edge (1-form angles)
    Eigen::MatrixXd directionVectors;    // [nF, 3] smooth direction field on faces
    Eigen::MatrixXd orthogonalVectors;   // [nF, 3] 90° rotated direction field
    double eulerCharacteristic;
    bool gaussBonnetSatisfied;
};

/**
 * Compute a smooth direction field with prescribed singularities.
 * Uses the trivial connections algorithm (Crane, de Goes, Desbrun).
 *
 * Input: singularityMap — vertex index → singularity index (typically ±1 or ±0.5)
 * The sum of indices MUST equal the mesh Euler characteristic (Gauss-Bonnet).
 *
 * Output: φ = δβ + γ (per-edge 1-form angles), plus the resulting
 * direction and orthogonal vector fields on faces.
 */
DirectionFieldResult computeDirectionField(
    ComputeContext& ctx,
    const DECOperators& dec,
    CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);

// ── Streamlines ──────────────────────────────────────────────

struct StreamlineResult {
    // Flat [2 * N, 3] array of line segment endpoints (pair-wise)
    Eigen::MatrixXd positions;
    int segmentCount;
};

/**
 * Trace streamlines through a face vector field using forward Euler
 * with face-crossing detection.
 *
 * @param faceField  [nF, 3] vector field on face centroids
 * @param numSeeds   target number of seed streamlines (quasi-uniform)
 * @param stepCoef   step size = stepCoef * meanEdgeLength
 * @param maxSteps   max integration steps per streamline
 */
StreamlineResult traceStreamlines(
    ComputeContext& ctx,
    const Eigen::MatrixXd& faceField,
    int numSeeds = 15,
    double stepCoef = 0.15,
    int maxSteps = 1000
);

} // namespace nxr::compute
