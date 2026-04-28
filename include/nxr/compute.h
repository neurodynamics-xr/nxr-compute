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

struct MeshOperators {
    Eigen::SparseMatrix<double> stiffness;  // cotangent Laplacian (PSD, symmetrized)
    Eigen::SparseMatrix<double> mass;       // lumped Voronoi mass (diagonal, SPD)
    Eigen::VectorXd vertexAreas;            // dual vertex areas [nV]
    Eigen::MatrixXd normals;                // vertex normals [nV, 3]
    double totalArea;
    int nV, nE, nF;
};

/** Assemble operators from a context. */
MeshOperators assembleMeshOperators(ComputeContext& ctx);

/** Convenience overload: creates context and assembles operators in one call. */
MeshOperators assembleMeshOperators(
    const double* vertices, int nV,
    const int32_t* faces, int nF
);

// ── DEC Operators ────────────────────────────────────────────

struct DECOperators {
    Eigen::SparseMatrix<double> d0;      // vertex → edge
    Eigen::SparseMatrix<double> d1;      // edge → face
    Eigen::SparseMatrix<double> hodge0;  // vertex → vertex (diagonal)
    Eigen::SparseMatrix<double> hodge1;  // edge → edge (diagonal)
    Eigen::SparseMatrix<double> hodge2;  // face → face (diagonal)
    Eigen::SparseMatrix<double> hodge1Inverse;
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

    /** Convenience overload: factors ops.stiffness. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& laplacian(
        const MeshOperators& ops) {
        return laplacian(ops.stiffness);
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

/** Convenience overload: forwards to (K, M) using ops.stiffness, ops.mass. */
inline Eigen::VectorXd solvePoisson(
    const MeshOperators& ops,
    CholeskyCache& cache,
    const std::map<int, double>& densityMap
) {
    return solvePoisson(ops.stiffness, ops.mass, cache, densityMap);
}

// ── Geodesic Distance (Heat Method) ──────────────────────────

/**
 * Compute geodesic distances from source vertices via the heat method.
 * Uses geometry-central's HeatMethodDistanceSolver.
 *
 * @param ctx             compute context (mesh + geometry)
 * @param sourceVertices  list of source vertex indices
 * @return                per-vertex geodesic distances
 */
Eigen::VectorXd computeGeodesicDistance(
    ComputeContext& ctx,
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
