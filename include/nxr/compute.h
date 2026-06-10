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
class IntrinsicGeometryInterface;
class SignpostIntrinsicTriangulation;
}
}

namespace nxr::manifold {

// Cross-cutting types live in nxr::core; bring them into manifold scope
// so unqualified uses below (Error, CancellationToken, ProgressObserver)
// resolve. See <nxr/errors.h>, <nxr/cancellation.h>, <nxr/progress.h>.
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;

// Forward declarations for facets and geometry bundles.
namespace facet {
    class TopologyFacet;  class EmbeddedFacet;  class IntrinsicFacet;
    class ExtrinsicFacet; class GaugeFacet;     class OperatorsFacet;
}
namespace geometry { struct MeshGeometry; struct MeshTopology; }
namespace ops { struct DECOperators; class CholeskyCache; }
namespace ops::laplacian::connection { enum class CovariantCoupling; }

enum class GaugeType { Euclidean, LeviCivita, Trivial };

enum class OperatorId {
    LaplacianCotan, LaplacianGraph, LaplacianConnection, LaplacianCovariant,
    Dec, MassLumped, MassGalerkin, Gradient3D, Dirac
};

// ── Compute Context ──────────────────────────────────────────
// Owns the geometry-central mesh and geometry objects.
// All compute functions operate on a context, which can be created
// once and reused across multiple calls.

class Manifold {
public:
    // trailing param defaults to false — every existing call site is unchanged
    Manifold(const double* vertices, int nV,
                   const int32_t* faces, int nF,
                   bool intrinsicDelaunay = false);
    ~Manifold();

    // Non-copyable
    Manifold(const Manifold&) = delete;
    Manifold& operator=(const Manifold&) = delete;

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

    // Returns true iff this Manifold was built with intrinsicDelaunay=true
    // (i.e. intrinsicTri_ is populated and flipToDelaunay() has been called).
    bool isIntrinsicDelaunay() const;

    // Geometry to assemble INTRINSIC-interface operators on (cotan, mass, dual
    // areas, …). Intrinsic Delaunay geometry when normalised, else the embedded
    // VertexPositionGeometry (itself an IntrinsicGeometryInterface). geometry()
    // is unchanged and still used for EXTRINSIC quantities (normals, frames).
    geometrycentral::surface::IntrinsicGeometryInterface& operatorGeometry();

    // ── Geometry facets (additive; thin views over the accessors above) ──
    // Each returns a lightweight view object; see include/nxr/facets.h.
    facet::TopologyFacet  topology();
    facet::EmbeddedFacet  embedded();
    facet::IntrinsicFacet intrinsic();
    facet::ExtrinsicFacet extrinsic();

    // Raw-input aliases (facet-agnostic literal input, retained from ctor).
    const Eigen::MatrixXd& vertexPositions() const;  // [nV,3]
    const Eigen::MatrixXi& faces() const;            // [nF,3], 0-based

    // Lazily-cached light per-element bundles backing the data facets.
    const geometry::MeshGeometry& lightGeometry();   // geometry::meshGeometry(*this), cached
    const geometry::MeshTopology& topologyData();    // geometry::meshTopology(*this), cached

    // ── Gauge workflow ──
    // Singularity-aware constructor (pattern 2): default active gauge = Trivial.
    // Throws Error(InvalidInput) if Σ singularities != eulerCharacteristic().
    Manifold(const double* vertices, int nV, const int32_t* faces, int nF,
             const std::map<int,double>& singularities, bool intrinsicDelaunay = false);

    int eulerCharacteristic() const;                 // nV - nE + nF
    GaugeType activeGaugeType() const;
    const std::map<int,double>& activeSingularities() const;
    // Re-point the active/default gauge. Trivial validates Gauss-Bonnet.
    void setGauge(GaugeType type, const std::map<int,double>& singularities = {});

    // Declared only; bodies implemented in B2 (require GaugeFacet complete type).
    facet::GaugeFacet gauge();
    facet::GaugeFacet gauge(GaugeType type,
                            const std::map<int,double>& singularities = {});

    // Lazy operator-assembly helpers shared by gauge + operators facets.
    ops::DECOperators& decOperators();     // assembleDECOperators(*this), cached
    ops::CholeskyCache& choleskyCache();   // owned cache, lazy-initialised

    // ── Operators facet ──
    facet::OperatorsFacet operators();
    void releaseOperator(OperatorId id);        // drop a cached operator
    bool isOperatorCached(OperatorId id) const;

private:
    // Per-operator independent cache slots. Requesting one NEVER builds another.
    // (Dec reuses decCache_ from the gauge path.)
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheLaplacianCotan_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheLaplacianGraph_;
    std::unique_ptr<Eigen::SparseMatrix<std::complex<double>>>           cacheLaplacianConnection_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheLaplacianCovariant_;
    // Coupling the covariant cache was built with (as int, since CovariantCoupling
    // is only forward-declared here) — rebuild on mismatch so covariant(Product)
    // and covariant(Ambient) on one handle never alias. -1 = nothing cached yet.
    int                                                                  cachedCovariantCoupling_ = -1;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheMassLumped_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheMassGalerkin_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheGradient3D_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDirac_;  // extrinsic block E

    // Private cache-fill helpers called by OperatorsFacet::LaplacianView.
    // These source DIRECTLY from operatorGeometry()'s GC cache without
    // calling assembleManifoldOperators (which would fuse cotan+mass+normals).
    const Eigen::SparseMatrix<double>& cotanLaplacianCached_();
    const Eigen::SparseMatrix<double>& graphLaplacianCached_();

    // Private cache-fill helpers for OperatorsFacet::MassView.
    // Each sources DIRECTLY from operatorGeometry()'s GC require* — independent:
    // requesting lumped NEVER builds galerkin and vice-versa.
    const Eigen::SparseMatrix<double>& massLumpedCached_();
    const Eigen::SparseMatrix<double>& massGalerkinCached_();

    // Private cache-fill helpers for OperatorsFacet::LaplacianView::connection/covariant.
    // connection: builds in the ACTIVE gauge (LC or trivial); stores K_complex.
    // covariant: delegates to connection() + gauge().grid() + cotan() then assembles.
    const Eigen::SparseMatrix<std::complex<double>>& connectionLaplacianCached_();
    const Eigen::SparseMatrix<double>& covariantLaplacianCached_(
        ops::laplacian::connection::CovariantCoupling coupling);

    // Covariant gradient G (3E x 3N) — differential::covariantGradient(*this), cached
    // (mesh-only; depends on the frames, not the gauge or any parameter).
    const Eigen::SparseMatrix<double>& gradient3DCached_();

    // Relative-Dirac extrinsic block E (4V×4V), cached (OperatorId::Dirac).
    const Eigen::SparseMatrix<double>& diracExtrinsicBlockCached_();
    // Assemble L(τ) = (1−τ)(cotanL⊗I₄) + τ·E by value. τ ∈ [0,1] (validated).
    Eigen::SparseMatrix<double> diracFamily_(double tau);

    friend class facet::OperatorsFacet;
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh>             mesh_;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry>         geometry_;
    std::unique_ptr<geometrycentral::surface::SignpostIntrinsicTriangulation>  intrinsicTri_;  // null unless intrinsicDelaunay=true

    Eigen::MatrixXd                          vertexPositions_;  // [nV,3] retained input
    Eigen::MatrixXi                          faces_;            // [nF,3] retained input (0-based)
    std::unique_ptr<geometry::MeshGeometry>  lightGeometryCache_;
    std::unique_ptr<geometry::MeshTopology>  topologyDataCache_;

    GaugeType            activeGaugeType_    = GaugeType::LeviCivita;
    std::map<int,double> activeSingularities_;
    // Validates Σ singularities == eulerCharacteristic(); throws on mismatch.
    void validateSingularities_(const std::map<int,double>& s) const;

    std::unique_ptr<ops::DECOperators>  decCache_;
    std::unique_ptr<ops::CholeskyCache> choleskyCachePtr_;
};

// Extrinsic Delaunay edge-flip repair (geometry-central fixDelaunay).
// Flips edges in place — same vertices (positions/count/indices), new faces.
// Best-effort PSD (reduces obtuse/negative cotan weights), not a certificate;
// the intrinsic-Delaunay certificate is a separate facility. Input must be a
// valid manifold (the underlying ManifoldSurfaceMesh throws otherwise).
struct DelaunayNormalization {
    Eigen::MatrixXi faces;  // [nF, 3] 0-based, re-triangulated; same vertex indices
    int flips = 0;          // edge flips performed (0 => already Delaunay)
};
DelaunayNormalization fixDelaunay(
    const double* vertices, int nV, const int32_t* faces, int nF);

} // namespace nxr::manifold

namespace nxr::manifold::ops {

// ── Operator Assembly ────────────────────────────────────────

/**
 * Mass-matrix variant for the FEM L² inner product ⟨u, v⟩ = uᵀ M v.
 *
 * Names match geometry-central's vocabulary exactly:
 * `vertexLumpedMassMatrix` and `vertexGalerkinMassMatrix`
 * (see `intrinsic_geometry_interface.h`). Both variants conserve total
 * surface area (Σᵢⱼ Mᵢⱼ == totalArea) and are symmetric. They differ
 * in M's structure and in the eigenvalues / eigenvectors of the
 * generalized problem K φ = λ M φ.
 *
 * - Lumped (default): diagonal, M_ii = (Σ adjacent triangle areas) / 3.
 *   Sources from geometry-central's `vertexLumpedMassMatrix`, which is
 *   `diag(vertexDualAreas)` where each `vertexDualAreas[v]` is the sum
 *   of A_T/3 over incident triangles. This is the lumped barycentric
 *   mass — NOT a circumcentric or mixed-Voronoi construction.
 *
 * - Galerkin: full sparse, off-diagonal couplings between adjacent
 *   vertices. Per-triangle element matrix is (A_T / 12) · [[2 1 1][1 2 1][1 1 2]],
 *   from L² integrals of piecewise-linear hat functions. Sources from
 *   geometry-central's `vertexGalerkinMassMatrix`. Eigenvalues converge
 *   to the continuous Laplace–Beltrami spectrum faster than the lumped
 *   variant on coarse meshes.
 */
enum class MassMatrixVariant {
    Lumped,    // diagonal; matches GC vertexLumpedMassMatrix
    Galerkin   // sparse FEM; matches GC vertexGalerkinMassMatrix
};

// ManifoldOperators / DECOperators mix view and value semantics. Field
// names follow geometry-central's canonical naming for the underlying
// cached matrices (cotanLaplacian, vertexDualAreas, vertexNormals,
// d0/d1/hodge*), so the surface here is a 1-to-1 viewer over GC's
// cache where the concept exists. Variant-aware quantities (mass)
// keep a generic name because GC has no single matrix covering them.
//
//   * View fields (const-references into geometry-central's cached matrices):
//       ManifoldOperators::cotanLaplacian, all DECOperators fields. Bound by
//       assembleManifoldOperators / assembleDECOperators after the relevant
//       require* call pins them on the geometry.
//
//   * Owned field:
//       ManifoldOperators::mass — variant-dependent (Lumped / Galerkin);
//       held by value because GC exposes two distinct matrices and we
//       want a single uniform field shape regardless of which variant
//       the caller requested.
//
// Lifetime contract: bindings must keep the owning Manifold alive
// for the entire lifetime of any ManifoldOperators / DECOperators they hold,
// and must NOT call unrequire* on the underlying geometry while the
// structs are alive. ContextHolder / ContextWrapper enforce this by
// keeping the operator structs and the context together.
//
// Because of the reference fields, both structs are copy-constructible
// (refs rebind to the same targets) but not copy-assignable. The
// bindings hold them via shared_ptr / unique_ptr and never reassign.

// Row-major V×3 alias. `vertexNormals` is exclusively consumed row-wise
// (each row is one (x,y,z) triple flattened directly into the §11 V*3
// row-major output buffer at the binding edge). Storing it row-major
// here makes the binding-side memcpy direct — no transpose needed,
// because the in-memory layout already matches the output convention.
// Eigen kernels read this as `m.row(v)` only; no column ops in any of
// the existing call sites.
using VertexNormalsMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct ManifoldOperators {
    ManifoldOperators(const Eigen::SparseMatrix<double>& cotanLaplacian_,
                      Eigen::SparseMatrix<double> mass_,
                      MassMatrixVariant massVariant_,
                      Eigen::VectorXd vertexDualAreas_,
                      VertexNormalsMatrix vertexNormals_,
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
    Eigen::VectorXd vertexDualAreas;                    // [nV] A/3-per-vertex, from GC vertexDualAreas (variant-independent)
    VertexNormalsMatrix vertexNormals;                  // [nV, 3] row-major; see alias comment above
    double totalArea;
    // nV / nE / nF intentionally omitted — read from the owning
    // ComputeContext (ctx.nV(), ctx.nE(), ctx.nF()) to avoid duplicating
    // mesh metadata that's already canonical on the context.
};

/** Assemble operators from a context. Default mass variant is Lumped
 *  (diagonal, A/3-per-vertex). Same numerical behavior as the prior
 *  default that was misnamed `Voronoi`. */
ManifoldOperators assembleManifoldOperators(
    Manifold& m,
    MassMatrixVariant variant = MassMatrixVariant::Lumped);

/** Convenience overload: creates context and assembles operators in one call. */
ManifoldOperators assembleManifoldOperators(
    const double* vertices, int nV,
    const int32_t* faces, int nF,
    MassMatrixVariant variant = MassMatrixVariant::Lumped
);

/** String → enum helper for binding shells (WASM, addon, MEX).
 *  Accepts "lumped" and "galerkin", matching geometry-central's
 *  `vertexLumpedMassMatrix` and `vertexGalerkinMassMatrix`. Throws
 *  Error(InvalidInput) on unknown. Case-sensitive. */
MassMatrixVariant parseMassMatrixVariant(const std::string& name);

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
    // see the comment block above ManifoldOperators for the lifetime
    // contract. dec.hodge0 is the same matrix as ops.mass when
    // ops.massVariant == Lumped (both source from GC vertexDualAreas).
    const Eigen::SparseMatrix<double>& d0;      // vertex → edge
    const Eigen::SparseMatrix<double>& d1;      // edge → face
    const Eigen::SparseMatrix<double>& hodge0;  // vertex → vertex (diagonal)
    const Eigen::SparseMatrix<double>& hodge1;  // edge → edge (diagonal)
    const Eigen::SparseMatrix<double>& hodge2;  // face → face (diagonal)
    const Eigen::SparseMatrix<double>& hodge1Inverse;
};

DECOperators assembleDECOperators(Manifold& m);

// Zero-copy passthrough accessors. Each triggers geometry-central's
// internal cache on first call (cascading through index / area /
// cotan-weight dependencies inside requireDECOperators or
// requireVertexDualAreas) and returns the cached field by const
// reference on every subsequent call. No matrix copies, in contrast
// to assembleDECOperators which builds a fresh DECOperators struct.
// The returned reference is valid for the Manifold's lifetime.
const Eigen::SparseMatrix<double>& d0(Manifold& m);
const Eigen::SparseMatrix<double>& d1(Manifold& m);
const Eigen::SparseMatrix<double>& hodge0(Manifold& m);
const Eigen::SparseMatrix<double>& hodge1(Manifold& m);
const Eigen::SparseMatrix<double>& hodge2(Manifold& m);
const Eigen::SparseMatrix<double>& hodge1Inverse(Manifold& m);
const Eigen::VectorXd&             vertexDualAreas(Manifold& m);

// Graph (combinatorial) Laplacian L = D − A, built as d0ᵀ d0 from the
// metric-free exterior derivative. Pure topology — independent of vertex
// positions. Symmetric PSD; diagonal = vertex degree,
// off-diagonal = −1 for adjacent vertices, 0 otherwise.
Eigen::SparseMatrix<double> graphLaplacian(Manifold& m);

// Extrinsic block of the relative Dirac family (Liu/Jacobson/Crane, SGP 2017):
// E = Dᵀ ⋆_F D, the [4V×4V] real-symmetric PSD Galerkin form of the relative
// Dirac operator D_N (the shape-operator / Gauss-map energy). Quaternion order
// [w,x,y,z]; index 4*v+c. Geometry-only (vertex normals + face areas) — the one
// operator assembled directly rather than wrapped from geometry-central.
namespace dirac {
Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m);
}  // namespace dirac

// ── Factor Cache ─────────────────────────────────────────────
//
// Pre-factored Cholesky / LU decompositions of mesh-derived
// matrices, cached across compute calls. Each factor is built
// lazily on first request and reused on every subsequent call
// for the lifetime of the owning Manifold.
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
        const ManifoldOperators& ops) {
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

} // namespace nxr::manifold::ops

namespace nxr::manifold::ops::laplacian::connection {

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
// eigensolver (`solve`) when paired with a block-diagonal
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
    Eigen::MatrixXd  frameE1;   // [baseDim, 3] tangent frame e1 per vertex/face (gauge)
    Eigen::MatrixXd  frameE2;   // [baseDim, 3] tangent frame e2 per vertex/face (gauge)
                                // Empty (0 rows) for EdgeCrouzeixRaviart domain.
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
    Manifold& m,
    const ConnectionLaplacianOptions& opts = {}
);

/** String → enum helpers for binding shells (WASM, addon).
 *  Domain accepts "vertex", "face", "edge". Format accepts "real2N"
 *  / "real" / "complex". Throws Error(InvalidInput) on unknown. */
ConnectionDomain          parseConnectionDomain(const std::string& name);
ConnectionLaplacianFormat parseConnectionLaplacianFormat(const std::string& name);

/**
 * Assemble the connection Laplacian using the trivial connection transport.
 *
 * The trivial connection for the given singularity map is computed first
 * (via computeTrivialConnection), then used to modulate the Levi-Civita
 * transport on each halfedge:
 *
 *   ρ^TC[he] = ρ^LC[he] · exp(i · sign(he) · φ[edge(he)])
 *
 * where sign(he) = +1 if he == edge.halfedge(), -1 otherwise, and
 * φ is the per-edge trivial connection 1-form (Eigen::VectorXd, nEdges).
 *
 * The off-diagonal entry K[i,j] for halfedge he (tail=i, tip=j) is then:
 *
 *   K[i,j] += -w_ij · (ρ^TC[he.twin()])^nSym
 *
 * Gauss-Bonnet: Σ singularityMap values must equal χ(mesh). Not checked
 * internally — throws nothing if violated, but φ will be incorrect.
 *
 * Only ConnectionDomain::Vertex is currently supported. Passing Face or
 * EdgeCrouzeixRaviart throws Error(InvalidInput).
 *
 * Output format follows opts.format (Real2N or Complex), same as
 * assembleConnectionLaplacian.
 */
ConnectionLaplacian assembleTrivialConnectionLaplacian(
    Manifold& m,
    const std::map<int, double>& singularityMap,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const ConnectionLaplacianOptions& opts = {}
);

// ── Covariant (3-frame) Laplacian ─────────────────────────────
//
// 3N×3N real symmetric sparse Laplacian on the full 3-frame
// [a; b; c] = [Re z; Im z; normal], component-major block layout.
// Two couplings: Product (blkdiag) and Ambient (frame-conjugate).
// See spec docs/superpowers/specs/2026-06-08-vector-laplacian-3d-design.md §3–4.

// Coupling for the 3-frame covariant Laplacian (see covariant-laplacian spec).
enum class CovariantCoupling {
    Product,   // blkdiag(real-expand(K), cotanL) — tangent ⊕ normal decoupled
    Ambient    // frame-conjugate of kron(I3, cotanL): L3[i,j] = cotanL[i,j]·(Fiᵀ Fj)
};

// 3N×3N real symmetric Laplacian on the 3-frame [a;b;c] = [Re z; Im z; normal],
// component-major block layout. Pure linear algebra over pieces the caller holds:
//   K        = gauge connection Laplacian (V×V complex Hermitian) — used by Product
//              (and equals the Ambient tangent block for the matching gauge)
//   gaugeGrid= realized gauge frame, V×3 complex c = e1 + i·e2 (normal = Re×Im) — used by Ambient
//   cotanL   = scalar cotan Laplacian (V×V real)
// See spec §3–4. Symmetric PSD.
Eigen::SparseMatrix<double> assembleCovariantLaplacian(
    CovariantCoupling coupling,
    const Eigen::SparseMatrix<std::complex<double>>& K,
    const Eigen::MatrixXcd& gaugeGrid,
    const Eigen::SparseMatrix<double>& cotanL);

} // namespace nxr::manifold::ops::laplacian::connection

namespace nxr::manifold::solve {

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
 * @param sigma      Shift for shift-invert mode. Values near 0
 *                   (e.g. -1e-8) target the lowest eigenmodes.
 *                   The solver pre-factors (K - σM) once, then
 *                   each IRAM iteration is a cheap forward/back-
 *                   substitution. This is dramatically faster than
 *                   standard mode for the smallest eigenvalues of
 *                   large sparse systems.
 * @param normalize  If true, M-orthonormalize eigenvectors so
 *                   U' * M * U = I (Cholesky whitening). Default
 *                   false — eigenvectors are returned in the form
 *                   Spectra produces. Set true for downstream
 *                   spectral synthesis / inner-product work.
 * @param removeDC   If true, strip the DC (zero-eigenvalue) mode
 *                   from the result. Default false. Set true when
 *                   the input is a closed-mesh Laplacian and the
 *                   null mode would dominate spectral filters.
 * @param cancel     Optional cancellation token. The solver polls
 *                   cancel.requested() once per Spectra perform_op
 *                   call; on true, throws Error(Cancelled).
 * @param progress   Optional observer of solver progress.
 *
 * Throws Error(EigensolveInvalidK) on k < 1 or k > N - 1.
 * Throws Error(Cancelled)          on cancel.requested().
 * Throws Error(EigensolveNotConverged) on Spectra failure with
 *                                          no partial result.
 */
EigenResult eigen(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int k,
    double sigma                       = -1e-8,
    bool normalize                     = false,
    bool removeDC                      = false,
    const CancellationToken& cancel    = {},
    const ProgressObserver& progress   = {}
);

/**
 * M-orthonormalize an eigenvector basis: U' * M * U = I.
 * Cholesky-whitening fast path with eigen-decomposition fallback.
 * Typically called via `eigen(..., normalize=true)`; this standalone
 * entry point is for transforming pre-computed bases (e.g. arriving
 * from MATLAB / Python).
 */
Eigen::MatrixXd normalize(
    const Eigen::MatrixXd& U,
    const Eigen::SparseMatrix<double>& M
);

/**
 * Strip the DC (zero-eigenvalue) mode from an EigenResult.
 * Returns a copy with the eigenvalue closest to zero removed.
 * Typically called via `eigen(..., removeDC=true)`; this standalone
 * entry point lets callers strip the DC mode from a pre-computed
 * EigenResult.
 */
EigenResult removeDC(const EigenResult& result);


// ── Poisson Solver ───────────────────────────────────────────

/**
 * Solve the scalar Poisson equation: K * φ = -M * (ρ - ρ̄)
 * where ρ̄ is the M-weighted mean of ρ.
 *
 * Uses cache.laplacian(K) for the LLT factor of (K + ε·I), so
 * repeated calls with the same cache skip the factorization step.
 *
 * Agnostic in K and M: works on cotangent Laplacians + lumped mass
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
Eigen::VectorXd poisson(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    ops::CholeskyCache& cache,
    const std::map<int, double>& densityMap
);

/** Convenience overload: forwards to (K, M) using ops.cotanLaplacian, ops.mass. */
inline Eigen::VectorXd poisson(
    const ops::ManifoldOperators& ops,
    ops::CholeskyCache& cache,
    const std::map<int, double>& densityMap
) {
    return poisson(ops.cotanLaplacian, ops.mass, cache, densityMap);
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
    explicit HeatGeodesicSolver(Manifold& m, double tCoef = 1.0);
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
Eigen::VectorXd heat(
    HeatGeodesicSolver& solver,
    const std::vector<int>& sourceVertices
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
    explicit SignedHeatSolver(Manifold& m, double tCoef = 1.0);
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
Eigen::VectorXd signedHeat(
    SignedHeatSolver& solver,
    const std::vector<int>& curveVertices,
    bool isLoop = true,
    SignedHeatLevelSet levelSet = SignedHeatLevelSet::ZeroSet
);

} // namespace nxr::manifold::solve

namespace nxr::manifold::query {

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
 *  Differs from `heat` (heat method) which
 *  returns a scalar field on all vertices, not a path. */
Eigen::MatrixXd tracePath(
    Manifold& m,
    int vStart,
    int vEnd
);

// ── Locus primitives (D11) ───────────────────────────────────
//
// `query::*` returns the geometric locus itself (the "where"); the
// paired `measure::*` (below) returns its scalar metric (the "how
// much"). Three loci: a point, a polyline, a region of faces.

struct PointLocus {
    int vertex;  // vertex index in the owning Manifold
};

struct PolylineLocus {
    Eigen::MatrixXd points;  // [N, 3] world-space points along the polyline
};

struct RegionLocus {
    std::vector<int> faces;  // face indices covered by the region
};

/** Trivial point query — echoes the input vertex after a range check.
 *  Throws Error(InvalidInput) on `v < 0 || v >= m.nV()`. */
PointLocus point(Manifold& m, int v);

/** Polyline between two vertices via the edge-flip geodesic algorithm
 *  (Sharp & Crane 2020). Identical to `tracePath` but typed as a
 *  PolylineLocus so it pairs with `measure::line`. */
PolylineLocus line(Manifold& m, int vStart, int vEnd);

/** Region of faces inside the geodesic ball of radius `level` around
 *  vertex `v`. A face is included iff all three of its vertices have
 *  heat-method geodesic distance `<= level` from `v` — conservative,
 *  slightly underestimates the true region area, but well-defined and
 *  free of barycentric interpolation.
 *
 *  Constructs a fresh `solve::HeatGeodesicSolver` per call (pays one
 *  Cholesky factor). For repeated queries on the same mesh, callers
 *  that want the cached path should compute the heat-distance scalar
 *  field themselves via the cached solver on `Manifold` and then
 *  threshold it manually.
 *
 *  Throws Error(InvalidInput) on `v` out of range or `level <= 0`. */
RegionLocus area(Manifold& m, int v, double level);

} // namespace nxr::manifold::query

namespace nxr::manifold::measure {

// ── Scalar metrics of query loci (D11) ───────────────────────
//
// Paired with `query::*`: each `measure::*` here takes the same
// inputs as the matching `query::*` and returns the scalar value
// (coordinate, length, area). Locus-input overloads avoid re-running
// the query when the caller already has the locus.

/** 3D position of vertex `v`, taken from the manifold's vertex
 *  positions. Throws Error(InvalidInput) on out-of-range `v`. */
Eigen::Vector3d point(Manifold& m, int v);

/** Polyline length along the edge-flip geodesic from `vStart` to `vEnd`.
 *  Equivalent to `line(query::line(m, vStart, vEnd))`. */
double line(Manifold& m, int vStart, int vEnd);

/** Polyline length of a pre-computed `PolylineLocus`. Sum of Euclidean
 *  distances between consecutive points. Returns 0 for empty / single-
 *  point polylines. */
double line(const query::PolylineLocus& locus);

/** Total face area of the geodesic-ball region of radius `level` around
 *  vertex `v`. Equivalent to `area(m, query::area(m, v, level))`. */
double area(Manifold& m, int v, double level);

/** Total face area of a pre-computed `RegionLocus`, summed from
 *  geometry-central's cached face areas. */
double area(Manifold& m, const query::RegionLocus& locus);

} // namespace nxr::manifold::measure

namespace nxr::manifold::transport {

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
    explicit VectorHeatSolver(Manifold& m, double tCoef = 1.0);
    ~VectorHeatSolver();
    VectorHeatSolver(const VectorHeatSolver&) = delete;
    VectorHeatSolver& operator=(const VectorHeatSolver&) = delete;

    // Internal — bindings should not call these directly.
    VectorHeatSolverImpl& impl();
    Manifold& m();

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
Eigen::MatrixXd parallel(
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
Eigen::VectorXd extendScalar(
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

LogMapResult logMap(
    VectorHeatSolver& solver,
    int sourceVertex,
    LogMapStrategy strategy = LogMapStrategy::AffineLocal
);

/** Karcher mean / surface center of a set of source vertices.
 *  Returns the 3D world position of the SurfacePoint that
 *  minimizes ∑ d(p, source_i)^p (default p=2 → Karcher mean). */
Eigen::Vector3d findCenter(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    int p = 2
);

} // namespace nxr::manifold::transport

namespace nxr::manifold::connection {

// ── Direction Field Design (Trivial Connections) ─────────────

struct DirectionFieldResult {
    Eigen::VectorXd connections;            // φ per edge (1-form angles)
    Eigen::MatrixXd directionVectors;       // [nF, 3] smooth direction field on faces
    Eigen::MatrixXd orthogonalVectors;      // [nF, 3] 90° rotated direction field on faces
    Eigen::MatrixXd vertexVectors;          // [nV, 3] smooth direction field on vertices
    Eigen::MatrixXd vertexOrthogonalVectors;// [nV, 3] 90° rotated direction field on vertices
    double eulerCharacteristic;
    bool gaussBonnetSatisfied;
};

// ── Smooth Direction Fields (NRoSy, Knöppel-Crane) ───────────
//
// Smoothest unit-norm direction field of order n via the
// Knöppel-Crane formulation — minimizes Dirichlet energy on
// the connection Laplacian, no prescribed singularities (they
// emerge automatically). Distinct from `trivial`
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

/** Result of `smoothVertex`. The face-based variant
 *  returns a plain Eigen::MatrixXd directly, since face stripes
 *  aren't a downstream consumer of its raw form. */
struct SmoothVertexFieldResult {
    Eigen::MatrixXd vertexVectors;   // [nV, 3] world-space, per vertex
    Eigen::VectorXd vertexFieldRaw;  // [nV * 2] raw Vector2 in vertex tangent basis
    int nSym;
};

/**
 * Compute the trivial connection 1-form φ = δβ (per-edge rotation angles).
 *
 * Solves the Poisson system A β = -K + 2π σ where:
 *   A   = d0ᵀ ★₁ d0 (cotangent Laplacian, via CholeskyCache::hodgeExact)
 *   K   = per-vertex Gaussian curvature (angle defect)
 *   σ   = singularity index at each vertex (0 for non-singular)
 *
 * Returns φ = ★₁ d₀ β as an Eigen::VectorXd of length nEdges.
 * Sign convention: φ(e) is positive in the direction of e.halfedge().
 *
 * Gauss-Bonnet requirement: Σ singularityMap values must equal χ(mesh).
 * This is NOT checked here — the caller is responsible (directionField
 * checks it; assembleTrivialConnectionLaplacian documents it).
 *
 * For sphere topology (genus 0, χ=2): two vertices with σ=1.0 each.
 * For n-RoSy fields: use σ = (desired holonomy) / (2π).
 *   e.g. nSym=4 cross field on sphere: two vertices with σ=0.5.
 */
Eigen::VectorXd computeTrivialConnection(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);

// ── Trivial-gauge per-vertex rotation ────────────────────────
//
// Integrates the trivial-connection 1-form φ (from computeTrivialConnection)
// into a per-vertex unit-complex rotation r_v = exp(iθ_v) relative to the
// Levi-Civita vertex frame, by BFS over the vertex graph from a root,
// accumulating the Levi-Civita transport modulated by exp(i·sign·φ[edge]).
// The realized trivial gauge frame is r .* vertexGrid (broadcast over the
// 3 complex columns). See bundle design spec §6.
//
// Gauss-Bonnet: Σ singularityMap values must equal χ(mesh) (the caller's
// responsibility — not checked here).
//
// Connectivity contract: BFS starts at vertex 0 and visits only its
// connected component; on a disconnected mesh other components receive
// 0+0i. Intended usage is one connected component (e.g. a single
// cortical hemisphere) at a time.
struct GaugeRotations {
    Eigen::VectorXcd vertex;   // [nV] r_v = exp(iθ_v), |r_v| = 1
};
GaugeRotations integrateTrivialGaugeRotations(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap);

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
DirectionFieldResult trivial(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);

/** Compute the smoothest face-based direction field.
 *  Returns [nF, 3] world-space vectors (principal nSym-RoSy
 *  representative; the other nSym-1 directions are obtained by
 *  rotating in the face tangent plane). */
Eigen::MatrixXd smoothFace(
    Manifold& m,
    int nSym = 4,
    bool alignToCurvature = false
);

/** Compute the smoothest vertex-based direction field (needed
 *  as input to stripe patterns).
 *  Returns the raw [nV * 2] Vector2 form alongside [nV, 3] world
 *  vectors, since stripes consume the Vector2 form internally. */
SmoothVertexFieldResult smoothVertex(
    Manifold& m,
    int nSym = 2,
    bool alignToCurvature = false
);

} // namespace nxr::manifold::connection

namespace nxr::manifold::solve {

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
HodgeResult hodge(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const Eigen::VectorXd& omega
);

} // namespace nxr::manifold::solve

namespace nxr::manifold::parametrization {

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
Eigen::MatrixXd bff(Manifold& m);

} // namespace nxr::manifold::parametrization

namespace nxr::manifold::parametrization::stripes {

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
 *  field must be 2-RoSy (use `smoothVertex(m, 2)`
 *  or pass an existing line field). */
StripePatternResult compute(
    Manifold& m,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2] in vertex tangent basis
    double uniformFrequency,
    bool connectOnSingularities = true
);

/** Stripe pattern with per-vertex target frequencies. */
StripePatternResult computeFreq(
    Manifold& m,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2]
    const Eigen::VectorXd& frequencies,     // [nV]
    bool connectOnSingularities = true
);

} // namespace nxr::manifold::parametrization::stripes

namespace nxr::manifold::geometry {

// ── Curvatures ───────────────────────────────────────────────

struct CurvatureResult {
    Eigen::VectorXd gaussian;        // K per vertex (angle defect)
    Eigen::VectorXd mean;            // H per vertex
    Eigen::VectorXd kMin;            // κ_min per vertex
    Eigen::VectorXd kMax;            // κ_max per vertex
    Eigen::MatrixXd principalDirMax; // [nV, 3] max principal direction lifted to 3D
};

CurvatureResult curvatures(Manifold& m);

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
Eigen::MatrixXd normals(Manifold& m, NormalType type);

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

FaceFrames frames(Manifold& m);

/* Per-vertex orthonormal tangent frame = geometry-central's vertexTangentBasis,
 * the 3D realization of the per-vertex tangent space (halfedgeVectorsInVertex) in
 * which the vertex connection-Laplacian's complex eigenvector coordinates live:
 *   e1 = basisX (the angle-0 / first-outgoing-halfedge axis)
 *   e2 = basisY = n x e1
 *   n  = vertex normal
 * A connection-mode coordinate z=(a,b) at vertex i decodes to a*e1_i + b*e2_i. */
struct VertexFrames {
    Eigen::MatrixXd e1;       // [nV, 3] — first tangent (basisX)
    Eigen::MatrixXd e2;       // [nV, 3] — second tangent (n × e1)
    Eigen::MatrixXd normals;  // [nV, 3] — vertex normals
};

VertexFrames vertexFrames(Manifold& m);

// ── Complex frame ("grid") ───────────────────────────────────
//
// The per-element tangent frame packed as a complex 3-vector
// c = e1 + i·e2 ∈ ℂ³ (real part = e1, imag part = e2). The unit
// normal is recovered as real(c) × imag(c) and is not stored
// separately. This is the representation Brainstorm's Geometry.*.grid
// fields use; see the bundle design spec §5.1.
Eigen::MatrixXcd vertexGrid(Manifold& m);   // [nV, 3] complex
Eigen::MatrixXcd faceGrid(Manifold& m);     // [nF, 3] complex

// ── Curvature as a 2-RoSy complex + mean ─────────────────────
//
// The shape operator (3 real DOF) split into its trace and deviatoric
// parts. `deviatoric` is the traceless symmetric part encoded as a
// 2-RoSy complex number q: arg(q)/2 is the max principal direction
// angle in the vertex tangent basis, |q| = (κmax − κmin)/2. `mean`
// is (κmax + κmin)/2. See bundle design spec §5.2. Everything else
// (principal curvatures/directions, extrinsic Gaussian) derives from
// these; intrinsic Gaussian comes from the angle-defect (2π − angleSum).
// If geometry-central principal directions are unavailable at a vertex, the
// deviatoric phase falls back to 0; only |q| (anisotropy) and mean (H) are
// reliable there.
struct VertexCurvature2RoSy {
    Eigen::VectorXcd deviatoric;  // [nV] q
    Eigen::VectorXd  mean;        // [nV] H
};
VertexCurvature2RoSy vertexCurvature(Manifold& m);

// ── Halfedge mesh topology ────────────────────────────────────
//
// Flat struct-of-arrays representation of the halfedge combinatorics,
// using 0-based indices matching geometry-central's internal storage.
// Bindings that want 1-based MATLAB indices convert at the binding edge
// (see indexVectorToMx1Based in marshal.h).
// A value of -1 (INVALID) means "none / boundary" for .face and .corner
// on exterior halfedges; all other entries are non-negative for a valid
// closed manifold mesh.
//
//   nH == 2 * nE   (every edge has exactly 2 halfedges)
//   nC == 3 * nF   (every interior face has 3 corners)

struct MeshTopology {
    // Element counts.
    int nV, nE, nF, nH, nC;

    // Reverse lookups: one representative halfedge per element (0-based).
    std::vector<long> vertexHalfedge;   // [nV]  he that "belongs to" vertex
    std::vector<long> edgeHalfedge;     // [nE]
    std::vector<long> faceHalfedge;     // [nF]
    std::vector<long> cornerHalfedge;   // [nC]

    // Halfedge adjacency table (0-based; -1 = none / exterior).
    std::vector<long> heTwin;      // [nH]
    std::vector<long> heNext;      // [nH]
    std::vector<long> heVertex;    // [nH]  tail vertex
    std::vector<long> heEdge;      // [nH]
    std::vector<long> heFace;      // [nH]  -1 for exterior halfedges
    std::vector<long> heCorner;    // [nH]  -1 for exterior halfedges

    // Boolean attributes (stored as char: 1 = true, 0 = false).
    std::vector<char> heOrientation;  // [nH]  true iff he == he.edge().halfedge()
    std::vector<char> heIsInterior;   // [nH]
};

/** Extract the halfedge mesh topology into a plain struct-of-arrays.
 *  All indices are 0-based (geometry-central convention). -1 encodes
 *  geometry-central's INVALID_IND for "none / boundary" slots. */
MeshTopology meshTopology(Manifold& m);

// ── Light per-element geometry bundle ────────────────────────
//
// All light O(V/E/F/H/C) per-element quantities, element-grouped.
// Heavy sparse operators are intentionally excluded (on-demand
// surface, separate). Frames are the complex `grid` (c = e1+i·e2);
// normals derive as real×imag. Curvature is the 2-RoSy deviatoric
// + mean (see vertexCurvature). See the bundle design spec §4.2.
//
// Corner-array sizing note: cornerAngles / cornerScaledAngles are
// sized to nH (not nC) for safety on boundary meshes, where
// geometry-central corner ids are halfedge-based and may be sparse.
// On a closed mesh nH == nC so the size is identical.
struct MeshGeometry {
    double totalArea = 0.0;
    // vertex
    Eigen::VectorXd  vertexDualAreas;           // [nV]
    Eigen::VectorXd  vertexAngleSums;           // [nV]
    Eigen::VectorXcd vertexCurvatureDeviatoric; // [nV] q (2-RoSy)
    Eigen::VectorXd  vertexMeanCurvature;       // [nV] H
    Eigen::MatrixXcd vertexGrid;                // [nV,3] c = e1+i·e2
    // edge
    Eigen::VectorXd  edgeLengths;              // [nE]
    Eigen::VectorXd  edgeCotanWeights;         // [nE]
    Eigen::VectorXd  edgeDihedralAngles;       // [nE]
    // face
    Eigen::VectorXd  faceAreas;                // [nF]
    Eigen::MatrixXd  faceCentroids;            // [nF,3]
    Eigen::MatrixXcd faceGrid;                 // [nF,3]
    // halfedge
    Eigen::VectorXd  halfedgeCotanWeights;     // [nH]
    Eigen::VectorXcd halfedgeVectorsInVertex;  // [nH]
    Eigen::VectorXcd halfedgeVectorsInFace;    // [nH]
    Eigen::VectorXcd halfedgeTransportAlong;   // [nH]
    Eigen::VectorXcd halfedgeTransportAcross;  // [nH]
    // corner (sized nH; on closed mesh nH==nC)
    Eigen::VectorXd  cornerAngles;             // [nH]
    Eigen::VectorXd  cornerScaledAngles;       // [nH]
};

MeshGeometry meshGeometry(Manifold& m);

} // namespace nxr::manifold::geometry

// ── Covariant differential operators on 3-vector cortical fields ──────────────
//
// Artifact-free transport + differentiation of a 3-vector field (e.g. a leadfield)
// expressed in per-vertex cortical frames Fv = [e1 | e2 | n] (columns), sourced from
// geometry::vertexFrames. The connection is the flat full-frame transport
// P_ij = Fj^T Fi — it accounts for the tangent rotation AND the normal tilt, and is
// path-independent (zero holonomy), which is exactly what removes the frame-rotation
// artifact (e.g. Cartesian-parallel leadfields on opposite sulcal walls reading as
// antiparallel in local frames). See the design doc.
namespace nxr::manifold::differential {

// 3x3 transport of a frame-local vector from vertex i's frame to vertex j's frame,
// P_ij = Fj^T Fi. Flat => exact for ANY pair (i, j) (adjacent or not). 0-based indices.
Eigen::Matrix3d frameTransport(Manifold& m, int i, int j);

// Per-vertex frames Fv stacked as [nV, 9], row v = Fv flattened ROW-major
// (Fv(0,0),Fv(0,1),Fv(0,2), Fv(1,0),...). Columns of Fv are [e1 | e2 | n].
Eigen::MatrixXd vertexFrameMatrices(Manifold& m);

// Lift a frame-local field to world (Cartesian): world[v] = Fv * Lloc[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToWorld(Manifold& m, const Eigen::MatrixXd& Lloc);

// Express a world (Cartesian) field in frames: Lloc[v] = Fv^T * Lworld[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToFrame(Manifold& m, const Eigen::MatrixXd& Lworld);

// Covariant gradient operator G : 3N -> 3E (component-major [a;b;c] blocks). For each
// oriented edge e: i->j, the covariant difference is delta_e = L_j - P_ij L_i. G has a +I3
// block at vertex j and a -P_ij block at vertex i. A Cartesian-constant field has G*L = 0.
Eigen::SparseMatrix<double> covariantGradient(Manifold& m);

} // namespace nxr::manifold::differential

namespace nxr::field::generate {

using nxr::manifold::Manifold;
using nxr::manifold::solve::EigenResult;
using nxr::manifold::ops::ManifoldOperators;
using nxr::manifold::ops::DECOperators;

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
// α / β / γ contributions, run hodge, verify the
// algorithm recovers what was put in.

/** Sparse delta vector: zeros everywhere except the listed source
 *  vertices, which take the supplied values. Useful as an initial
 *  condition for heat-method / Poisson workflows. */
Eigen::VectorXd delta(int nV, const std::map<int, double>& sources);

/** Random vertex scalar field, uniform in [-1, 1]. Reproducible by
 *  seed. Used as the α (gradient potential) in decomposed-1-form
 *  synthesis. */
Eigen::VectorXd randomVertexScalar(int nV, unsigned int seed = 42);

/** Random face scalar field, uniform in [-1, 1]. Reproducible by
 *  seed. Used as the β (curl potential) in decomposed-1-form
 *  synthesis. */
Eigen::VectorXd randomFaceScalar(int nF, unsigned int seed = 42);

/** Generate a random 1-form on edges. */
Eigen::VectorXd randomOmega(int nE, unsigned int seed = 42);

/** Extract the k-th eigenmode from a precomputed EigenResult as a
 *  vertex scalar field. */
Eigen::VectorXd eigenmodeField(const EigenResult& eig, int k_index);

/** Synthesize a 1-form on edges by combining independently-random
 *  exact and co-exact contributions:
 *
 *      ω = α_strength · (d0 · α_rand) + β_strength · (★₁⁻¹ d1ᵀ · β_rand)
 *
 *  where α_rand is a random vertex scalar (seed) and β_rand is a
 *  random face scalar (seed + 1). The Hodge decomposition of ω
 *  must recover the same dα and δβ within solver tolerance — this
 *  is the validation contract for hodge.
 *
 *  γ_strength is reserved for the harmonic component but ignored in
 *  v1; harmonic basis computation is deferred (cortical meshes are
 *  genus 3-5 due to ventricle topology, so the harmonic space is
 *  non-trivial in principle, but its construction needs its own
 *  primitive and isn't required for the gpjs-style validation). */
Eigen::VectorXd randomDecomposed1Form(
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
// the caller has already run normalize() on solve
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
 *  Agnostic in M: works on lumped mass (mesh) or node-weight
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
Eigen::MatrixXf heatDiffusion(
    const Eigen::SparseMatrix<double>& M,
    const EigenResult& eig,
    const Eigen::VectorXd& u0,
    const std::vector<double>& timesteps,
    double alpha = 1.0
);

/** Convenience overload: forwards to the (M, …) form using ops.mass. */
inline Eigen::MatrixXf heatDiffusion(
    const ManifoldOperators& ops,
    const EigenResult& eig,
    const Eigen::VectorXd& u0,
    const std::vector<double>& timesteps,
    double alpha = 1.0
) {
    return heatDiffusion(ops.mass, eig, u0, timesteps, alpha);
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
Eigen::MatrixXf dampedWave(
    const EigenResult& eig,
    const std::vector<int>& modeIndices,
    const std::vector<double>& amplitudes,
    const std::vector<double>& dampings,
    const std::vector<double>& phases,
    const std::vector<double>& timesteps
);

} // namespace nxr::field::generate

namespace nxr::field::interp {

using nxr::manifold::Manifold;
using nxr::manifold::ops::DECOperators;

// ── Vector Field Operations ──────────────────────────────────

/** Whitney interpolation: edge 1-form → face-centered 3D vectors.
 *  Formula: V(f) = (N × (a + b + c)) / (6A)
 *  Returns [nF, 3] matrix (row-major vectors). */
Eigen::MatrixXd whitney(
    Manifold& m,
    const DECOperators& dec,
    const Eigen::VectorXd& oneForm
);

} // namespace nxr::field::interp

namespace nxr::field::op {

using nxr::manifold::Manifold;

/** Gradient of a vertex scalar field → face-centered 3D vectors.
 *  For triangle (vi, vj, vk): ∇u = (1 / 2A) Σ (uₖ - uᵢ) (N × eⱼᵢ)
 *  Returns [nF, 3] matrix. */
Eigen::MatrixXd gradient(
    Manifold& m,
    const Eigen::VectorXd& scalarField
);

} // namespace nxr::field::op

namespace nxr::field::extract {

using nxr::manifold::Manifold;

// ── Isolines ─────────────────────────────────────────────────

struct IsolineResult {
    Eigen::MatrixXd positions;  // [segmentCount * 2, 3] endpoints of line segments
    int segmentCount;
};

/** Extract isoline (contours) from a per-vertex scalar field.
 *  numLevels evenly-spaced contour values in [minValue, maxValue].
 *  If minValue == maxValue, auto-detects range from the data. */
IsolineResult isoline(
    Manifold& m,
    const Eigen::VectorXd& scalarField,
    int numLevels = 20,
    double minValue = 0.0,
    double maxValue = 0.0
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
StreamlineResult streamline(
    Manifold& m,
    const Eigen::MatrixXd& faceField,
    int numSeeds = 15,
    double stepCoef = 0.15,
    int maxSteps = 1000
);

} // namespace nxr::field::extract

// ── Backward-compat using-declarations ───────────────────────
//
// Existing callers (addon.cpp, MEX bindings, tests, downstream
// consumers) reference these symbols at `nxr::manifold::*`. The
// sub-namespace split above is a pure organization change; the
// using-declarations below preserve every public name at its
// pre-split location.

namespace nxr::manifold {
    // Type using-declarations only — function using-decls conflict with
    // out-of-line definitions in src/*.cpp (the .cpp opens a sub-namespace
    // block like `namespace nxr::manifold::connection { trivial(...) { … } }`
    // and the using-decl `using connection::trivial` is treated as
    // declaring `nxr::manifold::trivial`, which clashes with the
    // canonical sub-namespace definition). Types don't have this issue.
    //
    // Consumers must use the sub-namespace paths for free functions
    // (e.g. `nxr::manifold::ops::d0(m)`, `nxr::manifold::solve::eigen(...)`).
    using ops::MassMatrixVariant;
    using ops::ManifoldOperators;
    using ops::DECOperators;
    using ops::CholeskyCache;
    using ops::laplacian::connection::ConnectionDomain;
    using ops::laplacian::connection::ConnectionLaplacianFormat;
    using ops::laplacian::connection::ConnectionLaplacianOptions;
    using ops::laplacian::connection::ConnectionLaplacian;
    using solve::EigenResult;
    using solve::HeatGeodesicSolver;
    using solve::SignedHeatSolver;
    using solve::SignedHeatLevelSet;
    using solve::HodgeResult;
    using transport::VectorHeatSolver;
    using transport::LogMapResult;
    using transport::LogMapStrategy;
    using connection::DirectionFieldResult;
    using connection::SmoothVertexFieldResult;
    using parametrization::stripes::StripePatternResult;
    using geometry::CurvatureResult;
    using geometry::NormalType;
    using geometry::FaceFrames;
    using geometry::VertexFrames;
    using geometry::MeshTopology;
} // namespace nxr::manifold
