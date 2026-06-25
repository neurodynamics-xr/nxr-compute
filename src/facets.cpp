#include "nxr/facets.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <cmath>
#include <string>

namespace nxr::manifold {

const geometry::MeshGeometry& Manifold::lightGeometry() {
    if (!lightGeometryCache_)
        lightGeometryCache_ =
            std::make_unique<geometry::MeshGeometry>(geometry::meshGeometry(*this));
    return *lightGeometryCache_;
}
const geometry::MeshTopology& Manifold::topologyData() {
    if (!topologyDataCache_)
        topologyDataCache_ =
            std::make_unique<geometry::MeshTopology>(geometry::meshTopology(*this));
    return *topologyDataCache_;
}
const Eigen::MatrixXd& Manifold::vertexPositions() const { return vertexPositions_; }
const Eigen::MatrixXi& Manifold::faces() const { return faces_; }

facet::TopologyFacet  Manifold::topology()  { return facet::TopologyFacet(*this); }
facet::EmbeddedFacet  Manifold::embedded()  { return facet::EmbeddedFacet(*this); }
facet::IntrinsicFacet Manifold::intrinsic() { return facet::IntrinsicFacet(*this); }
facet::ExtrinsicFacet Manifold::extrinsic() { return facet::ExtrinsicFacet(*this); }

// ── Gauge workflow ────────────────────────────────────────────

int Manifold::eulerCharacteristic() const { return nV() - nE() + nF(); }
GaugeType Manifold::activeGaugeType() const { return activeGaugeType_; }
const std::map<int,double>& Manifold::activeSingularities() const { return activeSingularities_; }

void Manifold::validateSingularities_(const std::map<int,double>& s) const {
    if (s.empty())
        throw Error(ErrorCode::InvalidInput,
            "Trivial gauge requires a singularity map; Levi-Civita needs none.",
            "Pass a {vertexIndex -> index} map whose values sum to the Euler "
            "characteristic chi = " + std::to_string(eulerCharacteristic()) + ".");
    double sum = 0.0;
    for (auto& kv : s) sum += kv.second;
    const int chi = eulerCharacteristic();
    if (std::abs(sum - static_cast<double>(chi)) > 1e-9)
        throw Error(ErrorCode::InvalidInput,
            "Gauss-Bonnet violated: singularity indices sum to " + std::to_string(sum) +
            " but must equal the Euler characteristic chi = " + std::to_string(chi) + ".",
            "A closed genus-0 surface has chi=2 (e.g. two +1 singularities); a disk chi=1.");
}

void Manifold::setGauge(GaugeType type, const std::map<int,double>& singularities) {
    if (type == GaugeType::Trivial) {
        validateSingularities_(singularities);
    } else if (!singularities.empty()) {
        // Fail loud rather than silently discard a caller's singularities.
        throw Error(ErrorCode::InvalidInput,
            "setGauge: singularities are only meaningful for the trivial gauge.",
            "Levi-Civita / Euclidean gauges take no singularities — pass an empty map.");
    }
    activeGaugeType_ = type;
    activeSingularities_ = (type == GaugeType::Trivial) ? singularities : std::map<int,double>{};
    // Gauge-dependent operators (spec §5b: connection/covariant source from
    // intrinsic + active gauge) are now stale — drop them so the next request
    // rebuilds in the new gauge.
    releaseOperator(OperatorId::LaplacianConnection);
    releaseOperator(OperatorId::LaplacianCovariant);
    // NOTE: ConnectionGradient is NOT invalidated here — v1 builds it in the
    // Levi-Civita gauge only (assembleConnectionGradient reads the LC transport
    // from operatorGeometry()), so it is gauge-independent. The trivial-gauge
    // gradient is a deferred follow-up; add the invalidation when it lands.
}

// ── B2: Lazy DEC / Cholesky cache + gauge() overload bodies ──────────────────

ops::DECOperators& Manifold::decOperators() {
    if (!decCache_)
        decCache_ = std::make_unique<ops::DECOperators>(ops::assembleDECOperators(*this));
    return *decCache_;
}

ops::CholeskyCache& Manifold::choleskyCache() {
    if (!choleskyCachePtr_)
        choleskyCachePtr_ = std::make_unique<ops::CholeskyCache>();
    return *choleskyCachePtr_;
}

facet::GaugeFacet Manifold::gauge() {
    return facet::GaugeFacet(*this, activeGaugeType_, activeSingularities_);
}

facet::GaugeFacet Manifold::gauge(GaugeType type, const std::map<int,double>& singularities) {
    if (type == GaugeType::Trivial) validateSingularities_(singularities);
    return facet::GaugeFacet(*this, type,
                             type == GaugeType::Trivial ? singularities : std::map<int,double>{});
}

// ── C1: Operators facet + independent-cache management ───────────────────────

facet::OperatorsFacet Manifold::operators() { return facet::OperatorsFacet(*this); }

bool Manifold::isOperatorCached(OperatorId id) const {
    switch (id) {
        case OperatorId::LaplacianCotan:      return (bool)cacheLaplacianCotan_;
        case OperatorId::LaplacianGraph:      return (bool)cacheLaplacianGraph_;
        case OperatorId::LaplacianConnection: return (bool)cacheLaplacianConnection_;
        case OperatorId::LaplacianCovariant:  return (bool)cacheLaplacianCovariant_;
        case OperatorId::Dec:                 return (bool)decCache_;
        case OperatorId::MassLumped:          return (bool)cacheMassLumped_;
        case OperatorId::MassGalerkin:        return (bool)cacheMassGalerkin_;
        case OperatorId::Gradient3D:          return (bool)cacheGradient3D_;
        case OperatorId::ExtrinsicWeitzenbock: return (bool)cacheExtrinsicWeitzenbock_;
        case OperatorId::Dirac:               return (bool)cacheDirac_;
        case OperatorId::DiracFace:           return (bool)cacheDiracFace_;
        case OperatorId::DiracD:              return (bool)cacheDiracD_;
        case OperatorId::DiracFaceD:          return (bool)cacheDiracFaceD_;
        case OperatorId::DiracIntrinsicD:     return (bool)cacheDiracIntrinsicD_;
        case OperatorId::DiracFaceIntrinsicD: return (bool)cacheDiracFaceIntrinsicD_;
        case OperatorId::GradFace:            return (bool)cacheGradFace_;
        case OperatorId::LapFace:             return (bool)cacheLapFace_;
        case OperatorId::ConnectionGradient:  return (bool)cacheConnectionGradient_;
    }
    return false;
}

void Manifold::releaseOperator(OperatorId id) {
    switch (id) {
        case OperatorId::LaplacianCotan:      cacheLaplacianCotan_.reset();      break;
        case OperatorId::LaplacianGraph:      cacheLaplacianGraph_.reset();      break;
        case OperatorId::LaplacianConnection: cacheLaplacianConnection_.reset(); break;
        case OperatorId::LaplacianCovariant:  cacheLaplacianCovariant_.reset();  break;
        case OperatorId::Dec:                 decCache_.reset();                 break;
        case OperatorId::MassLumped:          cacheMassLumped_.reset();          break;
        case OperatorId::MassGalerkin:        cacheMassGalerkin_.reset();        break;
        case OperatorId::Gradient3D:          cacheGradient3D_.reset();          break;
        case OperatorId::ExtrinsicWeitzenbock: cacheExtrinsicWeitzenbock_.reset(); break;
        case OperatorId::Dirac:               cacheDirac_.reset();               break;
        case OperatorId::DiracFace:           cacheDiracFace_.reset();           break;
        case OperatorId::DiracD:              cacheDiracD_.reset();              break;
        case OperatorId::DiracFaceD:          cacheDiracFaceD_.reset();          break;
        case OperatorId::DiracIntrinsicD:     cacheDiracIntrinsicD_.reset();     break;
        case OperatorId::DiracFaceIntrinsicD: cacheDiracFaceIntrinsicD_.reset();  break;
        case OperatorId::GradFace:            cacheGradFace_.reset();            break;
        case OperatorId::LapFace:             cacheLapFace_.reset();             break;
        case OperatorId::ConnectionGradient:
            cacheConnectionGradient_.reset();
            cachedConnectionGradientNSym_ = -1;
            break;
    }
}

// Private cache-fill helpers for LaplacianView.
// cotan: sources DIRECTLY from operatorGeometry().cotanLaplacian — does NOT
// call assembleManifoldOperators (which fuses cotan+mass+normals).
const Eigen::SparseMatrix<double>& Manifold::cotanLaplacianCached_() {
    if (!cacheLaplacianCotan_) {
        auto& geom = operatorGeometry();
        geom.requireCotanLaplacian();
        cacheLaplacianCotan_ =
            std::make_unique<Eigen::SparseMatrix<double>>(geom.cotanLaplacian);
    }
    return *cacheLaplacianCotan_;
}

const Eigen::SparseMatrix<double>& Manifold::graphLaplacianCached_() {
    if (!cacheLaplacianGraph_)
        cacheLaplacianGraph_ =
            std::make_unique<Eigen::SparseMatrix<double>>(ops::graphLaplacian(*this));
    return *cacheLaplacianGraph_;
}

// ── C2: Mass cache-fill helpers ───────────────────────────────────────────────
// Each sources DIRECTLY from operatorGeometry()'s GC require* — completely
// independent: requesting lumped NEVER triggers galerkin and vice-versa.

const Eigen::SparseMatrix<double>& Manifold::massLumpedCached_() {
    if (!cacheMassLumped_) {
        auto& geom = operatorGeometry();
        geom.requireVertexLumpedMassMatrix();
        cacheMassLumped_ =
            std::make_unique<Eigen::SparseMatrix<double>>(geom.vertexLumpedMassMatrix);
    }
    return *cacheMassLumped_;
}

const Eigen::SparseMatrix<double>& Manifold::massGalerkinCached_() {
    if (!cacheMassGalerkin_) {
        auto& geom = operatorGeometry();
        geom.requireVertexGalerkinMassMatrix();
        cacheMassGalerkin_ =
            std::make_unique<Eigen::SparseMatrix<double>>(geom.vertexGalerkinMassMatrix);
    }
    return *cacheMassGalerkin_;
}

const Eigen::SparseMatrix<double>& Manifold::gradient3DCached_() {
    if (!cacheGradient3D_)
        cacheGradient3D_ = std::make_unique<Eigen::SparseMatrix<double>>(
            differential::covariantGradient(*this));
    return *cacheGradient3D_;
}

const Eigen::SparseMatrix<double>& Manifold::extrinsicWeitzenbockCached_() {
    if (!cacheExtrinsicWeitzenbock_)
        cacheExtrinsicWeitzenbock_ = std::make_unique<Eigen::SparseMatrix<double>>(
            differential::assembleExtrinsicWeitzenbock(*this));
    return *cacheExtrinsicWeitzenbock_;
}

const Eigen::SparseMatrix<double>& Manifold::diracExtrinsicBlockCached_() {
    if (!cacheDirac_)
        cacheDirac_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::extrinsicBlock(*this));
    return *cacheDirac_;
}

// First-order Dirac D [4F×4V] — the rectangular operator E = DᵀW_F D is built
// from. Cached independently of E: requesting diracD() never builds E, and a
// dirac(τ) sweep does not force D (E is its own cache slot).
const Eigen::SparseMatrix<double>& Manifold::diracMatrixCached_() {
    if (!cacheDiracD_)
        cacheDiracD_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::matrix(*this));
    return *cacheDiracD_;
}

// First-order INTRINSIC Dirac D_int [4F×4V] — immersion/edge-based (spin-connection
// root). Cached independently of the extrinsic D and of the intrinsic square.
const Eigen::SparseMatrix<double>& Manifold::diracIntrinsicMatrixCached_() {
    if (!cacheDiracIntrinsicD_)
        cacheDiracIntrinsicD_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::matrixIntrinsic(*this));
    return *cacheDiracIntrinsicD_;
}

// L(τ) = (1−τ)(cotanL ⊗ I₄) + τ·E. Builds each term only when its coefficient is
// nonzero — τ=0 never assembles E; τ=1 never builds the intrinsic block. The
// τ==0.0 / τ==1.0 fast paths use exact equality deliberately: τ arrives unmodified
// from the binding-layer dispatch (a literal or a single mxGetScalar), never from
// arithmetic, so the constants are exact and an epsilon test would only blur them.
Eigen::SparseMatrix<double> Manifold::diracFamily_(double tau) {
    // Negated form so NaN is rejected too (NaN fails every ordered comparison, so
    // `tau < 0 || tau > 1` would let it through into a silent NaN-valued matrix).
    if (!(tau >= 0.0 && tau <= 1.0))
        throw Error(ErrorCode::InvalidInput, "dirac: tau must be in [0,1]",
                    "Got tau=" + std::to_string(tau) + ".");
    const int N = nV();

    Eigen::SparseMatrix<double> L4;
    if (tau < 1.0) {
        const auto& cotanL = cotanLaplacianCached_();    // sources operatorGeometry()
        std::vector<Eigen::Triplet<double>> T;
        T.reserve(static_cast<size_t>(cotanL.nonZeros()) * 4);
        for (int k = 0; k < cotanL.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(cotanL, k); it; ++it)
                for (int c = 0; c < 4; ++c)
                    T.emplace_back(4 * static_cast<int>(it.row()) + c,
                                   4 * static_cast<int>(it.col()) + c, it.value());
        L4.resize(4 * N, 4 * N);
        L4.setFromTriplets(T.begin(), T.end());
    }
    if (tau == 0.0) { L4.makeCompressed(); return L4; }
    const auto& E = diracExtrinsicBlockCached_();
    if (tau == 1.0) { Eigen::SparseMatrix<double> out = E; out.makeCompressed(); return out; }
    Eigen::SparseMatrix<double> L = (1.0 - tau) * L4 + tau * E;
    L.makeCompressed();
    return L;
}

const Eigen::SparseMatrix<double>& Manifold::diracFaceExtrinsicBlockCached_() {
    if (!cacheDiracFace_)
        cacheDiracFace_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::extrinsicBlockFace(*this));
    return *cacheDiracFace_;
}

// First-order face-domain Dirac D̃ [4V×4F]. Cached independently of Ẽ. Throws
// on an open boundary (closed-mesh v1), inheriting matrixFace's guard.
const Eigen::SparseMatrix<double>& Manifold::diracFaceMatrixCached_() {
    if (!cacheDiracFaceD_)
        cacheDiracFaceD_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::matrixFace(*this));
    return *cacheDiracFaceD_;
}

// First-order INTRINSIC face-domain Dirac D̃_int [4V×4F] (centroid immersion dual of
// matrixFace). Cached; throws on an open boundary (closed-mesh v1).
const Eigen::SparseMatrix<double>& Manifold::diracFaceIntrinsicMatrixCached_() {
    if (!cacheDiracFaceIntrinsicD_)
        cacheDiracFaceIntrinsicD_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::dirac::matrixFaceIntrinsic(*this));
    return *cacheDiracFaceIntrinsicD_;
}

// Barycentric dual-mesh face gradient G̃ [3F×F]. Cached; throws on an open boundary.
const Eigen::SparseMatrix<double>& Manifold::gradFaceCached_() {
    if (!cacheGradFace_)
        cacheGradFace_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::facegrad::gradient(*this));
    return *cacheGradFace_;
}

// Face Laplacian K̃ = G̃ᵀ⋆_F G̃ [F×F]. Cached; built from gradFace (never drifts).
const Eigen::SparseMatrix<double>& Manifold::lapFaceCached_() {
    if (!cacheLapFace_)
        cacheLapFace_ = std::make_unique<Eigen::SparseMatrix<double>>(
            ops::facegrad::laplacian(*this));
    return *cacheLapFace_;
}

// K̃ = d₁⋆₁⁻¹d₁ᵀ, the DEC 2-form Laplacian ([F×F]). Cached: depends only on the
// (mesh-fixed) cached DEC operators, so a diracFace(τ) sweep reuses it instead of
// re-running the sparse triple product each call. d1t is an explicit temporary
// (different object from dec.d1) so the d1·⋆₁⁻¹·d1ᵀ chain has no Eigen aliasing.
const Eigen::SparseMatrix<double>& Manifold::twoFormLaplacianCached_() {
    if (!cacheTwoFormLaplacian_) {
        const ops::DECOperators& dec = decOperators();
        Eigen::SparseMatrix<double> d1t = dec.d1.transpose();
        cacheTwoFormLaplacian_ = std::make_unique<Eigen::SparseMatrix<double>>(
            dec.d1 * dec.hodge1Inverse * d1t);
    }
    return *cacheTwoFormLaplacian_;
}

// L̃(τ) = (1−τ)(K̃ ⊗ I₄) + τ·Ẽ, K̃ = d₁⋆₁⁻¹d₁ᵀ (DEC 2-form Laplacian). Builds each
// term only when its coefficient is nonzero — τ=0 never assembles Ẽ; τ=1 never
// builds K̃. NaN-safe guard + exact-fast-path reasoning as diracFamily_. K̃ is read
// from its cache (mirrors the vertex diracFamily_ reading cotanLaplacianCached_).
Eigen::SparseMatrix<double> Manifold::diracFaceFamily_(double tau) {
    if (!(tau >= 0.0 && tau <= 1.0))
        throw Error(ErrorCode::InvalidInput, "diracFace: tau must be in [0,1]",
                    "Got tau=" + std::to_string(tau) + ".");
    const int Fn = nF();

    Eigen::SparseMatrix<double> K4;
    if (tau < 1.0) {
        const Eigen::SparseMatrix<double>& Ktilde = twoFormLaplacianCached_();  // [F×F], cached
        std::vector<Eigen::Triplet<double>> T;
        T.reserve(static_cast<size_t>(Ktilde.nonZeros()) * 4);
        for (int k = 0; k < Ktilde.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(Ktilde, k); it; ++it)
                for (int c = 0; c < 4; ++c)
                    T.emplace_back(4 * static_cast<int>(it.row()) + c,
                                   4 * static_cast<int>(it.col()) + c, it.value());
        K4.resize(4 * Fn, 4 * Fn);
        K4.setFromTriplets(T.begin(), T.end());
    }
    if (tau == 0.0) { K4.makeCompressed(); return K4; }
    const auto& E = diracFaceExtrinsicBlockCached_();
    if (tau == 1.0) { Eigen::SparseMatrix<double> out = E; out.makeCompressed(); return out; }
    Eigen::SparseMatrix<double> L = (1.0 - tau) * K4 + tau * E;
    L.makeCompressed();
    return L;
}

// ── C3: Connection / Covariant Laplacian cache-fill helpers ──────────────────
// connection: builds in the ACTIVE gauge (LC or trivial). Stores K_complex.
// covariant: calls connection() + gauge().grid() + cotanLaplacianCached_() and
// assembles the 3N×3N real symmetric matrix.

const Eigen::SparseMatrix<std::complex<double>>& Manifold::connectionLaplacianCached_() {
    if (!cacheLaplacianConnection_) {
        namespace cl = ops::laplacian::connection;
        cl::ConnectionLaplacianOptions o;
        o.domain = cl::ConnectionDomain::Vertex;
        o.nSym   = 1;
        o.format = cl::ConnectionLaplacianFormat::Complex;
        cl::ConnectionLaplacian r =
            (activeGaugeType_ == GaugeType::Trivial)
              ? cl::assembleTrivialConnectionLaplacian(*this, activeSingularities_,
                                                       decOperators(), choleskyCache(), o)
              : cl::assembleConnectionLaplacian(*this, o);
        cacheLaplacianConnection_ =
            std::make_unique<Eigen::SparseMatrix<std::complex<double>>>(std::move(r.K_complex));
    }
    return *cacheLaplacianConnection_;
}

const Eigen::SparseMatrix<double>& Manifold::covariantLaplacianCached_(
        ops::laplacian::connection::CovariantCoupling coupling) {
    // Rebuild when the slot is empty OR the cached coupling differs — the two
    // couplings (Product / Ambient) produce different matrices, so the cache
    // must key on the coupling, not just on presence.
    if (!cacheLaplacianCovariant_ ||
        cachedCovariantCoupling_ != static_cast<int>(coupling)) {
        namespace cl = ops::laplacian::connection;
        Eigen::MatrixXcd g  = gauge().grid();                     // realized active-gauge frame
        const auto& cotanL = cotanLaplacianCached_();             // real cotan Laplacian
        // The Ambient covariant (the default 3D-frame view) is built purely from
        // the frames + scalar weights and does NOT use the connection Laplacian K
        // (frame-conjugate of kron(I3, cotanL)). Only the Product coupling — which
        // puts the intrinsic connection on the tangent block — needs K, so we skip
        // the connection assembly (incl. the trivial-gauge Poisson solve) otherwise.
        if (coupling == cl::CovariantCoupling::Product) {
            const auto& K = connectionLaplacianCached_();
            cacheLaplacianCovariant_ =
                std::make_unique<Eigen::SparseMatrix<double>>(
                    cl::assembleCovariantLaplacian(coupling, K, g, cotanL));
        } else {
            const Eigen::SparseMatrix<std::complex<double>> noK;  // unused by Ambient
            cacheLaplacianCovariant_ =
                std::make_unique<Eigen::SparseMatrix<double>>(
                    cl::assembleCovariantLaplacian(coupling, noK, g, cotanL));
        }
        cachedCovariantCoupling_ = static_cast<int>(coupling);
    }
    return *cacheLaplacianCovariant_;
}

} // namespace nxr::manifold

namespace nxr::manifold::facet {
int TopologyFacet::nV() const { return m_.nV(); }
int TopologyFacet::nE() const { return m_.nE(); }
int TopologyFacet::nF() const { return m_.nF(); }
int TopologyFacet::nH() const { return m_.topologyData().nH; }
int TopologyFacet::nC() const { return m_.topologyData().nC; }
int TopologyFacet::eulerCharacteristic() const { return m_.nV() - m_.nE() + m_.nF(); }
const geometry::MeshTopology& TopologyFacet::all() const { return m_.topologyData(); }

Eigen::MatrixXd  EmbeddedFacet::VertexView::position() const { return m.vertexPositions(); }
Eigen::MatrixXd  EmbeddedFacet::VertexView::normal()   const { return geometry::vertexFrames(m).normals; }
Eigen::MatrixXcd EmbeddedFacet::VertexView::grid()     const { return m.lightGeometry().vertexGrid; }
Eigen::MatrixXd  EmbeddedFacet::FaceView::normal()     const { return geometry::frames(m).normals; }
Eigen::MatrixXcd EmbeddedFacet::FaceView::grid()       const { return m.lightGeometry().faceGrid; }
Eigen::MatrixXd  EmbeddedFacet::FaceView::centroid()   const { return m.lightGeometry().faceCentroids; }
Eigen::VectorXd  EmbeddedFacet::FaceView::area()       const { return m.lightGeometry().faceAreas; }

Eigen::VectorXd  IntrinsicFacet::VertexView::dualArea() const { return m.lightGeometry().vertexDualAreas; }
Eigen::VectorXd  IntrinsicFacet::VertexView::angleSum() const { return m.lightGeometry().vertexAngleSums; }
Eigen::VectorXd  IntrinsicFacet::EdgeView::length()       const { return m.lightGeometry().edgeLengths; }
Eigen::VectorXd  IntrinsicFacet::EdgeView::cotanWeight()  const { return m.lightGeometry().edgeCotanWeights; }
Eigen::VectorXd  IntrinsicFacet::HalfedgeView::cotanWeight()     const { return m.lightGeometry().halfedgeCotanWeights; }
Eigen::VectorXcd IntrinsicFacet::HalfedgeView::transportAlong()  const { return m.lightGeometry().halfedgeTransportAlong; }
Eigen::VectorXcd IntrinsicFacet::HalfedgeView::transportAcross() const { return m.lightGeometry().halfedgeTransportAcross; }

Eigen::VectorXcd ExtrinsicFacet::VertexView::curvature2RoSy() const { return m.lightGeometry().vertexCurvatureDeviatoric; }
Eigen::VectorXd  ExtrinsicFacet::VertexView::meanCurvature()  const { return m.lightGeometry().vertexMeanCurvature; }
Eigen::MatrixXd  ExtrinsicFacet::VertexView::principalDir()   const { return geometry::curvatures(m).principalDirMax; }
Eigen::VectorXd  ExtrinsicFacet::EdgeView::dihedralAngle()    const { return m.lightGeometry().edgeDihedralAngles; }

// ── B2: GaugeFacet::grid() ────────────────────────────────────────────────────

Eigen::MatrixXcd GaugeFacet::grid() const {
    Eigen::MatrixXcd g = geometry::vertexGrid(m_);            // LC / euclidean base frame
    if (type_ != GaugeType::Trivial) return g;
    connection::GaugeRotations gr = connection::integrateTrivialGaugeRotations(
        m_, m_.decOperators(), m_.choleskyCache(), sing_);
    for (int v = 0; v < g.rows(); ++v) g.row(v) *= gr.vertex(v);  // combing rotation .* grid
    return g;
}

Eigen::MatrixXcd GaugeFacet::faceGrid() const {
    Eigen::MatrixXcd g = geometry::faceGrid(m_);              // LC / euclidean base frame
    if (type_ != GaugeType::Trivial) return g;
    connection::GaugeRotations gr = connection::integrateTrivialGaugeRotations(
        m_, m_.decOperators(), m_.choleskyCache(), sing_);
    for (int f = 0; f < g.rows(); ++f) g.row(f) *= gr.face(f);    // combing rotation .* grid
    return g;
}

// ── C1: OperatorsFacet::LaplacianView bodies ─────────────────────────────────
// LaplacianView delegates through OperatorsFacet (which is friend of Manifold),
// calling the private Manifold cache-fill helpers.

const Eigen::SparseMatrix<double>& OperatorsFacet::LaplacianView::cotan() const {
    return m.cotanLaplacianCached_();
}

const Eigen::SparseMatrix<double>& OperatorsFacet::LaplacianView::graph() const {
    return m.graphLaplacianCached_();
}

// ── C3: connection / covariant bodies ────────────────────────────────────────
// Both delegate through the private Manifold cache-fill helpers so that
// connection() can be called standalone without dragging in covariant, and
// vice-versa. The covariant helper depends on connection + cotan being built,
// but the cotan slot is also independently reusable for other callers.

const Eigen::SparseMatrix<std::complex<double>>&
OperatorsFacet::LaplacianView::connection() const {
    return m.connectionLaplacianCached_();
}

const Eigen::SparseMatrix<double>&
OperatorsFacet::LaplacianView::covariant(
        ops::laplacian::connection::CovariantCoupling coupling) const {
    return m.covariantLaplacianCached_(coupling);
}

// ── C2: dec / mass / hodge bodies ────────────────────────────────────────────

// dec(): returns the lazily-cached DECOperators bundle via Manifold::decOperators().
const ops::DECOperators& OperatorsFacet::dec() const { return m_.decOperators(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::gradient3D() const { return m_.gradient3DCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::extrinsicWeitzenbock() const { return m_.extrinsicWeitzenbockCached_(); }

const Eigen::SparseMatrix<std::complex<double>>&
OperatorsFacet::connectionGradient(int nSym) const {
    if (!m_.cacheConnectionGradient_ || m_.cachedConnectionGradientNSym_ != nSym) {
        m_.cacheConnectionGradient_ = std::make_unique<Eigen::SparseMatrix<std::complex<double>>>(
            ops::laplacian::connection::assembleConnectionGradient(m_, nSym));
        m_.cachedConnectionGradientNSym_ = nSym;
    }
    return *m_.cacheConnectionGradient_;
}

Eigen::SparseMatrix<double> OperatorsFacet::dirac(double tau) const { return m_.diracFamily_(tau); }
Eigen::SparseMatrix<double> OperatorsFacet::diracFace(double tau) const { return m_.diracFaceFamily_(tau); }
const Eigen::SparseMatrix<double>& OperatorsFacet::diracD() const { return m_.diracMatrixCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::diracFaceD() const { return m_.diracFaceMatrixCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::diracFaceIntrinsicD() const { return m_.diracFaceIntrinsicMatrixCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::diracIntrinsicD() const { return m_.diracIntrinsicMatrixCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::gradFace() const { return m_.gradFaceCached_(); }
const Eigen::SparseMatrix<double>& OperatorsFacet::lapFace()  const { return m_.lapFaceCached_(); }

// MassView: each helper calls through to the independent private Manifold cache-fill.
// Holds Manifold& m (C1 pattern) — never dangles.
const Eigen::SparseMatrix<double>& OperatorsFacet::MassView::lumped()   const {
    return m.massLumpedCached_();
}
const Eigen::SparseMatrix<double>& OperatorsFacet::MassView::galerkin() const {
    return m.massGalerkinCached_();
}

// HodgeView: all four delegate through Manifold::decOperators() — the lazy DEC.
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h0()    const {
    return m.decOperators().hodge0;
}
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h1()    const {
    return m.decOperators().hodge1;
}
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h2()    const {
    return m.decOperators().hodge2;
}
const Eigen::SparseMatrix<double>& OperatorsFacet::HodgeView::h1inv() const {
    return m.decOperators().hodge1Inverse;
}

} // namespace nxr::manifold::facet
