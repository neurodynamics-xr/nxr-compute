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
    for (int v = 0; v < g.rows(); ++v) g.row(v) *= gr.vertex(v);  // exp(i phi_v) .* grid
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

// connection() and covariant() are declared in facets.h for Task C3.
// They are not implemented in C1 — calling them will produce a link error.

} // namespace nxr::manifold::facet
