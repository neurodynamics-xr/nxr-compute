#include "nxr/facets.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

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
} // namespace nxr::manifold::facet
