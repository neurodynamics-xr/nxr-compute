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

} // namespace nxr::manifold

namespace nxr::manifold::facet {
int TopologyFacet::nV() const { return m_.nV(); }
int TopologyFacet::nE() const { return m_.nE(); }
int TopologyFacet::nF() const { return m_.nF(); }
int TopologyFacet::nH() const { return m_.topologyData().nH; }
int TopologyFacet::nC() const { return m_.topologyData().nC; }
int TopologyFacet::eulerCharacteristic() const { return m_.nV() - m_.nE() + m_.nF(); }
const geometry::MeshTopology& TopologyFacet::all() const { return m_.topologyData(); }
} // namespace nxr::manifold::facet
