#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

// Unit icosphere (12 verts), closed genus-0, chi = 2.
static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testTopologyFacet() {
    std::cout << "\n=== facets: topology ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto topo = m.topology();
    EXPECT(topo.nV() == 12, "topology nV");
    EXPECT(topo.nF() == 20, "topology nF");
    EXPECT(topo.nE() == 30, "topology nE");
    EXPECT(topo.eulerCharacteristic() == 2, "chi = V-E+F = 2 (closed genus-0)");
    EXPECT(topo.nH() == 60, "topology nH = 2E");
    EXPECT((int)topo.all().heTwin.size() == 60, "MeshTopology SoA wired");
    // raw-input aliases retained at construction
    EXPECT(m.faces().rows() == 20 && m.faces().cols() == 3, "faces() [20,3]");
    EXPECT(m.faces()(0,0) == 0 && m.faces()(0,1) == 11 && m.faces()(0,2) == 5, "faces() is raw input row 0");
    EXPECT(m.vertexPositions().rows() == 12, "vertexPositions() [12,3]");
}

static void testEmbeddedFacet() {
    std::cout << "\n=== facets: embedded ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto e = m.embedded();
    EXPECT(e.vertex().position().rows() == 12 && e.vertex().position().cols() == 3, "vertex.position [12,3]");
    // raw-input identity: position row 0 equals the input
    EXPECT(std::abs(e.vertex().position()(0,0) - (-1.0)) < 1e-12, "vertex.position is raw input");
    EXPECT(e.vertex().grid().rows() == 12 && e.vertex().grid().cols() == 3, "vertex.grid [12,3] complex");
    // facet-identity: embedded.vertex.grid == geometry::vertexGrid(m)
    EXPECT((e.vertex().grid() - geometry::vertexGrid(m)).cwiseAbs().maxCoeff() < 1e-12, "vertex.grid == vertexGrid(m)");
    EXPECT(e.face().grid().rows() == 20, "face.grid [20,3]");
    EXPECT(e.face().centroid().rows() == 20 && e.face().centroid().cols() == 3, "face.centroid [20,3]");
    EXPECT(e.vertex().normal().rows() == 12, "vertex.normal [12,3]");
    // grid encodes the normal: Re(c) x Im(c) is unit and aligns with vertex.normal
    Eigen::Vector3d e1 = e.vertex().grid().row(0).real().transpose();
    Eigen::Vector3d e2 = e.vertex().grid().row(0).imag().transpose();
    Eigen::Vector3d nFromGrid = e1.cross(e2);
    EXPECT(std::abs(nFromGrid.norm() - 1.0) < 1e-9, "Re(c) x Im(c) is unit normal");
}

static void testIntrinsicFacet() {
    std::cout << "\n=== facets: intrinsic ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto in = m.intrinsic();
    EXPECT(in.vertex().dualArea().size() == 12, "vertex.dualArea [12]");
    EXPECT(in.vertex().angleSum().size() == 12, "vertex.angleSum [12]");
    EXPECT(in.edge().length().size() == 30, "edge.length [30]");
    EXPECT(in.edge().cotanWeight().size() == 30, "edge.cotanWeight [30]");
    EXPECT(in.halfedge().transportAlong().size() == 60, "halfedge.transportAlong [60]");
    EXPECT(in.halfedge().transportAcross().size() == 60, "halfedge.transportAcross [60]");
    EXPECT(in.halfedge().cotanWeight().size() == 60, "halfedge.cotanWeight [60]");
    // dual areas sum to total area (closed mesh)
    EXPECT(std::abs(in.vertex().dualArea().sum() - m.lightGeometry().totalArea) < 1e-9, "dualArea sums to totalArea");
    // facet-identity: intrinsic.edge.length matches a direct GC require
    auto& g = m.operatorGeometry(); g.requireEdgeLengths();
    Eigen::VectorXd direct(30);
    for (auto edge : m.mesh().edges()) direct(edge.getIndex()) = g.edgeLengths[edge];
    EXPECT((in.edge().length() - direct).cwiseAbs().maxCoeff() < 1e-12, "edge.length == GC edgeLengths");
}

int main() {
    testTopologyFacet();
    testEmbeddedFacet();
    testIntrinsicFacet();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
