#include "nxr/facets.h"
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

int main() {
    testTopologyFacet();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
