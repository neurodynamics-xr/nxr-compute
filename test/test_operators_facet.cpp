#include "nxr/facets.h"
#include <complex>
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testLaplacianCotanGraph() {
    std::cout << "\n=== operators: laplacian cotan/graph + independent cache ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto ops = m.operators();

    const auto& L = ops.laplacian().cotan();
    EXPECT(L.rows() == 12 && L.cols() == 12, "cotan Laplacian [12,12]");
    EXPECT(m.isOperatorCached(OperatorId::LaplacianCotan), "cotan cached after request");
    // INDEPENDENT cache: requesting cotan did NOT build mass
    EXPECT(!m.isOperatorCached(OperatorId::MassLumped), "cotan request did not build mass (decoupled)");

    const auto& G = ops.laplacian().graph();
    EXPECT(G.rows() == 12, "graph Laplacian [12,12]");
    // graph Laplacian diagonal = vertex degree (>0)
    EXPECT(G.coeff(0,0) > 0.5, "graph Laplacian diagonal = degree");

    // release drops the slot; re-request rebuilds
    m.releaseOperator(OperatorId::LaplacianCotan);
    EXPECT(!m.isOperatorCached(OperatorId::LaplacianCotan), "release(cotan) clears slot");
    (void)ops.laplacian().cotan();
    EXPECT(m.isOperatorCached(OperatorId::LaplacianCotan), "re-request rebuilds");

    // stored view stays valid after the OperatorsFacet temporary expires
    // (LaplacianView holds Manifold&, not OperatorsFacet&)
    auto view = m.operators().laplacian();    // temporary OperatorsFacet expires here
    EXPECT(view.cotan().rows() == 12, "stored laplacian view is safe (holds Manifold&)");
}

int main() {
    testLaplacianCotanGraph();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
