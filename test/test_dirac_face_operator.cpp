#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <Eigen/Eigenvalues>
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

static void testExtrinsicBlockFace() {
    std::cout << "\n=== diracFace: extrinsic block E~ ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int Fn = m.nF();   // 20

    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlockFace(m);
    EXPECT(E.rows() == 4*Fn && E.cols() == 4*Fn, "E~ is [4F, 4F] = [80, 80]");

    Eigen::SparseMatrix<double> asym = E - Eigen::SparseMatrix<double>(E.transpose());
    EXPECT(asym.norm() < 1e-10, "E~ is symmetric");

    Eigen::MatrixXd dense(E);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "E~ is positive-semidefinite");
    EXPECT(E.norm() > 1e-6, "E~ is nonzero on a curved mesh");

    // Per-face constant is in the kernel (telescoping around each vertex star).
    Eigen::VectorXd x(4*Fn);
    for (int f = 0; f < Fn; ++f) { x(4*f+0)=0.3; x(4*f+1)=-0.7; x(4*f+2)=0.2; x(4*f+3)=0.5; }
    EXPECT((E * x).cwiseAbs().maxCoeff() < 1e-10, "E~*(face-constant) = 0 (telescoping kernel)");

    // GENUINE quaternionic coupling: at least one 4×4 face-face block must be a
    // NON-scalar matrix (off-diagonal within the block). This is the headline
    // property the vertex-star aggregation buys — a degenerate edge/scalar-Laplacian
    // assembly would give every block ∝ I₄ and still pass all checks above.
    Eigen::MatrixXd densE(E);
    bool coupled = false;
    for (int fi = 0; fi < Fn && !coupled; ++fi)
        for (int fj = 0; fj < Fn && !coupled; ++fj) {
            if (fi == fj) continue;
            Eigen::Matrix4d blk = densE.block(4*fi, 4*fj, 4, 4);
            if (blk.norm() < 1e-12) continue;
            coupled = (blk - (blk.trace()/4.0) * Eigen::Matrix4d::Identity()).norm() > 1e-10;
        }
    EXPECT(coupled, "E~ has genuine quaternionic coupling (non-scalar 4x4 blocks)");
}

static void flatPatch(std::vector<double>& V, std::vector<int32_t>& F) {
    V = { 0,0,0,  1,0,0,  0.5,1,0,  -0.5,1,0,  -1,0,0,  -0.5,-1,0,  0.5,-1,0 };
    F = { 0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1 };
}

static void testFlatKernelFace() {
    std::cout << "\n=== diracFace: flat-region kernel ===\n";
    std::vector<double> V; std::vector<int32_t> F; flatPatch(V, F);
    Manifold m(V.data(), 7, F.data(), 6);
    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlockFace(m);
    EXPECT(E.norm() < 1e-10, "E~ vanishes on a flat patch (pure kernel)");
}

int main() {
    testExtrinsicBlockFace();
    testFlatKernelFace();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
