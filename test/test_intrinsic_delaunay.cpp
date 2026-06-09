#include "nxr/compute.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

using nxr::manifold::Manifold;
namespace ops = nxr::manifold::ops;

static int g_failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { std::cerr << "  [FAIL] " << msg << "\n"; ++g_failures; } \
    else { std::cout << "  [PASS] " << msg << "\n"; } } while (0)

static double maxOffDiag(const Eigen::SparseMatrix<double>& L) {
    double m = -1e300;
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            if (it.row() != it.col()) m = std::max(m, it.value());
    return m;
}
static double minEig(const Eigen::SparseMatrix<double>& L) {
    Eigen::MatrixXd dense = Eigen::MatrixXd(L);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    return es.eigenvalues().minCoeff();
}

static void testRhombus() {
    std::cout << "\n=== intrinsicDelaunay: non-Delaunay rhombus ===\n";
    std::vector<double>  V = {0,0,0,  2,0,0,  1,0.2,0,  1,-0.2,0};
    std::vector<int32_t> F = {0,2,1,  0,1,3};   // long-diagonal (0-1) split

    Manifold mRaw(V.data(), 4, F.data(), 2, /*intrinsicDelaunay=*/false);
    Manifold mN  (V.data(), 4, F.data(), 2, /*intrinsicDelaunay=*/true);
    EXPECT(!mRaw.isIntrinsicDelaunay(), "raw not normalized");
    EXPECT( mN.isIntrinsicDelaunay(),  "normalized flag set");

    auto Lraw = ops::assembleManifoldOperators(mRaw).cotanLaplacian;
    auto Ln   = ops::assembleManifoldOperators(mN).cotanLaplacian;
    EXPECT(Lraw.rows()==4 && Ln.rows()==4, "both cotan are 4x4 (same vertices)");

    // raw has a negative cotan weight (positive off-diagonal); normalized does not.
    EXPECT(maxOffDiag(Lraw) > 1e-9,  "raw cotan has a negative weight (non-Delaunay)");
    EXPECT(maxOffDiag(Ln)   < 1e-9,  "normalized cotan: all weights >= 0 (Delaunay)");
    // normalized is PSD (the certificate)
    EXPECT(minEig(Ln) > -1e-9,       "normalized cotan is PSD (min eig >= 0)");
}

static void testIcosphereNoOp() {
    std::cout << "\n=== intrinsicDelaunay: already-Delaunay icosphere (no-op) ===\n";
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<double> V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
                              0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    for (int i=0;i<12;++i){ double n=std::sqrt(V[3*i]*V[3*i]+V[3*i+1]*V[3*i+1]+V[3*i+2]*V[3*i+2]);
        V[3*i]/=n; V[3*i+1]/=n; V[3*i+2]/=n; }
    std::vector<int32_t> F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4,
        11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11,
        6,2,10, 8,6,7, 9,8,1};
    Manifold mRaw(V.data(), 12, F.data(), 20, false);
    Manifold mN  (V.data(), 12, F.data(), 20, true);
    auto Lraw = ops::assembleManifoldOperators(mRaw).cotanLaplacian;
    auto Ln   = ops::assembleManifoldOperators(mN).cotanLaplacian;
    double diff = (Eigen::MatrixXd(Lraw) - Eigen::MatrixXd(Ln)).cwiseAbs().maxCoeff();
    EXPECT(diff < 1e-9, "icosphere already Delaunay: normalized cotan == raw cotan");
}

int main() {
    testRhombus();
    testIcosphereNoOp();
    if (g_failures) { std::cerr << "\n" << g_failures << " failure(s)\n"; return 1; }
    std::cout << "\nALL PASSED\n"; return 0;
}
