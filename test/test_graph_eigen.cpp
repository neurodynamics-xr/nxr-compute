// The topological (combinatorial) graph Laplacian L = D - A on the mesh graph, unweighted: identity or degree mass.
#include "nxr/compute.h"
#include "mesh_helpers.h"
#include <set>
using namespace nxr::manifold;

int main() {
    std::vector<double> V; std::vector<int32_t> F; icosphere(0, V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    std::cout << "\n=== graph Laplacian eigensolve on the icosahedron (5-regular, 12 vertices) ===\n";
    // oracle: L = D - A from the face list
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(12, 12); std::set<std::pair<int,int>> E;
    for (int f = 0; f < 20; ++f) for (int k = 0; k < 3; ++k) { int a = F[3*f+k], b = F[3*f+(k+1)%3]; E.insert({std::min(a,b), std::max(a,b)}); }
    for (auto [a, b] : E) { L(a,a) += 1; L(b,b) += 1; L(a,b) -= 1; L(b,a) -= 1; }
    EXPECT((Eigen::MatrixXd(ops::graphLaplacian(m)) - L).norm() < 1e-12, "graphLaplacian == D - A");

    solve::EigenOperatorSpec idm{solve::EigenOperator::LaplacianGraph};
    auto r = solve::eigenOperator(m, idm, 11, solve::kShiftLadder, true, false, /*dense=*/true);
    // spectrum 0, 5-sqrt5 (x3), 6 (x5), 5+sqrt5 (x3)
    const double s5 = std::sqrt(5.0);
    double ex[12] = {0, 5-s5, 5-s5, 5-s5, 6, 6, 6, 6, 6, 5+s5, 5+s5, 5+s5};
    double d = 0; for (int i = 0; i < 11; ++i) d = std::max(d, std::abs(r.eigenvalues(i) - ex[i]));
    EXPECT(d < 1e-10, "identity mass: dense spectrum 0, 5-sqrt5 x3, 6 x5, 5+sqrt5 x3; max err " + std::to_string(d));
    auto rs = solve::eigenOperator(m, idm, 6, solve::kShiftLadder, true, false, false);
    double ds = 0; for (int i = 0; i < 6; ++i) ds = std::max(ds, std::abs(rs.eigenvalues(i) - ex[i]));
    EXPECT(ds < 1e-8, "identity mass: sparse (shift ladder) lowest 6 match, max err " + std::to_string(ds));

    solve::EigenOperatorSpec dg{solve::EigenOperator::LaplacianGraph}; dg.graphDegreeMass = true;
    auto p = solve::eigenProblemFor(m, dg);
    EXPECT((Eigen::MatrixXd(p.M) - 5.0 * Eigen::MatrixXd::Identity(12, 12)).norm() < 1e-12, "degree mass == diag(degree) (=5 here)");
    auto rd = solve::eigenOperator(m, dg, 6, solve::kShiftLadder, true, false, false);
    double dd = 0; for (int i = 0; i < 6; ++i) dd = std::max(dd, std::abs(rd.eigenvalues(i) - ex[i] / 5.0));
    EXPECT(dd < 1e-8, "degree mass (regular graph): spectrum / 5, max err " + std::to_string(dd));

    // irregular graph: mass matters. level-1 icosphere has 12 degree-5 and 30 degree-6 vertices
    icosphere(1, V, F);
    Manifold m1(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    auto a = solve::eigenOperator(m1, idm, 20, solve::kShiftLadder, true, false, true);
    auto b = solve::eigenOperator(m1, dg, 20, solve::kShiftLadder, true, false, true);
    EXPECT(a.eigenvalues(0) < 1e-10 && b.eigenvalues(0) < 1e-10, "constant mode is the zero eigenvalue under both masses");
    EXPECT(std::abs(a.eigenvalues(1) - b.eigenvalues(1)) > 1e-3, "irregular graph: degree mass changes the spectrum");
    auto pd = solve::eigenProblemFor(m1, dg);
    double res = (pd.K * b.eigenvectors - pd.M * b.eigenvectors * b.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(res < 1e-10, "L phi = lambda D phi, residual " + std::to_string(res));
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
