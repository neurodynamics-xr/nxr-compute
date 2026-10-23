// Spin Diracs (Hoffmann–Ye extrinsic, intrinsic HY, intrinsic Ye) on face spinors; the vertex-spinor quotient eigensolve.
#include "nxr/compute.h"
#include "mesh_helpers.h"
#include <algorithm>
using namespace nxr::manifold;
using ops::spin::Kind;

int main() {
    std::vector<double> V; std::vector<int32_t> F; icosphere(3, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    const int nF = m.nF();
    std::cout << "\n=== spin::assemble on the unit icosphere (level 3, " << nF << " faces) ===\n";
    auto ex = ops::spin::assemble(m, Kind::Extrinsic), hy = ops::spin::assemble(m, Kind::IntrinsicHY), ye = ops::spin::assemble(m, Kind::IntrinsicYe);
    EXPECT(ex.D.rows() == 4 * nF, "D is 4F x 4F");
    for (auto* s : {&ex, &hy, &ye}) EXPECT((s->D - Eigen::SparseMatrix<double>(s->D.transpose())).norm() < 1e-12, "D symmetric");
    Eigen::SparseMatrix<double> Hd(4 * nF, 4 * nF);
    Eigen::VectorXd h4(4 * nF); for (int f = 0; f < nF; ++f) h4.segment(4 * f, 4).setConstant(ex.H(f));
    Hd = h4.asDiagonal();
    EXPECT((hy.D - ex.D - Hd).norm() < 1e-12, "intrinsicHY = extrinsic + H");
    Eigen::VectorXd one = Eigen::VectorXd::Zero(4 * nF); for (int f = 0; f < nF; ++f) one(4 * f) = 1;
    EXPECT((ex.D * one).norm() < 1e-10, "extrinsic Dirac kills the constant spinors");
    EXPECT(std::abs(ex.H.sum() - 4 * M_PI) / (4 * M_PI) < 0.05, "total integrated mean curvature ~ 4 pi (unit sphere: H = 1, area 4 pi)");

    std::cout << "\n=== eigenSpin: quotient on vertex spinors; the round sphere has intrinsic +-1 (x4 each), extrinsic 1 and -2 ===\n";
    auto r = solve::eigenSpin(m, Kind::IntrinsicHY, 12);
    EXPECT(r.k == 12 && std::is_sorted(r.eigenvalues.data(), r.eigenvalues.data() + 12), "12 modes, mu ascending");
    std::cout << "  intrinsicHY mu:"; for (int i = 0; i < 12; ++i) std::cout << " " << r.eigenvalues(i); std::cout << "\n";
    int near1 = 0, pos = 0; for (int i = 0; i < 12; ++i) if (std::abs(std::abs(r.eigenvalues(i)) - 1.0) < 0.12) { ++near1; pos += r.eigenvalues(i) > 0; }
    EXPECT(near1 == 8 && pos == 4, "intrinsicHY: 8 modes |mu| ~ 1 (4 of each sign), got " + std::to_string(near1));
    auto rye = solve::eigenSpin(m, Kind::IntrinsicYe, 12);
    int near1ye = 0; for (int i = 0; i < 12; ++i) near1ye += std::abs(std::abs(rye.eigenvalues(i)) - 1.0) < 0.12;
    EXPECT(near1ye == 8, "intrinsicYe: 8 modes |mu| ~ 1, got " + std::to_string(near1ye));
    auto rx = solve::eigenSpin(m, Kind::Extrinsic, 24);
    std::cout << "  extrinsic mu:"; for (int i = 0; i < rx.k; ++i) std::cout << " " << rx.eigenvalues(i); std::cout << "\n";
    int e1 = 0, em2 = 0; for (int i = 0; i < rx.k; ++i) { e1 += std::abs(rx.eigenvalues(i) - 1.0) < 0.12; em2 += std::abs(rx.eigenvalues(i) + 2.0) < 0.2; }
    EXPECT(e1 == 8 && em2 == 4, "extrinsic (k=24): mu ~ 1 (x8) and mu ~ -2 (x4), got " + std::to_string(e1) + " / " + std::to_string(em2));
    // right-quaternion multiplication commutes: multiplicities in fours (HY lowest |mu| quartet)
    double spread = 0; for (int i = 0; i < 12; ++i) if (std::abs(r.eigenvalues(i) - 1.0) < 0.12) spread = std::max(spread, std::abs(r.eigenvalues(i) - 1.0));
    double q0 = 1e9, q1 = -1e9; for (int i = 0; i < 12; ++i) if (std::abs(r.eigenvalues(i) - 1.0) < 0.12) { q0 = std::min(q0, r.eigenvalues(i)); q1 = std::max(q1, r.eigenvalues(i)); }
    EXPECT(q1 - q0 < 1e-6, "the +1 quartet is degenerate to " + std::to_string(q1 - q0));

    // eigenOperator routes the spin kinds
    solve::EigenOperatorSpec spec{solve::EigenOperator::SpinIntrinsicHY};
    auto viaOp = solve::eigenOperator(m, spec, 12);
    EXPECT((viaOp.eigenvalues - r.eigenvalues).cwiseAbs().maxCoeff() < 1e-9, "eigenOperator(SpinIntrinsicHY) == eigenSpin");
    bool threw = false; try { solve::eigenProblemFor(m, spec); } catch (const nxr::core::Error&) { threw = true; }
    EXPECT(threw, "eigenProblemFor refuses the spin quotient (not a (K,M) pencil)");
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
