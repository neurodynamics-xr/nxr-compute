// Hodge–Dirac d+δ on the intrinsic Delaunay mesh: structure, ±μ symmetry, the exact/coexact split against the dense
// pencil, spectrum slicing, the explicit-shift indefinite solve, the ⋆1 floor.
#include "nxr/compute.h"
#include "mesh_helpers.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
using namespace nxr::manifold;
using ops::hodgedirac::HodgeDirac;

static Eigen::MatrixXd dense(const Eigen::SparseMatrix<double>& S) { return Eigen::MatrixXd(S); }

static void testStructure() {
    std::cout << "\n=== hodgedirac::assemble: structure on the unit icosphere (level 2) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(2, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    HodgeDirac h = ops::hodgedirac::assemble(m);
    EXPECT(h.nV == 162 && h.nE == 480 && h.nF == 320 && h.K.rows() == 962, "V,E,F = 162,480,320; N = 962");
    EXPECT((h.K - Eigen::SparseMatrix<double>(h.K.transpose())).norm() < 1e-14, "K symmetric");
    EXPECT(h.M.minCoeff() > 0, "M > 0");
    EXPECT((h.d1 * h.d0).norm() < 1e-14, "d1 d0 = 0");
    EXPECT(std::abs(h.star0.sum() - h.star2.cwiseInverse().sum()) < 1e-10, "sum star0 = total area = sum 1/star2");
    EXPECT(std::abs(h.star0.sum() - 4 * M_PI) / (4 * M_PI) < 0.02, "total area ~ 4 pi (inscribed polyhedron)");
    EXPECT((h.d0.transpose() * h.star1.asDiagonal() * h.d0 * Eigen::VectorXd::Ones(h.nV)).norm() < 1e-12, "cotan Laplacian kills constants");
    // K blocks: ⋆1 d0 under the 1-form/0-form block, ⋆2 d1 under 2/1
    Eigen::SparseMatrix<double> A = h.star1.asDiagonal() * h.d0;
    EXPECT((h.K.block(h.nV, 0, h.nE, h.nV) - A).norm() < 1e-14, "K[1-forms, 0-forms] = star1 d0");
    ops::hodgedirac::HodgeDirac hc = ops::hodgedirac::assemble(m, 10.0);   // floor above every weight
    EXPECT(hc.star1Clamped == hc.nE && hc.star1.minCoeff() >= 10.0, "star1 floor clamps (all edges at floor 10)");
}

static void testSpectrumAndSplit() {
    std::cout << "\n=== dense pencil: +-mu symmetry, zero modes, D^2 = Hodge Laplacians, the split ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(2, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    HodgeDirac h = ops::hodgedirac::assemble(m);
    Eigen::MatrixXd Kd = dense(h.K), Md = dense(ops::hodgedirac::massMatrix(h));
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Kd, Md);
    Eigen::VectorXd mu = es.eigenvalues();
    int zeros = 0; std::vector<double> pos, neg;
    for (int i = 0; i < mu.size(); ++i) { if (std::abs(mu(i)) < 1e-8) ++zeros; else if (mu(i) > 0) pos.push_back(mu(i)); else neg.push_back(-mu(i)); }
    std::sort(pos.begin(), pos.end()); std::sort(neg.begin(), neg.end());
    EXPECT(zeros == 2, "2 zero modes on a sphere (constant 0-form, constant 2-form), got " + std::to_string(zeros));
    double sym = 0; if (pos.size() == neg.size()) for (size_t i = 0; i < pos.size(); ++i) sym = std::max(sym, std::abs(pos[i] - neg[i]) / pos[i]);
    EXPECT(pos.size() == neg.size() && sym < 1e-9, "spectrum symmetric +-mu to " + std::to_string(sym));
    EXPECT(std::abs(pos[0] * pos[0] - 2.0) < 0.1, "lowest mu^2 ~ 2 (l(l+1), l=1), got " + std::to_string(pos[0] * pos[0]));

    const int k = 40;
    solve::EigenResult r = solve::hodgeDiracModes(h, k);
    EXPECT(r.k == k && r.kind.size() == k, "40 modes with kinds");
    double dmu = 0; for (int i = 0; i < k; ++i) dmu = std::max(dmu, std::abs(r.eigenvalues(i) - pos[i]) / pos[i]);
    EXPECT(dmu < 1e-8, "split mu == dense pencil mu (lowest 40 positive), max rel diff " + std::to_string(dmu));
    Eigen::MatrixXd X = r.eigenvectors;
    double res = (Kd * X - Md * X * r.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(res < 1e-9, "K x = mu M x residual " + std::to_string(res));
    double orth = (X.transpose() * Md * X - Eigen::MatrixXd::Identity(k, k)).cwiseAbs().maxCoeff();
    EXPECT(orth < 1e-9, "modes M-orthonormal " + std::to_string(orth));
    int nEx = (r.kind.array() == 0).count(), nCo = (r.kind.array() == 2).count();
    EXPECT(nEx > 0 && nCo > 0 && nEx + nCo == k, "both kinds present: exact " + std::to_string(nEx) + ", coexact " + std::to_string(nCo));
    solve::EigenResult rs = solve::hodgeDiracModes(h, 10, /*bothSigns=*/true);
    double mres = (Kd * rs.eigenvectors - Md * rs.eigenvectors * rs.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(rs.k == 20 && rs.eigenvalues(10) == -rs.eigenvalues(0) && mres < 1e-9, "bothSigns: -mu mirror (1-form block negated) is an eigenvector, residual " + std::to_string(mres));
    // the kind from the dense eigenvectors agrees with the split's kind
    Eigen::VectorXd dm(k); Eigen::MatrixXd dX(h.K.rows(), k); int c = 0;
    for (int i = 0; i < mu.size() && c < k; ++i) if (mu(i) > 1e-8) { dm(c) = mu(i); dX.col(c) = es.eigenvectors().col(i); ++c; }
    Eigen::VectorXi dk = solve::classifyHodgeModes(h, dX, dm);
    int agree = 0; for (int i = 0; i < k; ++i) agree += (dk(i) == r.kind(i));
    EXPECT(agree >= k - 4, "classifyHodgeModes on the dense eigenvectors matches the split's kinds (" + std::to_string(agree) + "/40; mixed inside degenerate pairs)");
}

static void testSlicingAndShift() {
    std::cout << "\n=== spectrum slicing and the explicit-shift indefinite solve (level 3, 642 vertices) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(3, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    HodgeDirac h = ops::hodgedirac::assemble(m);
    Eigen::SparseMatrix<double> L0 = h.d0.transpose() * h.star1.asDiagonal() * h.d0, M0(h.nV, h.nV);
    M0 = h.star0.asDiagonal();
    auto one = solve::eigenSliced(L0, M0, 150, 400);          // one window
    auto sl  = solve::eigenSliced(L0, M0, 150, 30);           // 5+ windows
    double d = (one.eigenvalues - sl.eigenvalues).cwiseAbs().maxCoeff();
    EXPECT(sl.k == 150 && d < 1e-7, "sliced (30/window) == single window, 150 modes, max diff " + std::to_string(d));
    double res = (L0 * sl.eigenvectors - M0 * sl.eigenvectors * sl.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(res < 1e-7, "sliced residual " + std::to_string(res));
    double orth = (sl.eigenvectors.transpose() * M0 * sl.eigenvectors - Eigen::MatrixXd::Identity(150, 150)).cwiseAbs().maxCoeff();
    EXPECT(orth < 1e-6, "sliced modes M-orthonormal " + std::to_string(orth));
    // sphere: lambda = l(l+1) with multiplicity 2l+1 -> the first 3 nonconstant are ~2
    EXPECT(std::abs(sl.eigenvalues(1) - 2.0) < 0.05 && std::abs(sl.eigenvalues(3) - 2.0) < 0.05 && std::abs(sl.eigenvalues(4) - 6.0) < 0.2, "lambda_2..4 ~ 2, lambda_5 ~ 6");

    // thousands of modes through the split: ask for 300 positive modes
    solve::EigenResult r = solve::hodgeDiracModes(h, 300, false, 60);
    EXPECT(r.k == 300 && std::is_sorted(r.eigenvalues.data(), r.eigenvalues.data() + 300), "300 +mu modes, sorted");
    Eigen::SparseMatrix<double> M = ops::hodgedirac::massMatrix(h);
    double res2 = (h.K * r.eigenvectors - M * r.eigenvectors * r.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(res2 < 1e-6, "300-mode residual " + std::to_string(res2));

    // explicit shift on the indefinite pencil: 8 nearest 3.0, both signs
    solve::EigenOperatorSpec spec{solve::EigenOperator::HodgeDirac};
    solve::EigenResult s = solve::eigenOperator(m, spec, 8, 0.01, true);
    double rs = (h.K * s.eigenvectors - M * s.eigenvectors * s.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(s.k == 8 && s.kind.size() == 8 && rs < 1e-7, "eigenOperator(HodgeDirac, sigma=0.01): 8 pairs, kinds, residual " + std::to_string(rs));
    bool hasZero = false, hasNeg = false;
    for (int i = 0; i < s.k; ++i) { hasZero |= std::abs(s.eigenvalues(i)) < 1e-6; hasNeg |= s.eigenvalues(i) < -1e-6; }
    EXPECT(hasZero && hasNeg, "window around 0 holds the two harmonic zero modes and negative mu");
}

int main() {
    testStructure();
    testSpectrumAndSplit();
    testSlicingAndShift();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
