// Named-operator eigensolve: eigenProblemFor / eigenOperator + ℍ-multiplet
// reconstruction + dense verification path.
#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
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

// kron(M, I_b) built independently — the oracle for blockKron.
static Eigen::SparseMatrix<double> kronOracle(const Eigen::SparseMatrix<double>& M, int b) {
    std::vector<Eigen::Triplet<double>> T;
    for (int o = 0; o < M.outerSize(); ++o)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, o); it; ++it)
            for (int c = 0; c < b; ++c)
                T.emplace_back(b*(int)it.row()+c, b*(int)it.col()+c, it.value());
    Eigen::SparseMatrix<double> K(M.rows()*b, M.cols()*b);
    K.setFromTriplets(T.begin(), T.end());
    return K;
}

static void testBlockKronAndProblem() {
    std::cout << "\n=== eigenProblemFor: assembly ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV(), Fn = m.nF();

    // blockKron == kron(M, I_b).
    Eigen::SparseMatrix<double> Mg = m.operators().mass().galerkin();
    EXPECT((solve::blockKron(Mg, 4) - kronOracle(Mg, 4)).norm() < 1e-12,
           "blockKron(M,4) == kron(M, I4)");

    // dirac problem: K = dirac(τ), M = M_galerkin ⊗ I4, block 4.
    solve::EigenProblem p = solve::eigenProblemFor(m, {solve::EigenOperator::Dirac, 0.5});
    EXPECT(p.blockSize == 4 && p.K.rows() == 4*N && p.M.rows() == 4*N, "dirac problem is [4V×4V], block 4");
    EXPECT((p.K - m.operators().dirac(0.5)).norm() < 1e-12, "dirac K == operators().dirac(τ)");
    EXPECT((p.M - kronOracle(Mg, 4)).norm() < 1e-12, "dirac M == M_galerkin ⊗ I4");

    solve::EigenProblem pf = solve::eigenProblemFor(m, {solve::EigenOperator::DiracFace, 1.0});
    EXPECT(pf.blockSize == 4 && pf.K.rows() == 4*Fn, "diracFace problem is [4F×4F], block 4");

    solve::EigenProblem pc = solve::eigenProblemFor(m, {solve::EigenOperator::LaplacianCotan});
    EXPECT(pc.blockSize == 1 && pc.K.rows() == N, "cotan problem is [V×V], block 1");
}

static void testCotanMatchesDirect() {
    std::cout << "\n=== eigenOperator(cotan) == eigen(cotanL, mass) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    solve::EigenResult a = solve::eigenOperator(m, {solve::EigenOperator::LaplacianCotan}, 6);
    solve::EigenResult b = solve::eigen(m.operators().laplacian().cotan(),
                                        m.operators().mass().galerkin(), 6, -1e-8, true);
    EXPECT((a.eigenvalues - b.eigenvalues).cwiseAbs().maxCoeff() < 1e-9,
           "named-operator cotan eigenvalues match the direct eigensolve");
}

// Largest deviation of any 4-consecutive eigenvalue group from constant.
static double maxMultipletSpread(const Eigen::VectorXd& ev) {
    double worst = 0;
    for (int g = 0; 4*g + 3 < ev.size(); ++g) {
        double lo = ev.segment(4*g, 4).minCoeff(), hi = ev.segment(4*g, 4).maxCoeff();
        worst = std::max(worst, (hi - lo) / (1.0 + std::abs(hi)));
    }
    return worst;
}

static void testDiracMultipletsAndDense() {
    std::cout << "\n=== eigenOperator(dirac): ℍ-reconstruction vs dense ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    solve::EigenProblem p = solve::eigenProblemFor(m, {solve::EigenOperator::Dirac, 0.5});

    // Reconstructed multiplets: M-orthonormal, exactly 4-fold, ascending.
    solve::EigenResult rec = solve::eigenOperator(
        m, {solve::EigenOperator::Dirac, 0.5}, 8,
        -1e-8, /*normalize=*/true, /*reconstructMultiplets=*/true);
    EXPECT(rec.k % 4 == 0 && rec.k >= 8, "reconstructed count is a multiple of 4 (>=8)");
    Eigen::MatrixXd gram = rec.eigenvectors.transpose() * (p.M * rec.eigenvectors);
    EXPECT((gram - Eigen::MatrixXd::Identity(gram.rows(), gram.cols())).cwiseAbs().maxCoeff() < 1e-8,
           "ΦᵀMΦ ≈ I (M-orthonormal reconstructed basis)");
    EXPECT(maxMultipletSpread(rec.eigenvalues) < 1e-9, "reconstructed eigenvalues are exact 4-fold multiplets");

    // Dense exact solve agrees with the reconstructed spectrum on the lowest 8.
    solve::EigenResult dns = solve::eigenOperator(
        m, {solve::EigenOperator::Dirac, 0.5}, 8,
        -1e-8, /*normalize=*/true, /*reconstructMultiplets=*/false, /*dense=*/true);
    EXPECT(dns.eigenvalues.size() == 8, "dense returns the lowest 8");
    double maxDiff = (rec.eigenvalues.head(8) - dns.eigenvalues.head(8)).cwiseAbs().maxCoeff();
    EXPECT(maxDiff < 1e-6, "reconstructed eigenvalues match the dense generalized solve");

    // The constant mode is a 4-fold zero.
    EXPECT(std::abs(dns.eigenvalues(0)) < 1e-8 && std::abs(dns.eigenvalues(3)) < 1e-8,
           "dirac(τ) has a 4-fold zero (constant mode)");
}

static void testDiracFaceEigensolve() {
    std::cout << "\n=== eigenOperator(diracFace) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    solve::EigenResult rec = solve::eigenOperator(
        m, {solve::EigenOperator::DiracFace, 0.5}, 8,
        -1e-8, true, /*reconstructMultiplets=*/true);
    EXPECT(rec.k % 4 == 0, "diracFace reconstructed count divisible by 4");
    EXPECT(maxMultipletSpread(rec.eigenvalues) < 1e-9, "diracFace exact 4-fold multiplets");

    solve::EigenResult dns = solve::eigenOperator(
        m, {solve::EigenOperator::DiracFace, 0.5}, 8, -1e-8, true, false, /*dense=*/true);
    EXPECT((rec.eigenvalues.head(8) - dns.eigenvalues.head(8)).cwiseAbs().maxCoeff() < 1e-6,
           "diracFace reconstructed == dense (lowest 8)");
}

int main() {
    testBlockKronAndProblem();
    testCotanMatchesDirect();
    testDiracMultipletsAndDense();
    testDiracFaceEigensolve();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
