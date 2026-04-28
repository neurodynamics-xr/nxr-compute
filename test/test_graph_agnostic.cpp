/**
 * test_graph_agnostic.cpp — verify cxf's spectral + Poisson + heat
 * pipeline operates correctly on a non-mesh input.
 *
 * Builds a synthetic graph Laplacian (50-node path graph) and feeds
 * it through:
 *   • solveEigenmodes(L, M, k)
 *   • normalizeEigenmodes
 *   • removeDC
 *   • CholeskyCache::laplacian(L)        — agnostic overload
 *   • solvePoisson(L, M, cache, density) — agnostic overload
 *   • generateHeatDiffusion(M, eig, …)   — agnostic overload
 *
 * Confirms that the convenience layer no longer requires a
 * MeshOperators struct and serves graph signal processing users
 * the same way it serves mesh users.
 */

#include "nxr/compute.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace nxr::compute;

static int testsRun = 0;
static int testsFailed = 0;
#define EXPECT(cond) do {                                       \
    testsRun++;                                                 \
    if (!(cond)) {                                              \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__    \
                  << "]: " << #cond << std::endl;               \
        testsFailed++;                                          \
    }                                                           \
} while (0)

namespace {

// Build the combinatorial Laplacian L = D - W of a path graph
// 0 — 1 — 2 — … — (n-1). Eigenvalues are known analytically:
//   λ_k = 2 (1 - cos(π k / n)),   k = 0, …, n-1
// with λ_0 = 0 (DC) and the spectrum strictly increasing.
struct PathGraph {
    Eigen::SparseMatrix<double> L;  // [n, n]
    Eigen::SparseMatrix<double> M;  // [n, n] identity (uniform node weights)
    int n;
};

PathGraph makePathGraph(int n) {
    PathGraph g;
    g.n = n;

    std::vector<Eigen::Triplet<double>> Ltrips;
    Ltrips.reserve(3 * n);
    for (int i = 0; i < n; ++i) {
        int deg = (i == 0 || i == n - 1) ? 1 : 2;
        Ltrips.emplace_back(i, i, static_cast<double>(deg));
        if (i + 1 < n) {
            Ltrips.emplace_back(i, i + 1, -1.0);
            Ltrips.emplace_back(i + 1, i, -1.0);
        }
    }
    g.L.resize(n, n);
    g.L.setFromTriplets(Ltrips.begin(), Ltrips.end());

    std::vector<Eigen::Triplet<double>> Mtrips;
    Mtrips.reserve(n);
    for (int i = 0; i < n; ++i) Mtrips.emplace_back(i, i, 1.0);
    g.M.resize(n, n);
    g.M.setFromTriplets(Mtrips.begin(), Mtrips.end());

    return g;
}

} // namespace

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  cxf graph-agnostic spectral pipeline" << std::endl;
    std::cout << "==========================================" << std::endl;

    const int n = 50;
    auto g = makePathGraph(n);

    // ── 1. solveEigenmodes on a graph Laplacian ────────────────
    std::cout << "[1] solveEigenmodes on path graph L\n";
    int k = 6;
    auto eigRaw = solveEigenmodes(g.L, g.M, k);
    EXPECT(eigRaw.k == k);
    EXPECT(eigRaw.eigenvectors.rows() == n);
    EXPECT(eigRaw.eigenvalues.size() == k);

    // λ_0 of a connected graph is 0 (numerically tiny).
    EXPECT(std::abs(eigRaw.eigenvalues(0)) < 1e-8);

    // λ_1 = 2 (1 - cos(π / n)) for the path graph.
    double lambda1Expected = 2.0 * (1.0 - std::cos(M_PI / n));
    double lambda1Actual = eigRaw.eigenvalues(1);
    EXPECT(std::abs(lambda1Actual - lambda1Expected) < 1e-6);
    std::cout << "    λ_1 expected " << lambda1Expected
              << ", got " << lambda1Actual << "\n";

    // ── 2. normalizeEigenmodes (M = I) is a no-op up to sign ───
    std::cout << "[2] normalizeEigenmodes leaves U M-orthonormal\n";
    auto Un = normalizeEigenmodes(eigRaw.eigenvectors, g.M);
    Eigen::MatrixXd Gram = Un.transpose() * (g.M * Un);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(k, k);
    double orthErr = (Gram - I).cwiseAbs().maxCoeff();
    EXPECT(orthErr < 1e-10);
    std::cout << "    ‖UᵀMU - I‖_∞ = " << orthErr << "\n";

    // ── 3. removeDC drops the zero eigenvalue ──────────────────
    std::cout << "[3] removeDC trims the connected-component DC mode\n";
    eigRaw.eigenvectors = Un;
    auto eig = removeDC(eigRaw);
    EXPECT(eig.k == k - 1);
    EXPECT(std::abs(eig.eigenvalues(0) - lambda1Expected) < 1e-6);

    // ── 4. CholeskyCache::laplacian on raw sparse ──────────────
    std::cout << "[4] CholeskyCache::laplacian(L) caches and reuses\n";
    CholeskyCache cache;
    const auto& f1 = cache.laplacian(g.L);
    const auto& f2 = cache.laplacian(g.L);
    EXPECT(&f1 == &f2);  // same factor returned

    // ── 5. solvePoisson(L, M, cache, …) ────────────────────────
    std::cout << "[5] solvePoisson on graph Laplacian\n";
    std::map<int, double> density = { {0, 1.0}, {n - 1, -1.0} };
    auto phi = solvePoisson(g.L, g.M, cache, density);
    EXPECT(phi.size() == n);

    // M-weighted mean of phi should be ≈ 0. The Poisson formulation
    // subtracts the weighted mean of ρ from the rhs, but the εI
    // regularization shifts the solution by O(ε / λ_1) from exact
    // zero-mean. Tolerance accordingly.
    double phiMean = (g.M * phi).sum() / static_cast<double>(n);
    EXPECT(std::abs(phiMean) < 1e-6);
    std::cout << "    range = [" << phi.minCoeff()
              << ", " << phi.maxCoeff() << "], mean = " << phiMean << "\n";

    // ── 6. generateHeatDiffusion(M, eig, u0, …) ────────────────
    std::cout << "[6] generateHeatDiffusion on graph eigenmodes\n";
    // u0 = a single eigenmode → exact pure exponential decay.
    Eigen::VectorXd u0 = eig.eigenvectors.col(0);
    std::vector<double> ts = { 0.0, 0.1, 0.5, 2.0 };
    double alpha = 1.0;
    auto heat = generateHeatDiffusion(g.M, eig, u0, ts, alpha);
    EXPECT(heat.rows() == static_cast<int>(ts.size()));
    EXPECT(heat.cols() == n);

    double lambda0 = eig.eigenvalues(0);
    for (size_t ti = 0; ti < ts.size(); ti++) {
        double expectedAtten = std::exp(-alpha * ts[ti] * lambda0);
        Eigen::VectorXd expected = expectedAtten * u0;
        Eigen::VectorXd actual   = heat.row(ti).cast<double>();
        double err = (actual - expected).cwiseAbs().maxCoeff();
        EXPECT(err < 1e-5);
    }
    std::cout << "    single-mode IC decays as exp(-αtλ_k) ✓\n";

    // Sum-to-uniform asymptotic: starting from a delta, very large t
    // should converge to a uniform constant equal to the mean. The
    // path graph has λ_1 ≈ π²/n² so the slowest mode decays as
    // exp(-α·t·π²/n²); with n=50 we need t ≳ 5000 for the slowest
    // mode to fall below 1e-5.
    Eigen::VectorXd u0Delta = Eigen::VectorXd::Zero(n);
    u0Delta(n / 2) = 1.0;
    std::vector<double> tsLong = { 5000.0 };
    auto heatLong = generateHeatDiffusion(g.M, eig, u0Delta, tsLong, 1.0);
    Eigen::VectorXd uInf = heatLong.row(0).cast<double>();
    double expectedMean = 1.0 / static_cast<double>(n);
    double maxDev = (uInf.array() - expectedMean).abs().maxCoeff();
    EXPECT(maxDev < 1e-5);
    std::cout << "    delta IC at t=5000 → uniform mean " << expectedMean
              << " (max dev " << maxDev << ") ✓\n";

    std::cout << "\n==========================================\n";
    std::cout << "  " << (testsRun - testsFailed) << "/" << testsRun
              << " tests passed";
    if (testsFailed > 0) std::cout << " — " << testsFailed << " FAILURES";
    std::cout << "\n==========================================\n";
    return testsFailed == 0 ? 0 : 1;
}
