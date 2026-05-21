/**
 * test_cholesky_cache.cpp — Verify the CholeskyCache contract.
 *
 *   1. Cache hit returns the same factor reference (no rebuild).
 *   2. Cached solve is bit-for-bit identical to a freshly-built
 *      reference factor — this is the contract that lets us refactor
 *      Poisson / Hodge / direction-field with confidence.
 *   3. Repeat Hodge with the same ω gives identical output.
 *   4. Different ω gives different output (no stale state in cache).
 */

#include "nxr/compute.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

using namespace nxr::manifold;
using namespace nxr::manifold::solve;
using namespace nxr::manifold::ops;
using namespace nxr::manifold::ops::laplacian::connection;
using namespace nxr::manifold::transport;
using namespace nxr::manifold::connection;
using namespace nxr::manifold::parametrization;
using namespace nxr::manifold::parametrization::stripes;
using namespace nxr::manifold::geometry;
using namespace nxr::manifold::query;
using namespace nxr::field::generate;
using namespace nxr::field::interp;
using namespace nxr::field::op;
using namespace nxr::field::extract;

// Same icosahedron mesh used by test_eigen.cpp — small enough to keep
// the test fast, large enough that the matrices are non-trivial.
static void buildIcosahedron(std::vector<double>& V, std::vector<int32_t>& F) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        V.push_back(raw[i] / len);
        V.push_back(raw[i+1] / len);
        V.push_back(raw[i+2] / len);
    }
    F = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
}

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << #cond << ")" << std::endl; \
        std::exit(1); \
    } \
} while (0)

int main() {
    std::cout << "[test_cholesky_cache] starting" << std::endl;

    std::vector<double> V;
    std::vector<int32_t> F;
    buildIcosahedron(V, F);
    int nV = static_cast<int>(V.size() / 3);
    int nF = static_cast<int>(F.size() / 3);

    Manifold m(V.data(), nV, F.data(), nF);
    auto ops = assembleManifoldOperators(m);
    auto dec = assembleDECOperators(m);

    // ── 1. Cache hit returns same reference ──────────────────
    {
        CholeskyCache cache;
        const auto& a = cache.laplacian(ops);
        const auto& b = cache.laplacian(ops);
        REQUIRE(&a == &b, "laplacian: cache hit must return same factor");

        const auto& c = cache.hodgeExact(dec);
        const auto& d = cache.hodgeExact(dec);
        REQUIRE(&c == &d, "hodgeExact: cache hit must return same factor");

        const auto& e = cache.hodgeCoExact(dec);
        const auto& f_ = cache.hodgeCoExact(dec);
        REQUIRE(&e == &f_, "hodgeCoExact: cache hit must return same factor");
        std::cout << "  [1] cache hit returns same reference ✓" << std::endl;
    }

    // ── 2. Cached solve == reference solve (bit-for-bit) ─────
    // Build the reference factor inline, exactly as the pre-cache
    // implementation did. The cached factor must produce identical
    // solves on identical RHS.
    {
        CholeskyCache cache;

        // Reference for laplacian: SimplicialLLT(L + 1e-8 I)
        Eigen::SparseMatrix<double> Lref = ops.cotanLaplacian;
        for (int i = 0; i < nV; i++) Lref.coeffRef(i, i) += 1e-8;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> refL;
        refL.compute(Lref);
        REQUIRE(refL.info() == Eigen::Success, "reference L factor failed");

        Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(nV, -1.0, 1.0);
        Eigen::VectorXd refSol = refL.solve(rhs);
        Eigen::VectorXd cacheSol = cache.laplacian(ops).solve(rhs);
        double dL = (refSol - cacheSol).cwiseAbs().maxCoeff();
        REQUIRE(dL == 0.0, "laplacian cached solve must be bit-identical to reference");
        std::cout << "  [2a] laplacian: cached solve bit-identical (max|diff|=0) ✓" << std::endl;

        // Reference for hodgeExact: SimplicialLLT(d0ᵀ ★₁ d0 + 1e-8 I)
        Eigen::SparseMatrix<double> Aref =
            dec.d0.transpose() * dec.hodge1 * dec.d0;
        for (int i = 0; i < nV; i++) Aref.coeffRef(i, i) += 1e-8;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> refA;
        refA.compute(Aref);
        REQUIRE(refA.info() == Eigen::Success, "reference A factor failed");

        Eigen::VectorXd refSolA = refA.solve(rhs);
        Eigen::VectorXd cacheSolA = cache.hodgeExact(dec).solve(rhs);
        double dA = (refSolA - cacheSolA).cwiseAbs().maxCoeff();
        REQUIRE(dA == 0.0, "hodgeExact cached solve must be bit-identical to reference");
        std::cout << "  [2b] hodgeExact: cached solve bit-identical (max|diff|=0) ✓" << std::endl;

        // Reference for hodgeCoExact: SparseLU(d1 ★₁⁻¹ d1ᵀ + 1e-8 I)
        Eigen::SparseMatrix<double> Bref =
            dec.d1 * dec.hodge1Inverse * dec.d1.transpose();
        for (int i = 0; i < nF; i++) Bref.coeffRef(i, i) += 1e-8;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> refB;
        refB.compute(Bref);
        REQUIRE(refB.info() == Eigen::Success, "reference B factor failed");

        Eigen::VectorXd rhsF = Eigen::VectorXd::LinSpaced(nF, -1.0, 1.0);
        Eigen::VectorXd refSolB = refB.solve(rhsF);
        Eigen::VectorXd cacheSolB = cache.hodgeCoExact(dec).solve(rhsF);
        double dB = (refSolB - cacheSolB).cwiseAbs().maxCoeff();
        REQUIRE(dB == 0.0, "hodgeCoExact cached solve must be bit-identical to reference");
        std::cout << "  [2c] hodgeCoExact: cached solve bit-identical (max|diff|=0) ✓" << std::endl;
    }

    // ── 3. Repeat Hodge with same ω — bit-identical output ───
    {
        CholeskyCache cache;
        auto omega = randomOmega(m.nE(), 7);
        auto h1 = hodge(m, dec, cache, omega);
        auto h2 = hodge(m, dec, cache, omega);
        double dAlpha = (h1.dAlpha - h2.dAlpha).cwiseAbs().maxCoeff();
        double dBeta  = (h1.deltaBeta - h2.deltaBeta).cwiseAbs().maxCoeff();
        double dGamma = (h1.gamma - h2.gamma).cwiseAbs().maxCoeff();
        REQUIRE(dAlpha == 0.0, "repeat Hodge: dα must match");
        REQUIRE(dBeta  == 0.0, "repeat Hodge: δβ must match");
        REQUIRE(dGamma == 0.0, "repeat Hodge: γ must match");
        std::cout << "  [3] repeat Hodge with same ω: bit-identical ✓" << std::endl;
    }

    // ── 4. Different ω — different output ────────────────────
    // Sanity check that the cache isn't returning a stale solve.
    {
        CholeskyCache cache;
        auto omegaA = randomOmega(m.nE(), 7);
        auto omegaB = randomOmega(m.nE(), 8);
        auto hA = hodge(m, dec, cache, omegaA);
        auto hB = hodge(m, dec, cache, omegaB);
        double diff = (hA.dAlpha - hB.dAlpha).norm();
        REQUIRE(diff > 1e-6, "different ω must yield different dα");
        std::cout << "  [4] different ω yields different output (||Δdα||="
                  << diff << ") ✓" << std::endl;
    }

    std::cout << "[test_cholesky_cache] all assertions passed ✓" << std::endl;
    return 0;
}
