#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"

#include <iostream>
#include <random>

namespace nxr::field::generate {

Eigen::VectorXd randomOmega(int nE, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::VectorXd omega(nE);
    for (int i = 0; i < nE; i++) {
        omega(i) = dist(gen);
    }
    return omega;
}

} // namespace nxr::field::generate

namespace nxr::manifold::solve {

using namespace geometrycentral::surface;
using nxr::manifold::ops::DECOperators;
using nxr::manifold::ops::CholeskyCache;
using nxr::field::interp::whitney;

HodgeResult hodge(
    Manifold& m,
    const DECOperators& dec,
    CholeskyCache& cache,
    const Eigen::VectorXd& omega
) {
    const auto& d0 = dec.d0;
    const auto& d1 = dec.d1;
    const auto& hodge1 = dec.hodge1;
    const auto& hodge1Inv = dec.hodge1Inverse;

    int nV = m.nV();
    int nE = static_cast<int>(omega.size());
    int nF = m.nF();

    HodgeResult result;
    result.omega = omega;
    result.exactPotential = Eigen::VectorXd::Zero(nV);
    result.coExactPotentialV = Eigen::VectorXd::Zero(nV);
    result.combinedPotential = Eigen::VectorXd::Zero(nV);
    result.dAlpha = Eigen::VectorXd::Zero(nE);
    result.deltaBeta = Eigen::VectorXd::Zero(nE);
    result.gamma = Eigen::VectorXd::Zero(nE);

    // ── Exact component: α and dα ────────────────────────────
    // Solve A α = d0ᵀ ★₁ ω  where  A = d0ᵀ ★₁ d0 (regularized).
    Eigen::VectorXd rhsAlpha = d0.transpose() * (hodge1 * omega);
    const auto& lltA = cache.hodgeExact(dec);
    result.exactPotential = lltA.solve(rhsAlpha);
    result.dAlpha = d0 * result.exactPotential;

    // ── Co-exact component: β (on faces) and δβ ──────────────
    // Solve B β̃ = d₁ ω  where  B = d₁ ★₁⁻¹ d₁ᵀ (regularized).
    // Then δβ = ★₁⁻¹ d₁ᵀ β̃  (the co-exact 1-form on edges).
    Eigen::VectorXd rhsBeta = d1 * omega;
    const auto& luB = cache.hodgeCoExact(dec);
    result.coExactPotentialF = luB.solve(rhsBeta);
    result.deltaBeta = hodge1Inv * (d1.transpose() * result.coExactPotentialF);

    // ── Harmonic component: γ = ω - dα - δβ ──────────────────
    result.gamma = omega - result.dAlpha - result.deltaBeta;

    // Average face β values to vertices (for visualization)
    auto& mesh = m.mesh();
    Eigen::VectorXd vertSum = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXi vertCount = Eigen::VectorXi::Zero(nV);
    for (Face f : mesh.faces()) {
        int fi = static_cast<int>(f.getIndex());
        double val = result.coExactPotentialF(fi);
        for (Vertex v : f.adjacentVertices()) {
            int vi = static_cast<int>(v.getIndex());
            vertSum(vi) += val;
            vertCount(vi)++;
        }
    }
    for (int i = 0; i < nV; i++) {
        if (vertCount(i) > 0) result.coExactPotentialV(i) = vertSum(i) / vertCount(i);
    }

    result.combinedPotential = result.exactPotential + result.coExactPotentialV;

    // ── Face-centered vector fields via Whitney interpolation ─
    result.omegaVectors     = whitney(m, dec, result.omega);
    result.dAlphaVectors    = whitney(m, dec, result.dAlpha);
    result.deltaBetaVectors = whitney(m, dec, result.deltaBeta);
    result.gammaVectors     = whitney(m, dec, result.gamma);

    std::cout << "[hodge] Decomposed 1-form. ||dα|| = " << result.dAlpha.norm()
              << ", ||δβ|| = " << result.deltaBeta.norm()
              << ", ||γ|| = " << result.gamma.norm() << std::endl;

    return result;
}

} // namespace nxr::manifold::solve
