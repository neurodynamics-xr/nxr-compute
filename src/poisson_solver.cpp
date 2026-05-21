#include "nxr/compute.h"

#include <iostream>

namespace nxr::manifold::solve {

Eigen::VectorXd poisson(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    CholeskyCache& cache,
    const std::map<int, double>& densityMap
) {
    int n = static_cast<int>(K.rows());

    // 1. Build density vector ρ
    Eigen::VectorXd rho = Eigen::VectorXd::Zero(n);
    for (const auto& [idx, val] : densityMap) {
        rho(idx) = val;
    }

    // 2. M-weighted mean of ρ.
    // totalMass = Σᵢⱼ Mᵢⱼ = (1ᵀ M 1) — equivalent to the sum of the
    // Voronoi areas in the mesh case (M is diagonal there) and to the
    // sum of node weights in the graph case.
    Eigen::VectorXd Mrho = M * rho;
    double totalMass = (M * Eigen::VectorXd::Ones(n)).sum();
    double rhoBar = Mrho.sum() / totalMass;

    // 3. Build right-hand side: rhs = M * (ρ̄ - ρ)
    Eigen::VectorXd rhoBarVec = Eigen::VectorXd::Constant(n, rhoBar);
    Eigen::VectorXd rhs = M * (rhoBarVec - rho);

    // 4. Solve (K + ε·I) · φ = rhs using the cached factor.
    const auto& llt = cache.laplacian(K);
    Eigen::VectorXd phi = llt.solve(rhs);
    if (llt.info() != Eigen::Success) {
        std::cerr << "[poisson] Solve failed" << std::endl;
        return Eigen::VectorXd::Zero(n);
    }

    std::cout << "[poisson] Solved: " << densityMap.size()
              << " density points, range [" << phi.minCoeff()
              << ", " << phi.maxCoeff() << "]" << std::endl;

    return phi;
}

} // namespace nxr::manifold::solve