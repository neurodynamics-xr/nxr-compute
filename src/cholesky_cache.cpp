#include "nxr/compute.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace nxr::compute {

// All cache methods follow the same shape: lazy-init on first call,
// return the stored factor on every subsequent call. Single-threaded
// by construction — the addon is invoked from Node's main thread, so
// no mutex is needed.

namespace {

// Add ε to the diagonal of a sparse matrix and return the result.
// Performed via coeffRef(i,i) += ε for every i, matching the order
// the pre-cache inline solvers used (preserves bit-for-bit equivalence).
Eigen::SparseMatrix<double> regularize(const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> R = A;
    int n = static_cast<int>(R.rows());
    for (int i = 0; i < n; i++) {
        R.coeffRef(i, i) += CholeskyCache::kRegularization;
    }
    return R;
}

double elapsedMs(std::chrono::steady_clock::time_point t0) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

} // namespace

const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>&
CholeskyCache::laplacian(const Eigen::SparseMatrix<double>& K) {
    if (!laplacian_) {
        auto t0 = std::chrono::steady_clock::now();

        Eigen::SparseMatrix<double> L_reg = regularize(K);
        auto llt = std::make_unique<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>>();
        llt->compute(L_reg);
        if (llt->info() != Eigen::Success) {
            throw Error(ErrorCode::GeometryDegenerate,
                "Cholesky factorization of (K + εI) failed",
                "Matrix may be singular or non-PSD (degenerate mesh, "
                "disconnected graph component, etc.)");
        }
        laplacian_ = std::move(llt);

        std::cout << "[cholesky_cache] Factored K (n=" << K.rows()
                  << ") in " << elapsedMs(t0) << " ms" << std::endl;
    }
    return *laplacian_;
}

const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>&
CholeskyCache::hodgeExact(const DECOperators& dec) {
    if (!hodgeExact_) {
        auto t0 = std::chrono::steady_clock::now();

        // A = d0ᵀ ★₁ d0   (the DEC vertex Laplacian)
        Eigen::SparseMatrix<double> A = dec.d0.transpose() * dec.hodge1 * dec.d0;
        Eigen::SparseMatrix<double> A_reg = regularize(A);

        auto llt = std::make_unique<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>>();
        llt->compute(A_reg);
        if (llt->info() != Eigen::Success) {
            throw Error(ErrorCode::GeometryDegenerate,
                "Cholesky factorization of (d0ᵀ ★₁ d0 + εI) failed",
                "Mesh may be degenerate or have boundary inconsistencies");
        }
        hodgeExact_ = std::move(llt);

        std::cout << "[cholesky_cache] Factored d0ᵀ★₁d0 (n=" << A.rows()
                  << ") in " << elapsedMs(t0) << " ms" << std::endl;
    }
    return *hodgeExact_;
}

const Eigen::SparseLU<Eigen::SparseMatrix<double>>&
CholeskyCache::hodgeCoExact(const DECOperators& dec) {
    if (!hodgeCoExact_) {
        auto t0 = std::chrono::steady_clock::now();

        // B = d1 ★₁⁻¹ d1ᵀ   (the face Laplacian for co-exact solve)
        Eigen::SparseMatrix<double> B = dec.d1 * dec.hodge1Inverse * dec.d1.transpose();
        Eigen::SparseMatrix<double> B_reg = regularize(B);

        auto lu = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
        lu->compute(B_reg);
        if (lu->info() != Eigen::Success) {
            throw Error(ErrorCode::GeometryDegenerate,
                "LU factorization of (d1 ★₁⁻¹ d1ᵀ + εI) failed",
                "Mesh may be degenerate or have boundary inconsistencies");
        }
        hodgeCoExact_ = std::move(lu);

        std::cout << "[cholesky_cache] Factored d1★₁⁻¹d1ᵀ (n=" << B.rows()
                  << ") in " << elapsedMs(t0) << " ms" << std::endl;
    }
    return *hodgeCoExact_;
}

} // namespace nxr::compute
