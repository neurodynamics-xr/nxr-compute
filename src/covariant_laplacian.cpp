#include "nxr/compute.h"
#include <Eigen/Geometry>
#include <vector>
#include <complex>

namespace nxr::manifold::ops::laplacian::connection {

Eigen::SparseMatrix<double> assembleCovariantLaplacian(
    CovariantCoupling coupling,
    const Eigen::SparseMatrix<std::complex<double>>& K,
    const Eigen::MatrixXcd& gaugeGrid,
    const Eigen::SparseMatrix<double>& cotanL) {

    const int N = static_cast<int>(cotanL.rows());
    if (K.rows() != N || K.cols() != N || gaugeGrid.rows() != N || gaugeGrid.cols() != 3 || cotanL.cols() != N) {
        throw Error(ErrorCode::InvalidInput,
            "assembleCovariantLaplacian: K, gaugeGrid, cotanL dimensions disagree",
            "Expected K [N×N] complex, gaugeGrid [N×3] complex, cotanL [N×N] real for a single N.");
    }
    std::vector<Eigen::Triplet<double>> T;

    if (coupling == CovariantCoupling::Product) {
        // tangent 2N block = [[ReK, -ImK],[ImK, ReK]]
        for (int k = 0; k < K.outerSize(); ++k)
            for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(K, k); it; ++it) {
                int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
                double re = it.value().real(), im = it.value().imag();
                T.emplace_back(i,       j,       re);   // aa
                T.emplace_back(i,       N + j,  -im);   // ab
                T.emplace_back(N + i,   j,       im);   // ba
                T.emplace_back(N + i,   N + j,   re);   // bb
            }
        // normal block cc = cotanL
        for (int k = 0; k < cotanL.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(cotanL, k); it; ++it)
                T.emplace_back(2*N + static_cast<int>(it.row()),
                               2*N + static_cast<int>(it.col()), it.value());
    } else { // Ambient: L3[i,j] (3×3) = cotanL[i,j] · (Fiᵀ Fj)
        // precompute per-vertex 3×3 frame F_v (columns e1, e2, n)
        std::vector<Eigen::Matrix3d> Fv(N);
        for (int v = 0; v < N; ++v) {
            Eigen::Vector3d e1 = gaugeGrid.row(v).real();
            Eigen::Vector3d e2 = gaugeGrid.row(v).imag();
            Eigen::Vector3d nrm = e1.cross(e2);
            Fv[v].col(0) = e1; Fv[v].col(1) = e2; Fv[v].col(2) = nrm;
        }
        for (int k = 0; k < cotanL.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(cotanL, k); it; ++it) {
                int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
                double w = it.value();
                Eigen::Matrix3d M = Fv[i].transpose() * Fv[j];   // (Fiᵀ Fj)
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q)
                        T.emplace_back(p*N + i, q*N + j, w * M(p,q));
            }
    }

    Eigen::SparseMatrix<double> L3(3*N, 3*N);
    L3.setFromTriplets(T.begin(), T.end());
    L3.makeCompressed();
    return L3;
}

} // namespace nxr::manifold::ops::laplacian::connection
