#include "nxr/compute.h"

namespace nxr::manifold::ops {

// L = d0ᵀ d0. d0 is the signed vertex→edge incidence (exterior derivative on
// 0-forms). d0ᵀd0 gives degree on the diagonal and −1 between adjacent vertices.
Eigen::SparseMatrix<double> graphLaplacian(Manifold& m) {
    const Eigen::SparseMatrix<double>& D0 = d0(m);   // passthrough accessor
    Eigen::SparseMatrix<double> L = D0.transpose() * D0;
    L.makeCompressed();
    return L;
}

} // namespace nxr::manifold::ops
