#include "nxr/compute.h"
#include "nxr/facets.h"

#include <vector>

namespace nxr::manifold::differential {

// Extrinsic vector Weitzenböck Δ₃ + W_extrinsic on the ambient (ℝ³) bundle.
//
// CONSTRUCTION (the principled, geometrically-meaningful one): this operator IS
// the imaginary (vector) block of the immersion SQUARED Dirac
//
//     S = 2·dirac(0.5) = dirac(0) + dirac(1) = Δ₄ + D_N            [4V × 4V]
//
// stored vertex-interleaved as 4v+c (c=0 scalar/real part, c∈{1,2,3} the
// imaginary quaternion components x,y,z). The Dirac embeds the Gauss map
// (per-vertex normals) as world-frame ℝ³ vectors, and the quaternion-imaginary
// axes (x,y,z) ARE those same ambient world axes — so the vector part needs NO
// per-vertex basis change. We extract the rows/cols with c∈{1,2,3}, drop the
// scalar (c=0) border, and reindex 4v+c → (c-1)·N + v, giving a [3V×3V] real
// matrix in WORLD-frame component-major layout (the same component-major layout
// gradient3D / covariantLaplacian use, but in the world frame rather than the
// per-vertex local frame).
//
// The component-DIAGONAL part of S comes from dirac(0) = cotanL ⊗ I₄, whose imag
// block is exactly kron(I₃, cotanLaplacian) (the flat ambient covariant
// Laplacian in world coords). The component-OFF-diagonal coupling comes from
// dirac(1) = D_N; its imag block is the extrinsic curvature endomorphism W_ext,
// which is identically zero on a flat/developable region (constant normals ⇒
// D_N=0) and nonzero on curved regions.
Eigen::SparseMatrix<double> assembleExtrinsicWeitzenbock(Manifold& m) {
    const int N = m.nV();

    // Squared immersion Dirac S = Δ₄ + D_N. Built through the public operators
    // facet so it shares the single-source-of-truth dirac assembly (and so the
    // τ=0 / τ=1 blocks are byte-identical to what dirac(τ) returns).
    Eigen::SparseMatrix<double> S = m.operators().dirac(0.0) + m.operators().dirac(1.0);
    S.makeCompressed();

    // Imaginary-block extraction: keep entries whose BOTH quaternion components
    // are imaginary (c ∈ {1,2,3}); reindex 4v+c → (c-1)·N + v.
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(S.nonZeros()));
    for (int k = 0; k < S.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
            const int r = static_cast<int>(it.row());
            const int c = static_cast<int>(it.col());
            const int rc = r & 3;   // r % 4
            const int cc = c & 3;   // c % 4
            if (rc == 0 || cc == 0) continue;       // drop scalar (w) border
            const int rv = r >> 2;  // r / 4
            const int cv = c >> 2;  // c / 4
            T.emplace_back((rc - 1) * N + rv, (cc - 1) * N + cv, it.value());
        }

    Eigen::SparseMatrix<double> W(3 * N, 3 * N);
    W.setFromTriplets(T.begin(), T.end());
    W.makeCompressed();
    return W;
}

} // namespace nxr::manifold::differential
