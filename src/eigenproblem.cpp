// Named-operator eigensolve: the single source of truth for
// "operator → generalized eigenproblem (K, M, σ)", plus the Dirac ℍ-multiplet
// reconstruction and the small-mesh dense verification path.
// Design: docs/superpowers/specs/2026-06-15-operator-eigensolve-design.md.

#include "nxr/compute.h"
#include "nxr/facets.h"
#include "nxr/operator_registry.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <Eigen/Eigenvalues>   // GeneralizedSelfAdjointEigenSolver / SelfAdjointEigenSolver

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace nxr::manifold::solve {

// Largest dense generalized eigenproblem we allow (matrix is size×size doubles;
// the solver keeps a handful of copies). 4096 ⇒ ~134 MB per dense copy — fine on
// native, comfortably under the WASM 2 GB cap. Cortical Dirac (4·V) blows past
// this, which is exactly why dense is verification-only.
static constexpr int kDenseSizeCap = 4096;

Eigen::SparseMatrix<double> blockKron(const Eigen::SparseMatrix<double>& M, int b) {
    // Interleaved kron(M, I_b): entry (i,j,v) → (b·i+c, b·j+c, v), c=0..b−1.
    // Matches the Dirac vertex-interleaved 4v+c storage.
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(M.nonZeros()) * b);
    for (int o = 0; o < M.outerSize(); ++o)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, o); it; ++it)
            for (int c = 0; c < b; ++c)
                T.emplace_back(b * static_cast<int>(it.row()) + c,
                               b * static_cast<int>(it.col()) + c, it.value());
    Eigen::SparseMatrix<double> K(M.rows() * b, M.cols() * b);
    K.setFromTriplets(T.begin(), T.end());
    return K;
}

namespace {
// The vertex FEM mass the registry names as this operator's natural mass.
// The registry documents the DEFAULT family (e.g. "massGalerkin"); spec.mass remains
// the caller's lumped/galerkin override (historical behavior preserved). This helper
// makes the registry the documented source of truth while the construction stays
// byte-identical — the matrices built are the same as before the refactor.
Eigen::SparseMatrix<double> naturalVertexMass(Manifold& m, const char* id,
                                              ops::MassMatrixVariant variant) {
    // Registry is the documented source of truth for WHICH mass family this operator's
    // eigenproblem is posed against; the runtime spec.mass picks lumped/galerkin within
    // that family. Drift-guard: every operator routed here must document a massGalerkin
    // -family natural mass — assert so a future entry can't silently disagree with the
    // construction below. Construction is byte-identical to the pre-refactor inline code.
    const auto* v = registry::operatorById(id);
    assert(v && v->natural_mass.rfind("massGalerkin", 0) == 0 &&
           "registry natural_mass disagrees with eigenProblemFor's vertex-mass family");
    (void)v;  // unused in release (assert compiled out)
    return (variant == ops::MassMatrixVariant::Lumped)
               ? m.operators().mass().lumped()
               : m.operators().mass().galerkin();
}
} // namespace

EigenProblem eigenProblemFor(Manifold& m, const EigenOperatorSpec& spec) {
    EigenProblem p;
    p.sigma = -1e-8;
    switch (spec.op) {
        case EigenOperator::LaplacianCotan: {
            p.K = m.operators().laplacian().cotan();
            // Registry documents natural_mass == "massGalerkin" for laplaceBeltrami.
            p.M = naturalVertexMass(m, "laplaceBeltrami", spec.mass);
            p.blockSize = 1;
            break;
        }
        case EigenOperator::LaplacianGraph: {
            p.K = m.operators().laplacian().graph();
            const int n = m.nV();
            p.M.resize(n, n);
            p.M.setIdentity();              // standard eigenproblem; registry natural_mass == "identity" (metric-free)
            p.blockSize = 1;
            break;
        }
        case EigenOperator::Dirac: {
            p.K = m.operators().dirac(spec.tau);              // [4V×4V] (caches E)
            // Registry documents natural_mass == "massGalerkin*I4" for relativeDirac.
            const Eigen::SparseMatrix<double> Mv =
                naturalVertexMass(m, "relativeDirac", spec.mass);
            p.M = blockKron(Mv, 4);                           // M ⊗ I₄ (vertex-interleaved)
            p.blockSize = 4;
            break;
        }
        case EigenOperator::DiracFace: {
            p.K = m.operators().diracFace(spec.tau);          // [4F×4F]; throws on open boundary
            // Registry documents natural_mass == "diag(faceArea)*I4" for relativeFaceDirac.
            auto& geom = m.geometry();
            geom.requireFaceAreas();
            const int Fn = m.nF();
            std::vector<Eigen::Triplet<double>> T;
            T.reserve(Fn);
            for (auto f : m.mesh().faces())
                T.emplace_back(static_cast<int>(f.getIndex()),
                               static_cast<int>(f.getIndex()), geom.faceAreas[f]);
            Eigen::SparseMatrix<double> Af(Fn, Fn);
            Af.setFromTriplets(T.begin(), T.end());
            p.M = blockKron(Af, 4);                           // diag(faceArea) ⊗ I₄
            p.blockSize = 4;
            break;
        }
    }
    return p;
}

namespace {

// Right-quaternion multiplication maps on a [4n × c] block, component order
// [w,x,y,z] per vertex (rows 4v..4v+3). q·i, q·j, q·k respectively. Used to close
// a Dirac eigenspace under the ℍ-action it commutes with.
Eigen::MatrixXd rightQuat(const Eigen::MatrixXd& U, char unit) {
    const int rows = static_cast<int>(U.rows());
    const int n = rows / 4;
    Eigen::MatrixXd R(rows, U.cols());
    for (int v = 0; v < n; ++v) {
        const auto w = U.row(4 * v + 0);
        const auto x = U.row(4 * v + 1);
        const auto y = U.row(4 * v + 2);
        const auto z = U.row(4 * v + 3);
        switch (unit) {
            case 'i':  // [w,x,y,z] → [-x, w, z, -y]
                R.row(4*v+0) = -x; R.row(4*v+1) =  w; R.row(4*v+2) =  z; R.row(4*v+3) = -y; break;
            case 'j':  // → [-y,-z, w, x]
                R.row(4*v+0) = -y; R.row(4*v+1) = -z; R.row(4*v+2) =  w; R.row(4*v+3) =  x; break;
            case 'k':  // → [-z, y,-x, w]
                R.row(4*v+0) = -z; R.row(4*v+1) =  y; R.row(4*v+2) = -x; R.row(4*v+3) =  w; break;
        }
    }
    return R;
}

// Close each eigenvalue cluster under right-ℍ and return an M-orthonormal basis of
// the true (multiple-of-4) eigenspaces — recovers complete multiplets even when
// Spectra under-converged a degenerate cluster.
EigenResult reconstructDiracMultiplets(const EigenResult& r,
                                       const Eigen::SparseMatrix<double>& M) {
    const int rows = static_cast<int>(r.eigenvectors.rows());
    const int kc   = static_cast<int>(r.eigenvalues.size());
    const double clusterTol = 1e-6;     // relative eigenvalue gap that splits clusters
    const double rankTol    = 1e-9;     // M-Gram eigenvalue floor (relative to max)

    std::vector<double> outVals;
    std::vector<Eigen::MatrixXd> outBlocks;

    int i = 0;
    while (i < kc) {
        int j = i + 1;
        while (j < kc &&
               std::abs(r.eigenvalues(j) - r.eigenvalues(i)) <
                   clusterTol * (1.0 + std::abs(r.eigenvalues(i))))
            ++j;
        const int c = j - i;
        const double lambda = r.eigenvalues.segment(i, c).mean();

        // Orbit O = [S | S·i | S·j | S·k]  ([rows × 4c]).
        Eigen::MatrixXd S = r.eigenvectors.middleCols(i, c);
        Eigen::MatrixXd O(rows, 4 * c);
        O.middleCols(0,     c) = S;
        O.middleCols(c,     c) = rightQuat(S, 'i');
        O.middleCols(2 * c, c) = rightQuat(S, 'j');
        O.middleCols(3 * c, c) = rightQuat(S, 'k');

        // Rank-reveal an M-orthonormal basis of span(O) via the dense M-Gram.
        Eigen::MatrixXd MO = M * O;
        Eigen::MatrixXd G  = O.transpose() * MO;                 // [4c × 4c], SPSD
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        const Eigen::VectorXd& g = es.eigenvalues();             // ascending
        const double gmax = g(g.size() - 1);
        for (int t = 0; t < g.size(); ++t) {
            if (g(t) <= rankTol * gmax) continue;                // null space of the orbit
            // basis column = O · w / sqrt(λ_G)  ⇒  columnᵀ M column = 1.
            Eigen::VectorXd col = O * es.eigenvectors().col(t) / std::sqrt(g(t));
            outBlocks.push_back(col);
            outVals.push_back(lambda);
        }
        i = j;
    }

    EigenResult out;
    const int kOut = static_cast<int>(outVals.size());
    out.eigenvectors.resize(rows, kOut);
    out.eigenvalues.resize(kOut);
    for (int c = 0; c < kOut; ++c) {
        out.eigenvectors.col(c) = outBlocks[c];
        out.eigenvalues(c)      = outVals[c];
    }
    out.k = kOut;
    out.nConverged = kOut;
    return out;
}

// Exact dense generalized eigensolve (small meshes / verification). Returns the
// lowest k pairs; GeneralizedSelfAdjointEigenSolver yields M-orthonormal vectors.
EigenResult denseEigen(const EigenProblem& p, int k) {
    const int n = static_cast<int>(p.K.rows());
    if (n > kDenseSizeCap)
        throw Error(ErrorCode::EigensolveInvalidK,
            "eigenOperator: dense solve size " + std::to_string(n) +
            " exceeds cap " + std::to_string(kDenseSizeCap),
            "dense=true is verification-only; use the (default) sparse path on large meshes.");
    if (k < 1 || k > n)
        throw Error(ErrorCode::EigensolveInvalidK,
            "eigenOperator(dense): k=" + std::to_string(k) + " out of range [1, " +
            std::to_string(n) + "].");

    Eigen::MatrixXd Kd = Eigen::MatrixXd(p.K);
    Eigen::MatrixXd Md = Eigen::MatrixXd(p.M);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(Kd, Md);
    if (ges.info() != Eigen::Success)
        throw Error(ErrorCode::EigensolveNotConverged,
            "eigenOperator(dense): GeneralizedSelfAdjointEigenSolver failed.");

    EigenResult r;
    r.eigenvalues  = ges.eigenvalues().head(k);             // ascending
    r.eigenvectors = ges.eigenvectors().leftCols(k);        // MᵀMV = I by construction
    r.k = k;
    r.nConverged = k;
    return r;
}

} // namespace

EigenResult eigenOperator(Manifold& m, const EigenOperatorSpec& spec, int k,
                          double sigma, bool normalize, bool reconstructMultiplets,
                          bool dense, const CancellationToken& cancel,
                          const ProgressObserver& progress) {
    EigenProblem p = eigenProblemFor(m, spec);

    if (dense)
        return denseEigen(p, k);

    const bool doMultiplets = reconstructMultiplets && p.blockSize == 4;
    if (doMultiplets) {
        // Round up so we never truncate a multiplet mid-cluster; reconstruction
        // re-orthonormalizes, so request raw (un-normalized) vectors.
        int kReq = ((k + 3) / 4) * 4;
        const int n = static_cast<int>(p.K.rows());
        if (kReq > n - 1) kReq = n - 1;
        EigenResult raw = eigen(p.K, p.M, kReq, sigma, /*normalize=*/false,
                                /*removeDC=*/false, cancel, progress);
        return reconstructDiracMultiplets(raw, p.M);
    }

    return eigen(p.K, p.M, k, sigma, normalize, /*removeDC=*/false, cancel, progress);
}

} // namespace nxr::manifold::solve
