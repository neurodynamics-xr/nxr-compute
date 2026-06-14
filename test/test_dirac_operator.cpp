#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <Eigen/Eigenvalues>
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

// A single flat triangle fan in the z=0 plane (Gauss map constant → D_N kernel).
static void flatPatch(std::vector<double>& V, std::vector<int32_t>& F) {
    V = { 0,0,0,  1,0,0,  0.5,1,0,  -0.5,1,0,  -1,0,0,  -0.5,-1,0,  0.5,-1,0 };
    F = { 0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1 };
}

static void testExtrinsicBlock() {
    std::cout << "\n=== dirac: extrinsic block E ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV();   // 12

    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlock(m);
    EXPECT(E.rows() == 4*N && E.cols() == 4*N, "E is [4V, 4V] = [48, 48]");

    Eigen::SparseMatrix<double> asym = E - Eigen::SparseMatrix<double>(E.transpose());
    EXPECT(asym.norm() < 1e-10, "E is symmetric");

    // PSD: smallest eigenvalue of dense(E) >= -tol
    Eigen::MatrixXd dense(E);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "E is positive-semidefinite");

    // On a curved mesh E is NOT all-zero (shape operator is nontrivial)
    EXPECT(E.norm() > 1e-6, "E is nonzero on a curved mesh");
}

static void testFlatKernel() {
    std::cout << "\n=== dirac: flat-region kernel ===\n";
    std::vector<double> V; std::vector<int32_t> F; flatPatch(V, F);
    Manifold m(V.data(), 7, F.data(), 6);
    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlock(m);
    // Flat: Gauss map constant ⇒ N_r − N_q ≡ 0 ⇒ D ≡ 0 ⇒ E ≡ 0 (entirely kernel).
    EXPECT(E.norm() < 1e-10, "E vanishes on a flat patch (pure kernel)");
}

// cotanL ⊗ I₄ built independently (the τ=0 anchor oracle).
static Eigen::SparseMatrix<double> kron4(const Eigen::SparseMatrix<double>& Lc) {
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(Lc.nonZeros()) * 4);
    for (int k = 0; k < Lc.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Lc, k); it; ++it)
            for (int c = 0; c < 4; ++c)
                T.emplace_back(4 * static_cast<int>(it.row()) + c,
                               4 * static_cast<int>(it.col()) + c, it.value());
    Eigen::SparseMatrix<double> K(4 * Lc.rows(), 4 * Lc.cols());
    K.setFromTriplets(T.begin(), T.end());
    return K;
}

static void testDiracFamily() {
    std::cout << "\n=== dirac: family L(τ) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV();

    // HEADLINE anchor: dirac(0) == cotanL ⊗ I₄ byte-for-byte.
    Eigen::SparseMatrix<double> L0 = m.operators().dirac(0.0);
    Eigen::SparseMatrix<double> anchor = kron4(m.operators().laplacian().cotan());
    EXPECT((L0 - anchor).norm() < 1e-12, "dirac(0) == cotanL ⊗ I4 (intrinsic anchor)");

    // Shape + symmetry across the family.
    for (double tau : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        Eigen::SparseMatrix<double> L = m.operators().dirac(tau);
        bool shape = (L.rows() == 4*N && L.cols() == 4*N);
        Eigen::SparseMatrix<double> asym = L - Eigen::SparseMatrix<double>(L.transpose());
        EXPECT(shape && asym.norm() < 1e-10,
               std::string("dirac(") + std::to_string(tau) + ") is [4V×4V] symmetric");
    }

    // dirac(1) == extrinsic block E.
    EXPECT((m.operators().dirac(1.0) - ops::dirac::extrinsicBlock(m)).norm() < 1e-12,
           "dirac(1) == extrinsicBlock");

    // Convex blend identity: dirac(τ) == (1−τ)dirac(0) + τ·dirac(1).
    double tau = 0.4;
    Eigen::SparseMatrix<double> blend =
        (1.0 - tau) * m.operators().dirac(0.0) + tau * m.operators().dirac(1.0);
    EXPECT((m.operators().dirac(tau) - blend).norm() < 1e-12, "dirac(τ) is the convex blend");

    // PSD for τ<1.
    Eigen::MatrixXd dense(m.operators().dirac(0.5));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "dirac(0.5) is PSD");

    // Out-of-range τ throws (Error derives from std::runtime_error — catch the base).
    bool threw = false;
    try { m.operators().dirac(1.5); } catch (const std::exception&) { threw = true; }
    EXPECT(threw, "dirac(τ>1) throws InvalidInput");
    bool threwNeg = false;
    try { m.operators().dirac(-0.1); } catch (const std::exception&) { threwNeg = true; }
    EXPECT(threwNeg, "dirac(τ<0) throws InvalidInput");
}

static void testDiracCache() {
    std::cout << "\n=== dirac: cache lifecycle ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    EXPECT(!m.isOperatorCached(OperatorId::Dirac), "Dirac not cached initially");
    m.operators().dirac(0.5);                       // builds E
    EXPECT(m.isOperatorCached(OperatorId::Dirac), "dirac(τ>0) caches E");
    // A second different-τ call reuses the cached E and still re-blends correctly.
    double tau = 0.9;
    Eigen::SparseMatrix<double> expect =
        (1.0 - tau) * kron4(m.operators().laplacian().cotan())
        + tau * ops::dirac::extrinsicBlock(m);
    EXPECT((m.operators().dirac(tau) - expect).norm() < 1e-12, "re-blend with cached E is correct");
    m.releaseOperator(OperatorId::Dirac);
    EXPECT(!m.isOperatorCached(OperatorId::Dirac), "releaseOperator(Dirac) clears E");
    // dirac(0) does NOT force the extrinsic build (τ=0 ⇒ pure intrinsic).
    m.operators().dirac(0.0);
    EXPECT(!m.isOperatorCached(OperatorId::Dirac), "dirac(0) does not build E");
}

static void testDiracEigenbasis() {
    std::cout << "\n=== dirac: eigenbasis ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    Eigen::SparseMatrix<double> L = m.operators().dirac(0.5);
    Eigen::SparseMatrix<double> B = kron4(m.operators().mass().galerkin());   // M_Galerkin ⊗ I₄
    const int k = 8;
    solve::EigenResult er = solve::eigen(L, B, k);
    Eigen::MatrixXd Phi = solve::normalize(er.eigenvectors, B);

    Eigen::MatrixXd gram = Phi.transpose() * (B * Phi);
    EXPECT((gram - Eigen::MatrixXd::Identity(gram.rows(), gram.cols())).cwiseAbs().maxCoeff() < 1e-9,
           "ΦᵀBΦ ≈ I (B-orthonormal eigenbasis)");
    EXPECT(er.eigenvalues.allFinite() && er.eigenvalues(0) <= er.eigenvalues(k-1),
           "eigenvalues finite & ascending");

    // Quaternionic structure: L(τ) commutes with right-ℍ-multiplication, so each
    // distinct eigenvalue has real multiplicity divisible by 4. With k=8 the
    // spectrum is two 4-fold multiplets — check each group is (near-)constant.
    auto groupSpread = [&](int g) {
        double lo = er.eigenvalues.segment(4*g, 4).minCoeff();
        double hi = er.eigenvalues.segment(4*g, 4).maxCoeff();
        return (hi - lo) / (1.0 + std::abs(hi));
    };
    EXPECT(groupSpread(0) < 1e-4 && groupSpread(1) < 1e-4,
           "eigenvalues form 4-fold quaternionic multiplets");
}

// ⋆_F : 4F×4F diagonal of face areas (each repeated 4×) — the Galerkin measure.
static Eigen::SparseMatrix<double> faceMass4(Manifold& m) {
    using namespace geometrycentral::surface;
    auto& geom = m.geometry();
    geom.requireFaceAreas();
    const int Fn = m.nF();
    std::vector<Eigen::Triplet<double>> T;
    for (Face f : m.mesh().faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        for (int a = 0; a < 4; ++a) T.emplace_back(4 * fi + a, 4 * fi + a, A);
    }
    Eigen::SparseMatrix<double> W(4 * Fn, 4 * Fn);
    W.setFromTriplets(T.begin(), T.end());
    return W;
}

static void testDiracMatrixD() {
    std::cout << "\n=== dirac: first-order operator D ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV();    // 12
    const int Fn = m.nF();   // 20

    Eigen::SparseMatrix<double> D = ops::dirac::matrix(m);
    EXPECT(D.rows() == 4*Fn && D.cols() == 4*N, "D is [4F, 4V] = [80, 48]");

    // HEADLINE anchor: E = DᵀW_F D byte-matches the cached extrinsic block (and
    // hence dirac(1)). The first-order D is the genuine square root of the family.
    Eigen::SparseMatrix<double> W = faceMass4(m);
    Eigen::SparseMatrix<double> Dt = D.transpose();
    Eigen::SparseMatrix<double> E_from_D = (Dt * W * D).pruned();
    EXPECT((E_from_D - ops::dirac::extrinsicBlock(m)).norm() < 1e-12,
           "DᵀW_F D == extrinsicBlock (E is D's Galerkin square)");

    // First-order property: D annihilates a CONSTANT quaternionic field (the
    // per-face cyclic normal differences telescope to zero) — the Dirac analogue
    // of "the Laplacian kills constants". Test all four constant unit quaternions.
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(4*N, 4);
    for (int v = 0; v < N; ++v)
        for (int c = 0; c < 4; ++c) U(4*v + c, c) = 1.0;
    EXPECT((D * U).cwiseAbs().maxCoeff() < 1e-10, "D kills constant quaternionic fields");

    // The facet accessor returns the same matrix as the free function.
    EXPECT((m.operators().diracD() - D).norm() < 1e-12, "operators().diracD() == ops::dirac::matrix");
}

static void testDiracMatrixFlat() {
    std::cout << "\n=== dirac: first-order D on a flat patch ===\n";
    std::vector<double> V; std::vector<int32_t> F; flatPatch(V, F);
    Manifold m(V.data(), 7, F.data(), 6);
    // Flat: Gauss map constant ⇒ every N_r − N_q ≡ 0 ⇒ D ≡ 0 (pure kernel).
    EXPECT(ops::dirac::matrix(m).norm() < 1e-10, "D vanishes on a flat patch");
}

static void testDiracMatrixCache() {
    std::cout << "\n=== dirac: first-order D cache lifecycle ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    EXPECT(!m.isOperatorCached(OperatorId::DiracD), "DiracD not cached initially");
    m.operators().diracD();
    EXPECT(m.isOperatorCached(OperatorId::DiracD), "diracD() caches D");
    // Independent of the squared-form cache: building D must not build E.
    EXPECT(!m.isOperatorCached(OperatorId::Dirac), "diracD() does not build E");
    m.releaseOperator(OperatorId::DiracD);
    EXPECT(!m.isOperatorCached(OperatorId::DiracD), "releaseOperator(DiracD) clears D");
}

int main() {
    testExtrinsicBlock();
    testFlatKernel();
    testDiracFamily();
    testDiracCache();
    testDiracEigenbasis();
    testDiracMatrixD();
    testDiracMatrixFlat();
    testDiracMatrixCache();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
