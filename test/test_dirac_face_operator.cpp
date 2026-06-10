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

static void testExtrinsicBlockFace() {
    std::cout << "\n=== diracFace: extrinsic block E~ ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int Fn = m.nF();   // 20

    Eigen::SparseMatrix<double> E = ops::dirac::extrinsicBlockFace(m);
    EXPECT(E.rows() == 4*Fn && E.cols() == 4*Fn, "E~ is [4F, 4F] = [80, 80]");

    Eigen::SparseMatrix<double> asym = E - Eigen::SparseMatrix<double>(E.transpose());
    EXPECT(asym.norm() < 1e-10, "E~ is symmetric");

    Eigen::MatrixXd dense(E);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "E~ is positive-semidefinite");
    EXPECT(E.norm() > 1e-6, "E~ is nonzero on a curved mesh");

    // Per-face constant is in the kernel (telescoping around each vertex star).
    Eigen::VectorXd x(4*Fn);
    for (int f = 0; f < Fn; ++f) { x(4*f+0)=0.3; x(4*f+1)=-0.7; x(4*f+2)=0.2; x(4*f+3)=0.5; }
    EXPECT((E * x).cwiseAbs().maxCoeff() < 1e-10, "E~*(face-constant) = 0 (telescoping kernel)");

    // GENUINE quaternionic coupling: at least one 4×4 face-face block must be a
    // NON-scalar matrix (off-diagonal within the block). This is the headline
    // property the vertex-star aggregation buys — a degenerate edge/scalar-Laplacian
    // assembly would give every block ∝ I₄ and still pass all checks above.
    Eigen::MatrixXd densE(E);
    bool coupled = false;
    for (int fi = 0; fi < Fn && !coupled; ++fi)
        for (int fj = 0; fj < Fn && !coupled; ++fj) {
            if (fi == fj) continue;
            Eigen::Matrix4d blk = densE.block(4*fi, 4*fj, 4, 4);
            if (blk.norm() < 1e-12) continue;
            coupled = (blk - (blk.trace()/4.0) * Eigen::Matrix4d::Identity()).norm() > 1e-10;
        }
    EXPECT(coupled, "E~ has genuine quaternionic coupling (non-scalar 4x4 blocks)");
}

// A triangle fan in the z=0 plane — an OPEN disk (it has a boundary loop).
static void borderedPatch(std::vector<double>& V, std::vector<int32_t>& F) {
    V = { 0,0,0,  1,0,0,  0.5,1,0,  -0.5,1,0,  -1,0,0,  -0.5,-1,0,  0.5,-1,0 };
    F = { 0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1 };
}

static void testBoundaryThrows() {
    std::cout << "\n=== diracFace: open boundary fails loud ===\n";
    std::vector<double> V; std::vector<int32_t> F; borderedPatch(V, F);
    Manifold m(V.data(), 7, F.data(), 6);
    // diracFace is closed-mesh v1: an open boundary (open vertex stars) must throw,
    // not silently produce a degraded operator (CLAUDE.md §6 fail-loud).
    bool threw = false;
    try { ops::dirac::extrinsicBlockFace(m); } catch (const std::exception&) { threw = true; }
    EXPECT(threw, "extrinsicBlockFace throws on an open boundary");
    bool threwFamily = false;
    try { m.operators().diracFace(1.0); } catch (const std::exception&) { threwFamily = true; }
    EXPECT(threwFamily, "diracFace(τ>0) throws on an open boundary");
}

// K̃ = d₁ ⋆₁⁻¹ d₁ᵀ, the DEC 2-form Laplacian (independent oracle for the τ=0 anchor).
static Eigen::SparseMatrix<double> twoFormLaplacian(Manifold& m) {
    const ops::DECOperators& dec = m.operators().dec();
    Eigen::SparseMatrix<double> d1t = dec.d1.transpose();
    return dec.d1 * dec.hodge1Inverse * d1t;     // [F×E][E×E][E×F] = [F×F]
}

// K ⊗ I₄ in the face-interleaved 4f+c layout (= kron(K, I₄)).
static Eigen::SparseMatrix<double> kron4(const Eigen::SparseMatrix<double>& K) {
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(K.nonZeros()) * 4);
    for (int k = 0; k < K.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it)
            for (int c = 0; c < 4; ++c)
                T.emplace_back(4 * static_cast<int>(it.row()) + c,
                               4 * static_cast<int>(it.col()) + c, it.value());
    Eigen::SparseMatrix<double> out(4 * K.rows(), 4 * K.cols());
    out.setFromTriplets(T.begin(), T.end());
    return out;
}

static void testDiracFaceFamily() {
    std::cout << "\n=== diracFace: family L~(τ) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int Fn = m.nF();

    Eigen::SparseMatrix<double> L0 = m.operators().diracFace(0.0);
    Eigen::SparseMatrix<double> anchor = kron4(twoFormLaplacian(m));
    EXPECT((L0 - anchor).norm() < 1e-12, "diracFace(0) == K~ kron I4 (intrinsic anchor)");

    for (double tau : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        Eigen::SparseMatrix<double> L = m.operators().diracFace(tau);
        bool shape = (L.rows() == 4*Fn && L.cols() == 4*Fn);
        Eigen::SparseMatrix<double> asym = L - Eigen::SparseMatrix<double>(L.transpose());
        EXPECT(shape && asym.norm() < 1e-10,
               std::string("diracFace(") + std::to_string(tau) + ") is [4F x 4F] symmetric");
    }

    EXPECT((m.operators().diracFace(1.0) - ops::dirac::extrinsicBlockFace(m)).norm() < 1e-12,
           "diracFace(1) == extrinsicBlockFace");

    double tau = 0.4;
    Eigen::SparseMatrix<double> blend =
        (1.0 - tau) * m.operators().diracFace(0.0) + tau * m.operators().diracFace(1.0);
    EXPECT((m.operators().diracFace(tau) - blend).norm() < 1e-12, "diracFace(τ) is the convex blend");

    Eigen::MatrixXd dense(m.operators().diracFace(0.5));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "diracFace(0.5) is PSD");

    bool threw = false;
    try { m.operators().diracFace(1.5); } catch (const std::exception&) { threw = true; }
    EXPECT(threw, "diracFace(τ>1) throws");
    bool threwNeg = false;
    try { m.operators().diracFace(-0.1); } catch (const std::exception&) { threwNeg = true; }
    EXPECT(threwNeg, "diracFace(τ<0) throws");
}

static void testDiracFaceCache() {
    std::cout << "\n=== diracFace: cache lifecycle ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    EXPECT(!m.isOperatorCached(OperatorId::DiracFace), "DiracFace not cached initially");
    m.operators().diracFace(0.5);
    EXPECT(m.isOperatorCached(OperatorId::DiracFace), "diracFace(τ>0) caches Ẽ");
    m.releaseOperator(OperatorId::DiracFace);
    EXPECT(!m.isOperatorCached(OperatorId::DiracFace), "releaseOperator(DiracFace) clears Ẽ");
    m.operators().diracFace(0.0);
    EXPECT(!m.isOperatorCached(OperatorId::DiracFace), "diracFace(0) does not build Ẽ");
}

static void testDiracFaceEigenbasis() {
    std::cout << "\n=== diracFace: eigenbasis ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto& geom = m.geometry(); geom.requireFaceAreas();
    const int Fn = m.nF();

    Eigen::SparseMatrix<double> L = m.operators().diracFace(0.5);
    std::vector<Eigen::Triplet<double>> TB; TB.reserve(4*Fn);
    for (auto f : m.mesh().faces())
        for (int c = 0; c < 4; ++c)
            TB.emplace_back(4*static_cast<int>(f.getIndex())+c,
                            4*static_cast<int>(f.getIndex())+c, geom.faceAreas[f]);
    Eigen::SparseMatrix<double> B(4*Fn, 4*Fn); B.setFromTriplets(TB.begin(), TB.end());

    // B-orthonormal face eigenbasis via the production solver (Spectra path).
    const int k = 8;
    solve::EigenResult er = solve::eigen(L, B, k);
    Eigen::MatrixXd Phi = solve::normalize(er.eigenvectors, B);
    Eigen::MatrixXd gram = Phi.transpose() * (B * Phi);
    EXPECT((gram - Eigen::MatrixXd::Identity(gram.rows(), gram.cols())).cwiseAbs().maxCoeff() < 1e-9,
           "Phi^T B Phi ~ I (B-orthonormal face eigenbasis)");
    EXPECT(er.eigenvalues.allFinite() && er.eigenvalues(0) <= er.eigenvalues(k-1),
           "eigenvalues finite & ascending");

    // Quaternionic structure: L̃(τ) is built from left-quaternion-mult blocks (Ẽ) plus the
    // block-scalar K̃⊗I₄ and is solved against B = (face areas)⊗I₄, so it commutes with
    // right-ℍ-multiplication ⇒ EVERY eigenvalue multiplicity is divisible by 4. Verify on the
    // EXACT dense generalized spectrum: Spectra miscounts tightly-degenerate clusters (it
    // returns e.g. 5 for an 8-fold on this icosphere), so it cannot be used for multiplicities.
    Eigen::MatrixXd Ld(L), Bd(B);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(Ld, Bd);
    Eigen::VectorXd ev = ges.eigenvalues();
    bool allDiv4 = true;
    for (int i = 0; i < ev.size(); ) {
        int j = i; while (j < ev.size() && std::abs(ev(j) - ev(i)) < 1e-6) ++j;
        if ((j - i) % 4 != 0) allDiv4 = false;
        i = j;
    }
    EXPECT(allDiv4, "all eigenvalue multiplicities divisible by 4 (right-ℍ quaternionic structure)");
}

int main() {
    testExtrinsicBlockFace();
    testBoundaryThrows();
    testDiracFaceFamily();
    testDiracFaceCache();
    testDiracFaceEigenbasis();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
