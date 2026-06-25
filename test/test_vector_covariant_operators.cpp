/**
 * test_vector_covariant_operators.cpp — Cell A (connectionGradient d^∇) squares to identity.
 *
 * Verifies: (d^∇)ᴴ ⋆₁ d^∇ == connectionLaplacian (nSym 1 and 2) to < 1e-9.
 *
 * Build: cmake --build build --target test_vector_covariant_operators
 * Run:   ./build/test_vector_covariant_operators
 */

#include "nxr/compute.h"
#include "nxr/facets.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace nxr::manifold::ops::laplacian::connection;

static int g_failures = 0;
#define CHECK(c, m)                                                      \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::cerr << "FAIL: " << (m) << "\n";                       \
            ++g_failures;                                                \
        }                                                                \
    } while (0)

// ── icosphere mesh ────────────────────────────────────────────────────────────
// Exact replication of generateIcosphere() from test_mass_variants.cpp.
static void generateIcosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    V.clear();
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        V.push_back(raw[i]     / len);
        V.push_back(raw[i + 1] / len);
        V.push_back(raw[i + 2] / len);
    }
    F = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
}

// ── helper: cast real sparse to complex ───────────────────────────────────────
static Eigen::SparseMatrix<std::complex<double>>
toComplex(const Eigen::SparseMatrix<double>& A) {
    return A.cast<std::complex<double>>();
}

// ── helper: max |A − B| over stored entries ───────────────────────────────────
static double maxAbsDiff(const Eigen::SparseMatrix<std::complex<double>>& A,
                         const Eigen::SparseMatrix<std::complex<double>>& B) {
    Eigen::SparseMatrix<std::complex<double>> D = A - B;
    double m = 0.0;
    for (int k = 0; k < D.outerSize(); ++k)
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(D, k); it; ++it)
            m = std::max(m, std::abs(it.value()));
    return m;
}

// ── helper: max |A − B| over stored entries (real) ────────────────────────────
static double maxAbsDiffReal(const Eigen::SparseMatrix<double>& A,
                             const Eigen::SparseMatrix<double>& B) {
    Eigen::SparseMatrix<double> D = A - B;
    double m = 0.0;
    for (int k = 0; k < D.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(D, k); it; ++it)
            m = std::max(m, std::abs(it.value()));
    return m;
}

static double maxAbsReal(const Eigen::SparseMatrix<double>& A) {
    double m = 0.0;
    for (int k = 0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
            m = std::max(m, std::abs(it.value()));
    return m;
}

// Extract the imaginary (vector) [3V×3V] block of a [4V×4V] vertex-interleaved
// (4v+c) operator, EXACTLY the way assembleExtrinsicWeitzenbock does — used to
// re-derive the cross-bundle reference independently from the public dirac(τ).
static Eigen::SparseMatrix<double> imagBlock(const Eigen::SparseMatrix<double>& S, int N) {
    std::vector<Eigen::Triplet<double>> T;
    for (int k = 0; k < S.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
            int r = static_cast<int>(it.row()), c = static_cast<int>(it.col());
            int rc = r & 3, cc = c & 3;
            if (rc == 0 || cc == 0) continue;
            T.emplace_back((rc - 1) * N + (r >> 2), (cc - 1) * N + (c >> 2), it.value());
        }
    Eigen::SparseMatrix<double> W(3 * N, 3 * N);
    W.setFromTriplets(T.begin(), T.end());
    W.makeCompressed();
    return W;
}

// kron(I₃, cotanL) in WORLD-frame component-major (c·N+i) — the flat ambient
// covariant Laplacian, built INDEPENDENTLY from the cotan Laplacian (no Dirac).
static Eigen::SparseMatrix<double> kronI3(const Eigen::SparseMatrix<double>& L, int N) {
    std::vector<Eigen::Triplet<double>> T;
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            for (int c = 0; c < 3; ++c)
                T.emplace_back(c * N + static_cast<int>(it.row()),
                               c * N + static_cast<int>(it.col()), it.value());
    Eigen::SparseMatrix<double> K(3 * N, 3 * N);
    K.setFromTriplets(T.begin(), T.end());
    K.makeCompressed();
    return K;
}

// Conjugate a LOCAL-frame component-major [3N×3N] operator into WORLD frame:
// world_block(i,j) = F_i · local_block(i,j) · F_jᵀ, with F_v = [e1|e2|n] (cols).
static Eigen::SparseMatrix<double> localToWorld(const Eigen::SparseMatrix<double>& A,
                                                nxr::manifold::Manifold& m) {
    const int N = m.nV();
    nxr::manifold::geometry::VertexFrames vf = nxr::manifold::geometry::vertexFrames(m);
    std::vector<Eigen::Matrix3d> F(N);
    for (int v = 0; v < N; ++v) {
        F[v].col(0) = vf.e1.row(v).transpose();
        F[v].col(1) = vf.e2.row(v).transpose();
        F[v].col(2) = vf.normals.row(v).transpose();
    }
    std::map<std::pair<int,int>, Eigen::Matrix3d> blocks;
    for (int k = 0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            int r = static_cast<int>(it.row()), c = static_cast<int>(it.col());
            int i = r % N, j = c % N, pr = r / N, pc = c / N;
            auto key = std::make_pair(i, j);
            auto found = blocks.find(key);
            if (found == blocks.end())
                found = blocks.emplace(key, Eigen::Matrix3d::Zero()).first;
            found->second(pr, pc) += it.value();
        }
    std::vector<Eigen::Triplet<double>> T;
    for (auto& kv : blocks) {
        int i = kv.first.first, j = kv.first.second;
        Eigen::Matrix3d W = F[i] * kv.second * F[j].transpose();
        for (int pr = 0; pr < 3; ++pr)
            for (int pc = 0; pc < 3; ++pc)
                if (W(pr, pc) != 0.0)
                    T.emplace_back(pr * N + i, pc * N + j, W(pr, pc));
    }
    Eigen::SparseMatrix<double> out(3 * N, 3 * N);
    out.setFromTriplets(T.begin(), T.end());
    out.makeCompressed();
    return out;
}

// ── Cell D: extrinsicWeitzenbock — construction, anchors, curvature sensitivity ─
static void test_cellD_extrinsic_weitzenbock() {
    using namespace nxr::manifold;
    std::vector<double>   V;
    std::vector<int32_t>  F;
    generateIcosphere(V, F);
    Manifold m(V.data(), static_cast<int>(V.size()) / 3,
               F.data(), static_cast<int>(F.size()) / 3);
    const int N = m.nV();

    // (D1) Facet wiring + cache.
    const Eigen::SparseMatrix<double>& W = m.operators().extrinsicWeitzenbock();
    CHECK(W.rows() == 3 * N && W.cols() == 3 * N, "extrinsicWeitzenbock shape [3V×3V]");
    CHECK(m.isOperatorCached(OperatorId::ExtrinsicWeitzenbock), "cached after access");
    const Eigen::SparseMatrix<double>& Wb = m.operators().extrinsicWeitzenbock();
    CHECK(&W == &Wb, "same cached ref on second call");
    std::cout << "  D1 facet/cache: [" << W.rows() << "×" << W.cols()
              << "] nnz=" << W.nonZeros() << "\n";

    // (D2) Cross-bundle anchor — TAUTOLOGICAL BY CONSTRUCTION (W *is* this block).
    // Kept only as a wiring/round-trip check that the public squared-Dirac surface
    // and the operator agree; it proves nothing geometric on its own.
    Eigen::SparseMatrix<double> S = m.operators().dirac(0.5);
    S = (2.0 * S).pruned();
    Eigen::SparseMatrix<double> Wref = imagBlock(S, N);
    double dCross = maxAbsDiffReal(W, Wref);
    std::cout << "  D2 cross-bundle anchor (TAUTOLOGY) maxAbsDiff = " << dCross << "\n";
    CHECK(dCross < 1e-9, "W == imag block of 2·dirac(0.5) (wiring)");

    // (D3) INDEPENDENT: symmetry of the resulting ℝ³ operator.
    Eigen::SparseMatrix<double> Wt = Eigen::SparseMatrix<double>(W.transpose());
    double dSym = maxAbsDiffReal(W, Wt);
    std::cout << "  D3 symmetry maxAbs(W - Wᵀ) = " << dSym << "\n";
    CHECK(dSym < 1e-12, "extrinsicWeitzenbock is symmetric");

    // (D4) INDEPENDENT: flat anchor + extrinsic content.
    // Build kron(I₃, cotanL) directly from the cotan Laplacian (no Dirac path),
    // and separately confirm the imag block of dirac(0) equals it byte-wise
    // (the documented τ=0 anchor). Then W − flat is the pure extrinsic term.
    Eigen::SparseMatrix<double> cotanL = m.operators().laplacian().cotan();
    Eigen::SparseMatrix<double> flatWorld = kronI3(cotanL, N);
    Eigen::SparseMatrix<double> imagDirac0 = imagBlock(
        (Eigen::SparseMatrix<double>)m.operators().dirac(0.0), N);
    double dFlatAnchor = maxAbsDiffReal(imagDirac0, flatWorld);
    std::cout << "  D4 imag(dirac(0)) == kron(I₃,cotanL) maxAbsDiff = " << dFlatAnchor << "\n";
    CHECK(dFlatAnchor < 1e-12, "flat block byte-matches kron(I₃,cotanL)");

    Eigen::SparseMatrix<double> Eimag = (W - flatWorld).pruned();
    double extMax = maxAbsReal(Eimag);
    std::cout << "  D4 extrinsic term ‖W − flat‖_max = " << extMax
              << "  (nnz=" << Eimag.nonZeros() << ")\n";
    CHECK(extMax > 1e-6, "extrinsic curvature term is nonzero on the (curved) sphere");

    // (D5) Constant-field response — HONEST FINDING (logged, not force-asserted).
    // A world-constant vector field [1;0;0] over all vertices. The flat part
    // (cotanL kills constants) annihilates it. The first-order Dirac ALSO kills
    // constant quaternions, so the FULL operator is expected to annihilate it too
    // — we report the actual norms rather than asserting "full ≠ 0".
    Eigen::VectorXd cst = Eigen::VectorXd::Zero(3 * N);
    for (int v = 0; v < N; ++v) cst(0 * N + v) = 1.0;   // world-x constant
    double flatConst = (flatWorld * cst).norm();
    double fullConst = (W * cst).norm();
    std::cout << "  D5 ‖flat·const‖ = " << flatConst
              << "   ‖W·const‖ = " << fullConst
              << "   (both ~0 expected: Dirac kills constants)\n";
    CHECK(flatConst < 1e-9, "flat part annihilates world-constant field");

    // (D6) 2·flat − product comparison — LOG ONLY (exploratory).
    // flat = Ambient covariant (world form = kron(I₃,cotanL)); product = Product
    // covariant transformed local→world. If maxAbsDiff ≈ 0 the algebraic Gauss
    // shortcut coincides with the geometric squared-Dirac construction.
    Eigen::SparseMatrix<double> prodLocal =
        m.operators().laplacian().covariant(
            ops::laplacian::connection::CovariantCoupling::Product);
    Eigen::SparseMatrix<double> ambLocal =
        m.operators().laplacian().covariant(
            ops::laplacian::connection::CovariantCoupling::Ambient);
    Eigen::SparseMatrix<double> prodWorld = localToWorld(prodLocal, m);
    Eigen::SparseMatrix<double> ambWorld  = localToWorld(ambLocal, m);
    // Sanity: ambient covariant in world == kron(I₃,cotanL).
    double dAmb = maxAbsDiffReal(ambWorld, flatWorld);
    Eigen::SparseMatrix<double> gauss = (2.0 * flatWorld - prodWorld).pruned();
    double dGauss = maxAbsDiffReal(W, gauss);
    std::cout << "  D6 (log) ‖ambientCovariant_world − kron(I₃,cotanL)‖ = " << dAmb << "\n";
    std::cout << "  D6 (log) ‖W − (2·flat − product_world)‖_max = " << dGauss << "\n";

    // Release.
    m.releaseOperator(OperatorId::ExtrinsicWeitzenbock);
    CHECK(!m.isOperatorCached(OperatorId::ExtrinsicWeitzenbock), "released");
}

// ── Cell A: (d^∇)ᴴ ⋆₁ d^∇ == connectionLaplacian ────────────────────────────
static void test_cellA_squares_to() {
    std::vector<double>   V;
    std::vector<int32_t>  F;
    generateIcosphere(V, F);

    for (int nSym : {1, 2}) {
        // Fresh Manifold per nSym to avoid any cached-state cross-contamination.
        nxr::manifold::Manifold m(V.data(), static_cast<int>(V.size()) / 3,
                                  F.data(), static_cast<int>(F.size()) / 3);

        // Assemble the gradient D: [E×V] complex
        Eigen::SparseMatrix<std::complex<double>> D = assembleConnectionGradient(m, nSym);

        // ⋆₁ = diag(edgeCotanWeights) — the Hodge-1 star in real form.
        // nxr::manifold::ops::hodge1(m) returns const SparseMatrix<double>&;
        // cast to complex for the Hermitian product.
        Eigen::SparseMatrix<std::complex<double>> W = toComplex(nxr::manifold::ops::hodge1(m));

        // DᴴWD — squares the gradient to the connection Laplacian.
        Eigen::SparseMatrix<std::complex<double>> sq = (D.adjoint() * W * D).pruned();

        // Reference: connection Laplacian (Complex format, no regularization)
        ConnectionLaplacianOptions opts;
        opts.nSym           = nSym;
        opts.regularization = 0.0;
        opts.format         = ConnectionLaplacianFormat::Complex;
        ConnectionLaplacian cl = assembleConnectionLaplacian(m, opts);

        double d = maxAbsDiff(sq, cl.K_complex);
        std::cout << "  Cell A squares_to nSym=" << nSym
                  << ": maxAbsDiff=" << d << "\n";
        CHECK(d < 1e-9,
              std::string("(d^∇)ᴴ⋆₁d^∇ == connectionLaplacian nSym=") +
              std::to_string(nSym));
    }
}

// ── Cell B: shape checks ──────────────────────────────────────────────────────
static void test_cellB_shape() {
    std::vector<double>   V;
    std::vector<int32_t>  F;
    generateIcosphere(V, F);

    nxr::manifold::Manifold m(V.data(), static_cast<int>(V.size()) / 3,
                              F.data(), static_cast<int>(F.size()) / 3);

    Eigen::SparseMatrix<std::complex<double>> D = assembleConnectionGradient(m, 1);
    const int E  = static_cast<int>(m.nE());
    const int Nv = static_cast<int>(m.nV());

    std::cout << "  Cell B shape: D is " << D.rows() << "×" << D.cols()
              << " (expected " << E << "×" << Nv << ")\n";
    CHECK(D.rows() == E,  "D.rows() == nEdges");
    CHECK(D.cols() == Nv, "D.cols() == nVertices");
    // Each edge contributes exactly 2 non-zeros (one +1 and one −ρ).
    CHECK(D.nonZeros() == 2 * E, "D.nonZeros() == 2*nEdges");
}

// ── Cell C: nSym <= 0 throws ──────────────────────────────────────────────────
static void test_cellC_invalid_nsym() {
    std::vector<double>   V;
    std::vector<int32_t>  F;
    generateIcosphere(V, F);

    nxr::manifold::Manifold m(V.data(), static_cast<int>(V.size()) / 3,
                              F.data(), static_cast<int>(F.size()) / 3);

    bool threw = false;
    try {
        (void) assembleConnectionGradient(m, 0);
    } catch (const nxr::core::Error& e) {
        threw = (e.code() == nxr::core::ErrorCode::InvalidInput);
    }
    std::cout << "  Cell C nSym=0 throws InvalidInput: " << (threw ? "yes" : "no") << "\n";
    CHECK(threw, "nSym=0 throws Error(InvalidInput)");
}

// ── Cell A2: facet accessor — same matrix as free function, cached per nSym ──
static void test_cellA_facet_accessor() {
    std::vector<double>   V;
    std::vector<int32_t>  F;
    generateIcosphere(V, F);

    nxr::manifold::Manifold m(V.data(), static_cast<int>(V.size()) / 3,
                              F.data(), static_cast<int>(F.size()) / 3);

    const auto& G1 = m.operators().connectionGradient(1);
    CHECK(G1.rows() == m.nE() && G1.cols() == m.nV(),
          "connectionGradient(1) shape [E×V]");
    CHECK(m.isOperatorCached(nxr::manifold::OperatorId::ConnectionGradient),
          "cached after access");

    // Second call must return the same object (pointer identity).
    const auto& G1b = m.operators().connectionGradient(1);
    CHECK(&G1 == &G1b, "same cached ref for same nSym");

    // Release + check no longer cached.
    m.releaseOperator(nxr::manifold::OperatorId::ConnectionGradient);
    CHECK(!m.isOperatorCached(nxr::manifold::OperatorId::ConnectionGradient),
          "released");

    // nSym=2 must produce a different (cached) matrix.
    const auto& G2 = m.operators().connectionGradient(2);
    CHECK(G2.rows() == m.nE() && G2.cols() == m.nV(),
          "connectionGradient(2) shape [E×V]");
    CHECK(m.isOperatorCached(nxr::manifold::OperatorId::ConnectionGradient),
          "cached after nSym=2 access");
    std::cout << "  Cell A2 facet accessor: OK (nV=" << m.nV()
              << " nE=" << m.nE() << ")\n";
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "── Cell A: squares to connection Laplacian ──\n";
    test_cellA_squares_to();

    std::cout << "── Cell A2: facet accessor + cache ──\n";
    test_cellA_facet_accessor();

    std::cout << "── Cell B: shape ──\n";
    test_cellB_shape();

    std::cout << "── Cell C: invalid nSym ──\n";
    test_cellC_invalid_nsym();

    std::cout << "── Cell D: extrinsicWeitzenbock (ambient vector part of squared Dirac) ──\n";
    test_cellD_extrinsic_weitzenbock();

    std::cout << (g_failures
                  ? "VECTOR COVARIANT TESTS FAILED\n"
                  : "ALL VECTOR COVARIANT TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
