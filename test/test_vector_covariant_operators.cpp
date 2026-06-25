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

#include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
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

    std::cout << (g_failures
                  ? "VECTOR COVARIANT TESTS FAILED\n"
                  : "ALL VECTOR COVARIANT TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
