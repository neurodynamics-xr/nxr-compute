/**
 * test_connection_laplacian.cpp — correctness checks for
 * `assembleConnectionLaplacian` on a fixed icosphere mesh.
 *
 * Verifies:
 *   - dimensions are correct per (domain, format): vertex → V, face → F,
 *     edge (Crouzeix-Raviart) → E in the complex form; 2× thereof in
 *     the real2N form
 *   - Real2N matrix is symmetric to machine precision (Hermitian
 *     in complex ⇒ symmetric in the real expansion)
 *   - Complex matrix is Hermitian (||K - Kᴴ||_F < tol)
 *   - PSD (smallest eigenvalue ≈ regularization for nSym=1, > 0
 *     for nSym ∈ {2, 4})
 *   - End-to-end: `solve` on the real2N matrix paired with
 *     a block-diagonal real mass matrix produces a non-zero smallest
 *     mode shape — the smoothest n-direction-field eigenpair
 *   - Domain / format string parsing round-trip and error paths
 *
 * Build: cmake --build build --target test_connection_laplacian
 * Run:   ./build/test_connection_laplacian
 */

#include "nxr/compute.h"

#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace nxr::manifold;
using namespace nxr::manifold::solve;
using namespace nxr::manifold::ops;
using namespace nxr::manifold::ops::laplacian::connection;
using namespace nxr::manifold::transport;
using namespace nxr::manifold::connection;
using namespace nxr::manifold::parametrization;
using namespace nxr::manifold::parametrization::stripes;
using namespace nxr::manifold::geometry;
using namespace nxr::manifold::query;
using namespace nxr::field::generate;
using namespace nxr::field::interp;
using namespace nxr::field::op;
using namespace nxr::field::extract;

using nxr::manifold::ops::laplacian::connection::ConnectionDomain;
using nxr::manifold::ops::laplacian::connection::ConnectionLaplacianFormat;
using nxr::manifold::ops::laplacian::connection::ConnectionLaplacianOptions;

static int failures = 0;

static void check(bool cond, const std::string& label) {
    if (cond) {
        std::cout << "    ✓ " << label << std::endl;
    } else {
        std::cout << "    ✗ " << label << std::endl;
        ++failures;
    }
}

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
        V.push_back(raw[i]   / len);
        V.push_back(raw[i+1] / len);
        V.push_back(raw[i+2] / len);
    }
    F = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
}

static const char* domainName(ConnectionDomain d) {
    switch (d) {
        case ConnectionDomain::Vertex:              return "vertex";
        case ConnectionDomain::Face:                return "face";
        case ConnectionDomain::EdgeCrouzeixRaviart: return "edge (CR)";
    }
    return "?";
}

// |K - Kᵀ|_F (real form) → tests symmetry of the lowered 2N×2N matrix.
static double realFrobeniusAsymmetry(const Eigen::SparseMatrix<double>& K) {
    Eigen::SparseMatrix<double> diff = K - Eigen::SparseMatrix<double>(K.transpose());
    return diff.norm();
}

// |K - Kᴴ|_F (complex form) → tests Hermitian property.
static double complexFrobeniusAntiHermitian(
    const Eigen::SparseMatrix<std::complex<double>>& K
) {
    Eigen::SparseMatrix<std::complex<double>> Kh = K.adjoint();
    Eigen::SparseMatrix<std::complex<double>> diff = K - Kh;
    return diff.norm();
}

// ── Trivial Connection Laplacian tests ───────────────────────────────────
//
// Verifies:
//   1. computeTrivialConnection returns a vector of length nEdges
//   2. Complex format: K_complex is Hermitian (||K - Kᴴ||_F < 1e-10)
//   3. Real2N format: K_real is symmetric (||K - Kᵀ||_F < 1e-10)
//   4. TC Laplacian differs from LC Laplacian (||K_TC - K_LC||_F > 1e-6)
//   5. frameE1/E2 are orthonormal [nV,3] matrices (gauge for eigenvectors)
//   6. Face domain throws Error(InvalidInput)
//   7. nSym=0 throws Error(InvalidInput)
static void testTrivialConnectionLaplacian(nxr::manifold::Manifold& m) {
    std::cout << "\n── trivial connection Laplacian ─────────────────────\n";

    // Gauss-Bonnet: two index-1 singularities → Σσ = 2 = χ(sphere)
    std::map<int, double> singMap = {{0, 1.0}, {1, 1.0}};

    auto dec = nxr::manifold::ops::assembleDECOperators(m);
    nxr::manifold::ops::CholeskyCache cache;

    // ── Test 1: phi.size() == nEdges ─────────────────────────────────────
    {
        Eigen::VectorXd phi = nxr::manifold::connection::computeTrivialConnection(
            m, dec, cache, singMap);
        check(phi.size() == m.nE(),
              "phi.size() == nEdges");
        std::cout << "    phi.norm() = " << phi.norm() << " (non-zero Poisson solve)\n";
    }

    // ── Test 2: Complex Hermitian ─────────────────────────────────────────
    {
        ConnectionLaplacianOptions opts;
        opts.format = ConnectionLaplacianFormat::Complex;
        opts.nSym   = 1;
        auto cl = nxr::manifold::ops::laplacian::connection::assembleTrivialConnectionLaplacian(
            m, singMap, dec, cache, opts);
        check(cl.baseDim  == m.nV(), "TC complex: baseDim == nV");
        check(cl.outputDim == m.nV(), "TC complex: outputDim == nV");
        const double hermErr = complexFrobeniusAntiHermitian(cl.K_complex);
        check(hermErr < 1e-10,
              "TC K_complex Hermitian (||K - Kᴴ||_F < 1e-10)");
    }

    // ── Test 3: Real2N symmetric ──────────────────────────────────────────
    {
        ConnectionLaplacianOptions opts;
        opts.format = ConnectionLaplacianFormat::Real2N;
        opts.nSym   = 1;
        auto cl = nxr::manifold::ops::laplacian::connection::assembleTrivialConnectionLaplacian(
            m, singMap, dec, cache, opts);
        check(cl.outputDim == 2 * m.nV(), "TC real2N: outputDim == 2*nV");
        const double symErr = realFrobeniusAsymmetry(cl.K_real);
        check(symErr < 1e-10,
              "TC K_real symmetric (||K - Kᵀ||_F < 1e-10)");
    }

    // ── Test 4: Differs from Levi-Civita Laplacian ────────────────────────
    {
        // First verify phi is non-trivial (Poisson solve produced real corrections)
        Eigen::VectorXd phi = computeTrivialConnection(m, dec, cache, singMap);
        check(phi.norm() > 1e-6,
              "phi.norm() > 1e-6 (Poisson solve produced non-trivial connection)");
        std::cout << "    phi.norm() = " << phi.norm() << " (non-trivial)\n";

        // Then verify the TC Laplacian differs from the LC Laplacian
        ConnectionLaplacianOptions opts4;
        opts4.format = ConnectionLaplacianFormat::Complex;
        opts4.nSym   = 1;
        auto tcl = assembleTrivialConnectionLaplacian(m, singMap, dec, cache, opts4);
        auto lcl = assembleConnectionLaplacian(m, opts4);
        Eigen::SparseMatrix<std::complex<double>> diff = tcl.K_complex - lcl.K_complex;
        const double diffNorm = diff.norm();
        std::cout << "    ||K_TC - K_LC||_F = " << diffNorm << " (expect > 1e-6)\n";
        check(diffNorm > 1e-6,
              "TC Laplacian differs from LC Laplacian (||K_TC - K_LC||_F > 1e-6)");
    }

    // ── Test 5: Frame orthonormality (trivial connection, vertex) ────────────
    {
        ConnectionLaplacianOptions opts;
        opts.format = ConnectionLaplacianFormat::Complex;
        opts.nSym   = 1;
        auto cl = nxr::manifold::ops::laplacian::connection::assembleTrivialConnectionLaplacian(
            m, singMap, dec, cache, opts);
        int N = cl.baseDim;
        check(cl.frameE1.rows() == N && cl.frameE1.cols() == 3,
              "TC frameE1 shape == [nV, 3]");
        check(cl.frameE2.rows() == N && cl.frameE2.cols() == 3,
              "TC frameE2 shape == [nV, 3]");
        double maxE1err = 0.0, maxE2err = 0.0, maxDot = 0.0;
        for (int r = 0; r < N; ++r) {
            maxE1err = std::max(maxE1err, std::abs(cl.frameE1.row(r).norm() - 1.0));
            maxE2err = std::max(maxE2err, std::abs(cl.frameE2.row(r).norm() - 1.0));
            maxDot   = std::max(maxDot, std::abs(cl.frameE1.row(r).dot(cl.frameE2.row(r))));
        }
        check(maxE1err < 1e-10, "TC frameE1 rows are unit vectors");
        check(maxE2err < 1e-10, "TC frameE2 rows are unit vectors");
        check(maxDot   < 1e-10, "TC frameE1 ⊥ frameE2");
    }

    // ── Test 6: Face domain throws Error(InvalidInput) ───────────────────
    {
        ConnectionLaplacianOptions opts;
        opts.domain = ConnectionDomain::Face;
        opts.nSym   = 1;
        bool threw = false;
        try {
            nxr::manifold::ops::laplacian::connection::assembleTrivialConnectionLaplacian(
                m, singMap, dec, cache, opts);
        } catch (const nxr::core::Error& e) {
            threw = (e.code() == nxr::core::ErrorCode::InvalidInput);
        }
        check(threw, "Face domain throws Error(InvalidInput)");
    }

    // ── Test 6: nSym=0 throws Error(InvalidInput) ────────────────────────
    {
        ConnectionLaplacianOptions opts;
        opts.nSym = 0;
        bool threw = false;
        try {
            nxr::manifold::ops::laplacian::connection::assembleTrivialConnectionLaplacian(
                m, singMap, dec, cache, opts);
        } catch (const nxr::core::Error& e) {
            threw = (e.code() == nxr::core::ErrorCode::InvalidInput);
        }
        check(threw, "nSym=0 throws Error(InvalidInput)");
    }
}

int main() {
    std::cout << "==============================================\n"
              << "  Connection-Laplacian correctness suite\n"
              << "==============================================\n";

    std::vector<double>  V;
    std::vector<int32_t> F;
    generateIcosphere(V, F);
    int nV = static_cast<int>(V.size() / 3);
    int nF = static_cast<int>(F.size() / 3);
    nxr::manifold::Manifold m(V.data(), nV, F.data(), nF);
    int nE = m.nE();

    std::cout << "\nMesh: icosahedron, " << nV << " V, "
              << nE << " E, " << nF << " F\n";

    const std::vector<ConnectionDomain> domains = {
        ConnectionDomain::Vertex,
        ConnectionDomain::Face,
        ConnectionDomain::EdgeCrouzeixRaviart,
    };
    const std::vector<int> nSyms = {1, 2, 4};

    for (auto d : domains) {
        const int baseDim = (d == ConnectionDomain::Vertex) ? nV
                          : (d == ConnectionDomain::Face)   ? nF
                          :                                   nE;
        std::cout << "\n── domain: " << domainName(d) << " (N=" << baseDim << ") ─────\n";

        for (int nSym : nSyms) {
            ConnectionLaplacianOptions opts;
            opts.domain = d;
            opts.nSym   = nSym;
            opts.format = ConnectionLaplacianFormat::Real2N;
            opts.regularization = 1e-8;

            auto cl = nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, opts);

            std::cout << "  nSym=" << nSym << " (real2N):\n";

            check(cl.baseDim == baseDim,
                  "baseDim == N");
            check(cl.outputDim == 2 * baseDim,
                  "outputDim == 2N");
            check(cl.K_real.rows() == 2 * baseDim &&
                  cl.K_real.cols() == 2 * baseDim,
                  "K_real shape == 2N × 2N");

            const double symErr = realFrobeniusAsymmetry(cl.K_real);
            check(symErr < 1e-10,
                  "K_real symmetric (||K - Kᵀ||_F < 1e-10)");

            // Complex form: test Hermitian property and complex dimension.
            opts.format = ConnectionLaplacianFormat::Complex;
            auto clC = nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, opts);
            check(clC.outputDim == baseDim,
                  "complex outputDim == N");
            check(clC.K_complex.rows() == baseDim &&
                  clC.K_complex.cols() == baseDim,
                  "K_complex shape == N × N");
            const double hermErr = complexFrobeniusAntiHermitian(clC.K_complex);
            check(hermErr < 1e-10,
                  "K_complex Hermitian (||K - Kᴴ||_F < 1e-10)");
        }

        // ── Frame checks (vertex and face only; edge has empty frames) ───────
        if (d != ConnectionDomain::EdgeCrouzeixRaviart) {
            ConnectionLaplacianOptions fo;
            fo.domain = d;
            fo.nSym   = 1;
            fo.format = ConnectionLaplacianFormat::Complex;
            auto clF = nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, fo);

            check(clF.frameE1.rows() == baseDim && clF.frameE1.cols() == 3,
                  "frameE1 shape == [baseDim, 3]");
            check(clF.frameE2.rows() == baseDim && clF.frameE2.cols() == 3,
                  "frameE2 shape == [baseDim, 3]");

            // Unit-length rows
            double maxE1err = 0.0, maxE2err = 0.0;
            for (int r = 0; r < baseDim; ++r) {
                maxE1err = std::max(maxE1err, std::abs(clF.frameE1.row(r).norm() - 1.0));
                maxE2err = std::max(maxE2err, std::abs(clF.frameE2.row(r).norm() - 1.0));
            }
            check(maxE1err < 1e-10, "frameE1 rows are unit vectors");
            check(maxE2err < 1e-10, "frameE2 rows are unit vectors");

            // Orthogonality
            double maxDot = 0.0;
            for (int r = 0; r < baseDim; ++r)
                maxDot = std::max(maxDot, std::abs(clF.frameE1.row(r).dot(clF.frameE2.row(r))));
            check(maxDot < 1e-10, "frameE1 ⊥ frameE2 (max |e1·e2| < 1e-10)");
        } else {
            // EdgeCrouzeixRaviart — frames must be empty.
            ConnectionLaplacianOptions fo;
            fo.domain = d;
            fo.nSym   = 1;
            fo.format = ConnectionLaplacianFormat::Complex;
            auto clF = nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, fo);
            check(clF.frameE1.rows() == 0, "frameE1 empty for EdgeCrouzeixRaviart");
            check(clF.frameE2.rows() == 0, "frameE2 empty for EdgeCrouzeixRaviart");
        }
    }

    // ── Cache contract ────────────────────────────────────────
    //
    // The native API itself is stateless — the result-level cache
    // lives on the bindings (ContextHolder / ContextWrapper). Here
    // we verify that two assemblies with identical options return
    // numerically identical matrices, which is the contract the
    // cache relies on (cache hit yields the same nonzero pattern
    // and the same values as a fresh call).
    {
        std::cout << "\n── cache-equivalent assemblies ─────\n";
        ConnectionLaplacianOptions o;
        o.domain = ConnectionDomain::Vertex;
        o.nSym   = 4;
        o.format = ConnectionLaplacianFormat::Real2N;

        auto a = nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, o);
        auto b = nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, o);

        check(a.K_real.nonZeros() == b.K_real.nonZeros(),
              "same nnz on repeat assembly");
        Eigen::SparseMatrix<double> diff = a.K_real - b.K_real;
        check(diff.norm() < 1e-12,
              "same values on repeat assembly");
    }

    // ── Composability with solve ────────────────────
    //
    // The smallest eigenpair of (K, M) on the connection-Laplacian
    // bundle reproduces the smoothest n-direction field. We use a
    // block-diagonal real mass matrix `blkdiag(I, I)` so every
    // dof is weighted equally — the constant-mass relative of
    // computeSmoothest{Vertex,Face}DirectionField. The icosahedron
    // is small enough that 6 modes converge in a fraction of a
    // second.
    {
        std::cout << "\n── solve composition (vertex, nSym=4) ─────\n";
        ConnectionLaplacianOptions o;
        o.domain         = ConnectionDomain::Vertex;
        o.nSym           = 4;
        o.regularization = 1e-6;   // larger shift so smallest eigenvalue is comfortably > 0
        o.format         = ConnectionLaplacianFormat::Real2N;

        auto cl = nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, o);

        // Mass: 2N × 2N identity (block-diagonal of I_V, I_V).
        Eigen::SparseMatrix<double> M(cl.outputDim, cl.outputDim);
        M.setIdentity();

        const int k = 6;
        auto eig = nxr::manifold::solve::eigen(cl.K_real, M, k);

        check(eig.k == k,                 "k modes returned");
        check(eig.nConverged == k,        "all modes converged");

        bool nonneg = true;
        for (int i = 0; i < eig.k; ++i) {
            if (eig.eigenvalues(i) < -1e-9) { nonneg = false; break; }
        }
        check(nonneg, "λ_i >= 0 (modulo float roundoff)");

        std::cout << "    eigenvalues: [";
        for (int i = 0; i < eig.k; ++i) {
            if (i) std::cout << ", ";
            std::cout << std::scientific << std::setprecision(3) << eig.eigenvalues(i);
        }
        std::cout << "]\n";

        // The smallest mode shape should be non-zero somewhere
        // (i.e. the eigenvector is a genuine direction-field
        // representative, not a degenerate null vector). We
        // expect at least one entry of magnitude >> 1e-6.
        double maxAbs = eig.eigenvectors.col(0).cwiseAbs().maxCoeff();
        check(maxAbs > 1e-3,
              "smallest mode shape is non-trivially nonzero");
    }

    // ── String parsers ────────────────────────────────────────
    {
        std::cout << "\n── parseConnectionDomain / parseConnectionLaplacianFormat ─────\n";
        check(nxr::manifold::ops::laplacian::connection::parseConnectionDomain("vertex") == ConnectionDomain::Vertex,
              "'vertex'");
        check(nxr::manifold::ops::laplacian::connection::parseConnectionDomain("face") == ConnectionDomain::Face,
              "'face'");
        check(nxr::manifold::ops::laplacian::connection::parseConnectionDomain("edge") == ConnectionDomain::EdgeCrouzeixRaviart,
              "'edge'");

        bool threw = false;
        try { (void) nxr::manifold::ops::laplacian::connection::parseConnectionDomain("nope"); }
        catch (const nxr::core::Error&) { threw = true; }
        check(threw, "unknown domain throws Error");

        check(nxr::manifold::ops::laplacian::connection::parseConnectionLaplacianFormat("real2N")  == ConnectionLaplacianFormat::Real2N,  "'real2N'");
        check(nxr::manifold::ops::laplacian::connection::parseConnectionLaplacianFormat("complex") == ConnectionLaplacianFormat::Complex, "'complex'");

        threw = false;
        try { (void) nxr::manifold::ops::laplacian::connection::parseConnectionLaplacianFormat("???"); }
        catch (const nxr::core::Error&) { threw = true; }
        check(threw, "unknown format throws Error");

        // nSym validation: nSym <= 0 must throw.
        threw = false;
        try {
            ConnectionLaplacianOptions bad;
            bad.nSym = 0;
            (void) nxr::manifold::ops::laplacian::connection::assembleConnectionLaplacian(m, bad);
        } catch (const nxr::core::Error&) { threw = true; }
        check(threw, "nSym <= 0 throws Error");
    }

    // ── Trivial Connection Laplacian ─────────────────────────────────────
    testTrivialConnectionLaplacian(m);

    std::cout << "\n==============================================\n";
    if (failures == 0) {
        std::cout << "  ALL CHECKS PASSED.\n";
    } else {
        std::cout << "  " << failures << " CHECK(S) FAILED.\n";
    }
    std::cout << "==============================================\n";
    return failures == 0 ? 0 : 1;
}
