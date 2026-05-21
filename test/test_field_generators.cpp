/**
 * test_field_generators.cpp — validate the field generator library
 * and the Hodge decomposition contract.
 *
 * The headline test (Section 4) is the geometry-processing-js demo
 * pattern: synthesize ω from independent α and β contributions, run
 * hodge, verify the algorithm recovers what was put in.
 * This is the first piece of correctness validation we have for
 * hodge itself — until now the test exercised it without
 * checking the answer.
 *
 * Tolerance note: hodge uses regularized factors
 * (CholeskyCache::kRegularization = 1e-8). The recovered components
 * differ from truth by O(ε / λ_1) in the eigenbasis of the
 * vertex Laplacian A = d0ᵀ★₁d0. For the icosahedron λ_1 = 2, so
 * the recovery tolerance is ~5e-9; we use 1e-7 as a safety margin.
 * Real cortical meshes have smaller λ_1 and would need a looser
 * tolerance (~1e-4), but on this fixture 1e-7 is correct.
 */

#include "nxr/compute.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << #cond << ")" << std::endl; \
        std::exit(1); \
    } \
} while (0)

// Same icosahedron as the other tests — small, symmetric, predictable.
static void buildIcosahedron(std::vector<double>& V, std::vector<int32_t>& F) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        V.push_back(raw[i] / len);
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

int main() {
    std::cout << "[test_field_generators] starting" << std::endl;

    std::vector<double> V;
    std::vector<int32_t> F;
    buildIcosahedron(V, F);
    int nV = static_cast<int>(V.size() / 3);
    int nF = static_cast<int>(F.size() / 3);

    Manifold m(V.data(), nV, F.data(), nF);
    auto ops = assembleManifoldOperators(m);
    auto dec = assembleDECOperators(m);
    int nE = m.nE();

    // ── 1. Delta ─────────────────────────────────────────────
    {
        std::map<int, double> sources = {{0, 1.0}, {3, -1.0}};
        auto d = delta(nV, sources);
        REQUIRE(d.size() == nV, "delta has correct size");
        REQUIRE(d(0) == 1.0, "delta source value");
        REQUIRE(d(3) == -1.0, "delta sink value");
        REQUIRE(d(1) == 0.0 && d(2) == 0.0, "delta zero elsewhere");
        REQUIRE(d.norm() == std::sqrt(2.0), "delta norm");
        std::cout << "  [1] delta ✓" << std::endl;
    }

    // ── 2. Random scalars: bounds + reproducibility ──────────
    {
        auto a1 = randomVertexScalar(nV, 7);
        auto a2 = randomVertexScalar(nV, 7);
        auto a3 = randomVertexScalar(nV, 8);
        REQUIRE((a1 - a2).cwiseAbs().maxCoeff() == 0.0,
                "same seed → identical output");
        REQUIRE((a1 - a3).norm() > 0.1,
                "different seed → different output");
        REQUIRE(a1.cwiseAbs().maxCoeff() <= 1.0, "vertex scalar in [-1, 1]");

        auto b1 = randomFaceScalar(nF, 7);
        REQUIRE(b1.size() == nF, "face scalar correct size");
        REQUIRE(b1.cwiseAbs().maxCoeff() <= 1.0, "face scalar in [-1, 1]");

        std::cout << "  [2] random scalars: bounds + reproducibility ✓" << std::endl;
    }

    // ── 3. Eigenmode field: extract column k ─────────────────
    {
        // Solve a few eigenmodes so we have something to extract from.
        auto eig = solve::eigen(ops.cotanLaplacian, ops.mass, 6);
        auto phi3 = eigenmodeField(eig, 3);
        REQUIRE(phi3.size() == nV, "eigenmode field has nV entries");
        REQUIRE((phi3 - eig.eigenvectors.col(3)).cwiseAbs().maxCoeff() == 0.0,
                "eigenmode field equals column k");

        // Out-of-range should throw Error(InvalidInput)
        bool threw = false;
        ErrorCode caughtCode = ErrorCode::InternalError;
        try { (void)eigenmodeField(eig, 999); }
        catch (const Error& e) { threw = true; caughtCode = e.code(); }
        REQUIRE(threw, "eigenmode index out of range throws");
        REQUIRE(caughtCode == ErrorCode::InvalidInput,
                "out-of-range error has InvalidInput code");

        std::cout << "  [3] eigenmodeField ✓" << std::endl;
    }

    // ── 4. Hodge validation — the headline test ──────────────
    //
    // Synthesize ω = a·dα + b·δβ from independent random α and β.
    // The contract (header doc) is that the generator uses seed for α
    // and seed+1 for β, so we can independently reconstruct the
    // expected dα and δβ.
    //
    // Then run hodge and verify recovery within tolerance.
    {
        const unsigned seed = 31;
        const double aStrength = 1.5;
        const double bStrength = 0.7;

        auto omega = randomDecomposed1Form(
            dec, nV, nE, nF, aStrength, bStrength, 0.0, seed);

        // Independently reconstruct what the generator built —
        // mirrors the generator's internal contract.
        auto alphaRand = randomVertexScalar(nV, seed);
        auto betaRand  = randomFaceScalar(nF, seed + 1);
        Eigen::VectorXd dAlphaExpected   = aStrength * (dec.d0 * alphaRand);
        Eigen::VectorXd deltaBetaExpected = bStrength *
            dec.hodge1Inverse * (dec.d1.transpose() * betaRand);
        Eigen::VectorXd omegaExpected = dAlphaExpected + deltaBetaExpected;

        // 4a. Generator agrees with its own contract.
        double genErr = (omega - omegaExpected).cwiseAbs().maxCoeff();
        REQUIRE(genErr < 1e-14, "generator output matches contract");
        std::cout << "  [4a] generator output matches contract (max|diff|="
                  << genErr << ") ✓" << std::endl;

        // 4b. Run hodge.
        CholeskyCache cache;
        auto result = hodge(m, dec, cache, omega);

        // 4c. Decomposition identity: ω == dα + δβ + γ algebraically.
        Eigen::VectorXd reconstructed =
            result.dAlpha + result.deltaBeta + result.gamma;
        double identityErr = (omega - reconstructed).cwiseAbs().maxCoeff();
        REQUIRE(identityErr < 1e-12,
                "Hodge identity ω = dα + δβ + γ holds");
        std::cout << "  [4c] Hodge identity holds (max|diff|="
                  << identityErr << ") ✓" << std::endl;

        // 4d. Recovered exact part matches expected exact part.
        // Tolerance set by O(ε / λ_1); see header note.
        double dAlphaErr = (result.dAlpha - dAlphaExpected).cwiseAbs().maxCoeff();
        REQUIRE(dAlphaErr < 1e-7,
                "dα recovery within regularization-bound tolerance");
        std::cout << "  [4d] dα recovered (max|diff|=" << dAlphaErr
                  << ") ✓" << std::endl;

        // 4e. Recovered co-exact part matches expected co-exact part.
        double dBetaErr = (result.deltaBeta - deltaBetaExpected).cwiseAbs().maxCoeff();
        REQUIRE(dBetaErr < 1e-7,
                "δβ recovery within regularization-bound tolerance");
        std::cout << "  [4e] δβ recovered (max|diff|=" << dBetaErr
                  << ") ✓" << std::endl;

        // 4f. Harmonic residual is small (we synthesized with γ = 0;
        // anything that shows up here is solver / regularization noise).
        // Looser bound — γ catches both ε bias and any residuals.
        double gammaResid = result.gamma.cwiseAbs().maxCoeff();
        REQUIRE(gammaResid < 1e-7,
                "γ residual small (input had no harmonic component)");
        std::cout << "  [4f] γ residual small (max|γ|=" << gammaResid
                  << ") ✓" << std::endl;
    }

    // ── 5. Different strengths give different ω ──────────────
    // Sanity: cache isn't returning a fixed answer.
    {
        auto omega1 = randomDecomposed1Form(dec, nV, nE, nF, 1.0, 0.0, 0.0, 7);
        auto omega2 = randomDecomposed1Form(dec, nV, nE, nF, 0.0, 1.0, 0.0, 7);
        REQUIRE((omega1 - omega2).norm() > 0.1,
                "pure-α and pure-β fields differ");
        std::cout << "  [5] α-only vs β-only fields differ ✓" << std::endl;
    }

    // ── Solve eigenmodes for the time-varying tests ──────────
    // Pipeline: solve → M-orthonormalize → removeDC. Same as the addon.
    auto eigRaw = solve::eigen(ops.cotanLaplacian, ops.mass, 6);
    eigRaw.eigenvectors = solve::normalize(eigRaw.eigenvectors, ops.mass);
    auto eig = removeDC(eigRaw);
    int K = eig.k;
    REQUIRE(K == 5, "icosahedron yields 5 non-DC modes from k=6");

    // ── 6. Heat diffusion: shape, init, eigenmode pure decay ─
    {
        // Initial condition is a single eigenmode → expected evolution
        // is exact spectral decay: u(t) = exp(-α t λ_k) · φ_k.
        Eigen::VectorXd u0 = eigenmodeField(eig, 0);
        std::vector<double> ts = {0.0, 0.1, 0.5, 2.0};
        double alpha = 1.0;
        auto heat = heatDiffusion(ops, eig, u0, ts, alpha);

        REQUIRE(heat.rows() == 4 && heat.cols() == nV,
                "heat output shape [T, nV]");

        double lambda0 = eig.eigenvalues(0);
        for (size_t ti = 0; ti < ts.size(); ti++) {
            double expectedAtten = std::exp(-alpha * ts[ti] * lambda0);
            Eigen::VectorXd expected = expectedAtten * u0;
            Eigen::VectorXd uActual = heat.row(ti).cast<double>();
            double err = (uActual - expected).cwiseAbs().maxCoeff();
            REQUIRE(err < 1e-6,
                    "heat: u(t) = exp(-αtλ_k) φ_k for single-mode IC");
        }
        std::cout << "  [6] heat: single-mode IC → pure exponential decay ✓"
                  << std::endl;
    }

    // ── 7. Heat: mass-weighted mean preservation ─────────────
    // Heat equation conserves the spatial integral; we encode this
    // by adding u_mean to every frame outside the spectral sum.
    {
        // Non-zero-mean initial condition: a single positive delta.
        std::map<int, double> sources = {{0, 1.0}};
        Eigen::VectorXd u0 = delta(nV, sources);
        std::vector<double> ts = {0.0, 0.1, 1.0, 10.0, 100.0};
        auto heat = heatDiffusion(ops, eig, u0, ts, 1.0);

        double initialMean = (ops.mass * u0).sum() / ops.totalArea;
        for (size_t ti = 0; ti < ts.size(); ti++) {
            Eigen::VectorXd ut = heat.row(ti).cast<double>();
            double meanT = (ops.mass * ut).sum() / ops.totalArea;
            REQUIRE(std::abs(meanT - initialMean) < 1e-6,
                    "heat: mass-weighted mean preserved across all t");
        }

        // Asymptotic behaviour: at very large t, all non-zero modes
        // have decayed and u → uniform mean field.
        Eigen::VectorXd uInf = heat.row(ts.size() - 1).cast<double>();
        double spread = uInf.maxCoeff() - uInf.minCoeff();
        REQUIRE(spread < 1e-6,
                "heat: u(∞) → uniform mean field");
        std::cout << "  [7] heat: mean preserved, asymptotes to uniform ✓"
                  << std::endl;
    }

    // ── 8. Damped wave: shape, single-mode evolution ─────────
    {
        std::vector<double> ts = {0.0, 0.5, 1.0, 2.0};
        double amp = 1.5, gamma = 0.3, phase = 0.0;
        auto wave = dampedWave(eig,
            {0}, {amp}, {gamma}, {phase}, ts);

        REQUIRE(wave.rows() == 4 && wave.cols() == nV,
                "wave output shape [T, nV]");

        double lambda0 = eig.eigenvalues(0);
        double omega0 = std::sqrt(lambda0);
        Eigen::VectorXd phi0 = eigenmodeField(eig, 0);

        for (size_t ti = 0; ti < ts.size(); ti++) {
            double t = ts[ti];
            double scalar = amp * std::exp(-gamma * t) * std::cos(omega0 * t + phase);
            Eigen::VectorXd expected = scalar * phi0;
            Eigen::VectorXd uActual = wave.row(ti).cast<double>();
            double err = (uActual - expected).cwiseAbs().maxCoeff();
            REQUIRE(err < 1e-5,
                    "wave: single-mode evolution matches analytical formula");
        }
        std::cout << "  [8] wave: single-mode → a·exp(-γt)·cos(ωt+φ)·φ_k ✓"
                  << std::endl;
    }

    // ── 9. Damped wave: multi-mode superposition at t = 0 ────
    {
        // At t=0 with phases=0, output should be the literal
        // weighted superposition of the chosen eigenmodes.
        auto wave = dampedWave(eig,
            {0, 2}, {1.0, 0.5}, {0.0, 0.0}, {0.0, 0.0},
            {0.0});
        Eigen::VectorXd expected =
            1.0 * eigenmodeField(eig, 0)
          + 0.5 * eigenmodeField(eig, 2);
        Eigen::VectorXd uActual = wave.row(0).cast<double>();
        double err = (uActual - expected).cwiseAbs().maxCoeff();
        REQUIRE(err < 1e-6, "wave: multi-mode superposition at t=0");
        std::cout << "  [9] wave: multi-mode superposition ✓" << std::endl;
    }

    // ── 10. Damped wave: zero damping → constant amplitude ───
    {
        // With γ = 0, the per-mode temporal envelope is a pure
        // cosine; amplitude over a full period should match the
        // initial amplitude. Use a period of T = 2π/ω.
        double lambda0 = eig.eigenvalues(0);
        double period = 2.0 * M_PI / std::sqrt(lambda0);
        std::vector<double> ts = {0.0, period};
        auto wave = dampedWave(eig,
            {0}, {1.0}, {0.0}, {0.0}, ts);

        Eigen::VectorXd u0 = wave.row(0).cast<double>();
        Eigen::VectorXd uPeriod = wave.row(1).cast<double>();
        double err = (u0 - uPeriod).cwiseAbs().maxCoeff();
        REQUIRE(err < 1e-5,
                "wave: undamped, one period later → same amplitude");
        std::cout << "  [10] wave: undamped period preserves amplitude ✓"
                  << std::endl;
    }

    std::cout << "[test_field_generators] all assertions passed ✓" << std::endl;
    return 0;
}
