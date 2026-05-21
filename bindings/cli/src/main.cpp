/**
 * nxr-compute — command-line smoke for the nxr-compute library.
 *
 * The library is binding-shell-agnostic: it reads typed arrays in
 * and returns matrices out. It does not parse mesh files. The CLI
 * is therefore intentionally minimal — it verifies the library
 * builds, links, and runs end-to-end against a synthetic fixture.
 *
 *   nxr_compute --version
 *   nxr_compute smoke      → eigensolve on a unit icosahedron,
 *                            prints the canonical λ spectrum
 *
 * Real pipelines that consume mesh files live in the host app
 * (cortical-flow, brainstorm, your code) and bring their own I/O.
 */

#include "nxr/compute.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

// Process-wide SIGINT flag → CancellationToken. Async-signal-safe.
std::atomic<int32_t> g_sigintFlag{0};

void onSigint(int /*sig*/) {
    g_sigintFlag.store(1, std::memory_order_relaxed);
}

nxr::core::CancellationToken ctrlCToken() {
    return nxr::core::CancellationToken(&g_sigintFlag);
}

constexpr const char* kVersion = "nxr-compute 0.1.0";

void printUsage() {
    std::cout
        << "Usage: nxr_compute <command>\n"
        << "\n"
        << "Commands:\n"
        << "  smoke     Eigensolve on a unit icosahedron and print λ spectrum.\n"
        << "            Exits 0 if the canonical spectrum matches.\n"
        << "\n"
        << "Flags:\n"
        << "  -h, --help      Print this help and exit.\n"
        << "  -v, --version   Print version and exit.\n";
}

double elapsedMs(std::chrono::steady_clock::time_point t0) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

int cmdSmoke() {
    // Unit icosahedron: 12 vertices, 20 faces. The combinatorial
    // Laplace–Beltrami spectrum on the regular icosahedron has a
    // triplet at λ ≈ 2 and a doublet at λ ≈ 4.34, which is what
    // we assert below. (Same fixture as test_eigen.cpp.)
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double rawVerts[36] = {
        -1,  t,  0,    1,  t,  0,   -1, -t,  0,    1, -t,  0,
         0, -1,  t,    0,  1,  t,    0, -1, -t,    0,  1, -t,
         t,  0, -1,    t,  0,  1,   -t,  0, -1,   -t,  0,  1,
    };
    double verts[36];
    for (int i = 0; i < 36; i += 3) {
        const double len = std::sqrt(rawVerts[i]   * rawVerts[i]
                                    + rawVerts[i+1] * rawVerts[i+1]
                                    + rawVerts[i+2] * rawVerts[i+2]);
        verts[i]   = rawVerts[i]   / len;
        verts[i+1] = rawVerts[i+1] / len;
        verts[i+2] = rawVerts[i+2] / len;
    }
    std::int32_t faces[60] = {
        0,11,5, 0,5,1,  0,1,7,  0,7,10, 0,10,11,
        1,5,9,  5,11,4, 11,10,2,10,7,6, 7,1,8,
        3,9,4,  3,4,2,  3,2,6,  3,6,8,  3,8,9,
        4,9,5,  2,4,11, 6,2,10, 8,6,7,  9,8,1,
    };

    auto t0 = std::chrono::steady_clock::now();
    nxr::manifold::Manifold m(verts, 12, faces, 20);
    auto ops = nxr::manifold::ops::assembleManifoldOperators(m);
    auto eig = nxr::manifold::solve::eigen(
        ops.cotanLaplacian, ops.mass, 6, -1e-8,
        /*normalize=*/true, /*removeDC=*/true, ctrlCToken());

    // Touch the new vector / signed heat / smooth field surfaces so the
    // smoke catches link-time and basic-execution regressions across all
    // four new families.
    nxr::manifold::transport::VectorHeatSolver vhm(m);
    Eigen::MatrixXd srcVec(1, 3); srcVec << 1.0, 0.0, 0.0;
    Eigen::MatrixXd transported = nxr::manifold::transport::parallel(vhm, {0}, srcVec);
    auto logmap = nxr::manifold::transport::logMap(vhm, 0);
    Eigen::Vector3d center = nxr::manifold::transport::findCenter(vhm, {0, 1, 2});

    nxr::manifold::solve::SignedHeatSolver shs(m);
    Eigen::VectorXd sd = nxr::manifold::solve::signedHeat(shs, {0, 1, 5}, true);

    Eigen::MatrixXd faceField = nxr::manifold::connection::smoothFace(m, 4);
    auto vfield = nxr::manifold::connection::smoothVertex(m, 2);
    auto stripes = nxr::manifold::parametrization::stripes::compute(m, vfield.vertexFieldRaw, 8.0);

    const double dt = elapsedMs(t0);

    std::cout << "[smoke] icosahedron precompute in " << dt << " ms\n";
    std::cout << "[smoke]   vhm transport vec[0]=" << transported.row(0)
              << "  logmap rows=" << logmap.logCoords.rows()
              << "  center=" << center.transpose() << "\n";
    std::cout << "[smoke]   signedHeat range=[" << sd.minCoeff() << ", " << sd.maxCoeff() << "]\n";
    std::cout << "[smoke]   smoothFace nF=" << faceField.rows()
              << "  stripes segs=" << stripes.segmentCount << "\n";
    std::cout << "[smoke]   nV=" << m.nV() << "  nF=" << m.nF()
              << "  totalArea=" << ops.totalArea << "\n";
    std::cout << "[smoke]   eigenvalues (k=" << eig.k << " post-removeDC):";
    for (int i = 0; i < eig.k; ++i) {
        std::cout << "  " << eig.eigenvalues(i);
    }
    std::cout << "\n";

    // Canonical icosahedron spectrum: triplet at λ≈2 (λ_0..λ_2),
    // doublet at λ≈4.34 (λ_3, λ_4). Tolerance generous for
    // floating-point + Spectra convergence noise.
    int fail = 0;
    for (int i = 0; i < 3 && i < eig.k; ++i) {
        if (std::abs(eig.eigenvalues(i) - 2.0) > 1e-5) {
            std::cerr << "[smoke] FAIL: λ_" << i << " expected ~2.0, got "
                      << eig.eigenvalues(i) << "\n";
            ++fail;
        }
    }
    if (eig.k > 3 && std::abs(eig.eigenvalues(3) - 4.34164) > 1e-3) {
        std::cerr << "[smoke] FAIL: λ_3 expected ~4.34, got "
                  << eig.eigenvalues(3) << "\n";
        ++fail;
    }
    if (fail == 0) {
        std::cout << "[smoke] canonical spectrum matched ✓\n";
    }
    return fail == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, onSigint);

    if (argc < 2) {
        printUsage();
        return 2;
    }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") {
        printUsage();
        return 0;
    }
    if (cmd == "-v" || cmd == "--version") {
        std::cout << kVersion << "\n";
        return 0;
    }
    if (cmd == "smoke") {
        try {
            return cmdSmoke();
        } catch (const nxr::core::Error& e) {
            std::cerr << "nxr_compute smoke: ["
                      << nxr::core::errorCodeName(e.code()) << "] "
                      << e.what() << "\n";
            return e.code() == nxr::core::ErrorCode::Cancelled ? 130 : 1;
        } catch (const std::exception& e) {
            std::cerr << "nxr_compute smoke: " << e.what() << "\n";
            return 1;
        }
    }

    std::cerr << "nxr_compute: unknown command \"" << cmd << "\"\n";
    printUsage();
    return 2;
}
