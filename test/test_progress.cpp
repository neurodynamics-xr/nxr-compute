/**
 * test_progress.cpp — ProgressObserver contract for nxr-compute solvers.
 *
 * Validates that:
 *   1. Default-constructed observer is a no-op (no slots written).
 *   2. solve increments the iteration counter monotonically.
 *   3. totalIterations slot is set to a positive bound.
 *   4. Final iteration count >= 1 (some Spectra perform_op calls fired).
 *
 * Build: cmake --build build --target test_progress
 * Run:   ./build/Release/test_progress.exe
 */

#include "nxr/compute.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct Mesh {
    std::vector<double>  vertices;
    std::vector<int32_t> faces;
};

Mesh makeIcosahedron() {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    Mesh mesh;
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        mesh.vertices.push_back(raw[i] / len);
        mesh.vertices.push_back(raw[i+1] / len);
        mesh.vertices.push_back(raw[i+2] / len);
    }
    mesh.faces = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
    return mesh;
}

} // namespace

static int testsRun = 0;
static int testsFailed = 0;

#define EXPECT(cond) do {                                       \
    testsRun++;                                                 \
    if (!(cond)) {                                              \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__    \
                  << "]: " << #cond << std::endl;               \
        testsFailed++;                                          \
    }                                                           \
} while (0)

void test_DefaultObserverIsNoOp() {
    std::cout << "[1] Default-constructed observer is no-op" << std::endl;
    nxr::core::ProgressObserver obs;  // all slots null
    obs.update(42, 100, 1e-3);  // must not crash
    EXPECT(obs.iteration == nullptr);
    EXPECT(obs.totalIterations == nullptr);
    EXPECT(obs.residualMicro == nullptr);
}

void test_ObserverWritesAtomicSlots() {
    std::cout << "[2] update() writes the wired atomic slots" << std::endl;
    std::atomic<int32_t> iter{0}, total{0}, residual{0};
    nxr::core::ProgressObserver obs;
    obs.iteration = &iter;
    obs.totalIterations = &total;
    obs.residualMicro = &residual;
    obs.update(7, 100, 0.000123);
    EXPECT(iter.load() == 7);
    EXPECT(total.load() == 100);
    EXPECT(residual.load() == 123);  // 0.000123 × 1e6 = 123
}

void test_ObserverPartialSlots() {
    std::cout << "[3] Partial slot wiring writes only set slots" << std::endl;
    std::atomic<int32_t> iter{0};
    nxr::core::ProgressObserver obs;
    obs.iteration = &iter;
    // totalIterations and residualMicro are intentionally null
    obs.update(11, 50, 0.5);
    EXPECT(iter.load() == 11);
}

void test_SolveEigenmodesUpdatesProgress() {
    std::cout << "[4] solve ticks the iteration counter" << std::endl;
    auto mesh = makeIcosahedron();
    int nV = static_cast<int>(mesh.vertices.size() / 3);
    int nF = static_cast<int>(mesh.faces.size() / 3);

    nxr::manifold::Manifold m(mesh.vertices.data(), nV, mesh.faces.data(), nF);
    auto ops = nxr::manifold::ops::assembleManifoldOperators(m);

    std::atomic<int32_t> iter{0}, total{0}, residual{0};
    nxr::core::ProgressObserver obs;
    obs.iteration = &iter;
    obs.totalIterations = &total;
    obs.residualMicro = &residual;

    auto result = nxr::manifold::solve::eigen(ops.cotanLaplacian, ops.mass, 6, -1e-8,
                                       /*normalize=*/false, /*removeDC=*/false,
                                       nxr::core::CancellationToken{}, obs);

    EXPECT(result.k > 0);
    EXPECT(iter.load() > 0);          // at least one perform_op fired
    EXPECT(total.load() > 0);         // high-water bound set
    EXPECT(iter.load() <= total.load()); // never exceed the bound
    std::cout << "    final iteration counter: " << iter.load()
              << " / " << total.load() << std::endl;
}

void test_ResidualSaturation() {
    std::cout << "[5] residualMicro saturates int32 range" << std::endl;
    std::atomic<int32_t> residual{0};
    nxr::core::ProgressObserver obs;
    obs.residualMicro = &residual;

    obs.update(0, 0, 1e10);  // 1e10 * 1e6 = 1e16, way past INT32_MAX
    EXPECT(residual.load() == 2147483647);

    obs.update(0, 0, -1e10);
    EXPECT(residual.load() == -2147483648);

    obs.update(0, 0, 0.0);
    EXPECT(residual.load() == 0);
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  nxr-compute progress observer contract" << std::endl;
    std::cout << "==========================================" << std::endl;

    test_DefaultObserverIsNoOp();
    test_ObserverWritesAtomicSlots();
    test_ObserverPartialSlots();
    test_SolveEigenmodesUpdatesProgress();
    test_ResidualSaturation();

    std::cout << "\n==========================================" << std::endl;
    std::cout << "  " << (testsRun - testsFailed) << "/" << testsRun << " tests passed";
    if (testsFailed > 0) std::cout << " — " << testsFailed << " FAILURES";
    std::cout << std::endl;
    std::cout << "==========================================" << std::endl;

    return testsFailed == 0 ? 0 : 1;
}
