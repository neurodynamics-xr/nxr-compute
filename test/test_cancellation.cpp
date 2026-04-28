/**
 * test_cancellation.cpp — Cancellation contract for cxf solvers.
 *
 * Validates that:
 *   1. CancellationToken::requested() reads the atomic flag.
 *   2. CancellationToken::requested() invokes the function poller.
 *   3. solveEigenmodes throws Error(Cancelled) when the flag is
 *      pre-set (cancel before factorization).
 *   4. solveEigenmodes throws Error(Cancelled) when the flag flips
 *      mid-solve (cancel during IRAM iteration).
 *   5. Default-constructed token is a no-op.
 *
 * Build: cmake --build build --target test_cancellation
 * Run:   ./build/Release/test_cancellation.exe
 */

#include "nxr/compute.h"

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <thread>
#include <utility>
#include <vector>

// ── Subdivided icosphere generator ──────────────────────────
//
// The 12-vertex icosahedron is too small to give the eigensolver
// enough work to cancel in a measurable interval. One level of
// subdivision (162 verts) is the smallest that produces a solve
// running long enough to interrupt reliably.
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
    Mesh m;
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        m.vertices.push_back(raw[i] / len);
        m.vertices.push_back(raw[i+1] / len);
        m.vertices.push_back(raw[i+2] / len);
    }
    m.faces = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
    return m;
}

// Replace each face with 4 by inserting midpoints, then renormalize
// each new vertex to unit length so the result stays on the sphere.
Mesh subdivide(const Mesh& in) {
    auto vertOf = [&](int i) {
        return std::array<double, 3>{
            in.vertices[3*i + 0], in.vertices[3*i + 1], in.vertices[3*i + 2]};
    };
    Mesh out;
    out.vertices = in.vertices;
    std::map<std::pair<int,int>, int> midpointCache;
    auto midpoint = [&](int a, int b) -> int {
        auto key = std::make_pair(std::min(a,b), std::max(a,b));
        auto it = midpointCache.find(key);
        if (it != midpointCache.end()) return it->second;
        auto va = vertOf(a), vb = vertOf(b);
        double mx = (va[0]+vb[0])*0.5, my = (va[1]+vb[1])*0.5, mz = (va[2]+vb[2])*0.5;
        double len = std::sqrt(mx*mx + my*my + mz*mz);
        out.vertices.push_back(mx / len);
        out.vertices.push_back(my / len);
        out.vertices.push_back(mz / len);
        int idx = static_cast<int>(out.vertices.size() / 3) - 1;
        midpointCache[key] = idx;
        return idx;
    };
    for (size_t f = 0; f < in.faces.size(); f += 3) {
        int a = in.faces[f], b = in.faces[f+1], c = in.faces[f+2];
        int ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        out.faces.insert(out.faces.end(), {a, ab, ca,  b, bc, ab,  c, ca, bc,  ab, bc, ca});
    }
    return out;
}

} // namespace

// ── Test cases ──────────────────────────────────────────────

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

void test_DefaultTokenIsNotCancelled() {
    std::cout << "[1] Default token is not cancelled" << std::endl;
    nxr::compute::CancellationToken tok;
    EXPECT(tok.requested() == false);
}

void test_AtomicFlagFlipsRequested() {
    std::cout << "[2] Atomic flag drives requested()" << std::endl;
    std::atomic<int32_t> flag{0};
    nxr::compute::CancellationToken tok(&flag);
    EXPECT(tok.requested() == false);
    flag.store(1, std::memory_order_relaxed);
    EXPECT(tok.requested() == true);
    flag.store(0, std::memory_order_relaxed);
    EXPECT(tok.requested() == false);
}

void test_FunctionPollerIsInvoked() {
    std::cout << "[3] Function poller is invoked" << std::endl;
    int callCount = 0;
    nxr::compute::CancellationToken tok([&callCount]() {
        callCount++;
        return callCount >= 3;
    });
    EXPECT(tok.requested() == false);  // call 1
    EXPECT(tok.requested() == false);  // call 2
    EXPECT(tok.requested() == true);   // call 3
    EXPECT(callCount == 3);
}

void test_SolveCancelledBeforeFactorization() {
    std::cout << "[4] solveEigenmodes throws Cancelled when flag pre-set" << std::endl;
    auto m = makeIcosahedron();
    int nV = static_cast<int>(m.vertices.size() / 3);
    int nF = static_cast<int>(m.faces.size() / 3);

    nxr::compute::ComputeContext ctx(m.vertices.data(), nV, m.faces.data(), nF);
    auto ops = nxr::compute::assembleMeshOperators(ctx);

    std::atomic<int32_t> flag{1};  // already cancelled
    nxr::compute::CancellationToken tok(&flag);

    bool threw = false;
    nxr::compute::ErrorCode code = nxr::compute::ErrorCode::InternalError;
    try {
        nxr::compute::solveEigenmodes(ops.stiffness, ops.mass, 6, -1e-8, tok);
    } catch (const nxr::compute::Error& e) {
        threw = true;
        code = e.code();
    }
    EXPECT(threw);
    EXPECT(code == nxr::compute::ErrorCode::Cancelled);
}

void test_SolveCancelledMidIteration() {
    std::cout << "[5] solveEigenmodes throws Cancelled when flag flips mid-solve" << std::endl;
    auto m = subdivide(makeIcosahedron());  // 162 verts, enough for measurable solve
    int nV = static_cast<int>(m.vertices.size() / 3);
    int nF = static_cast<int>(m.faces.size() / 3);

    nxr::compute::ComputeContext ctx(m.vertices.data(), nV, m.faces.data(), nF);
    auto ops = nxr::compute::assembleMeshOperators(ctx);

    // Function-based token: returns true after the first perform_op call,
    // so cancellation fires inside Spectra's iteration. This isolates the
    // test from threading races and is bit-deterministic.
    int callCount = 0;
    nxr::compute::CancellationToken tok([&callCount]() {
        // The eigensolver checks once before factorization; let that pass.
        // Then the wrapper checks once per perform_op; cancel on the
        // first such check.
        callCount++;
        return callCount > 1;
    });

    bool threw = false;
    nxr::compute::ErrorCode code = nxr::compute::ErrorCode::InternalError;
    try {
        nxr::compute::solveEigenmodes(ops.stiffness, ops.mass, 20, -1e-8, tok);
    } catch (const nxr::compute::Error& e) {
        threw = true;
        code = e.code();
    }
    EXPECT(threw);
    EXPECT(code == nxr::compute::ErrorCode::Cancelled);
    std::cout << "    poller calls before cancel: " << callCount << std::endl;
}

void test_SolveSucceedsWithIdleToken() {
    std::cout << "[6] solveEigenmodes succeeds when token is never set" << std::endl;
    auto m = makeIcosahedron();
    int nV = static_cast<int>(m.vertices.size() / 3);
    int nF = static_cast<int>(m.faces.size() / 3);

    nxr::compute::ComputeContext ctx(m.vertices.data(), nV, m.faces.data(), nF);
    auto ops = nxr::compute::assembleMeshOperators(ctx);

    std::atomic<int32_t> flag{0};
    nxr::compute::CancellationToken tok(&flag);

    bool ok = false;
    try {
        auto result = nxr::compute::solveEigenmodes(ops.stiffness, ops.mass, 6, -1e-8, tok);
        ok = (result.k > 0);
    } catch (const nxr::compute::Error&) {
        ok = false;
    }
    EXPECT(ok);
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  cxf cancellation contract" << std::endl;
    std::cout << "==========================================" << std::endl;

    test_DefaultTokenIsNotCancelled();
    test_AtomicFlagFlipsRequested();
    test_FunctionPollerIsInvoked();
    test_SolveCancelledBeforeFactorization();
    test_SolveCancelledMidIteration();
    test_SolveSucceedsWithIdleToken();

    std::cout << "\n==========================================" << std::endl;
    std::cout << "  " << (testsRun - testsFailed) << "/" << testsRun << " tests passed";
    if (testsFailed > 0) std::cout << " — " << testsFailed << " FAILURES";
    std::cout << std::endl;
    std::cout << "==========================================" << std::endl;

    return testsFailed == 0 ? 0 : 1;
}
