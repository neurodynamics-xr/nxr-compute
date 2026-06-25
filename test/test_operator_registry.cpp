/**
 * test_operator_registry.cpp — registry completeness + cross-link identities.
 * Build: cmake --build build --target test_operator_registry
 * Run:   ./build/test_operator_registry
 */
#include "nxr/operator_registry.h"
#include "nxr/compute.h"
#include <iostream>
#include <string>

using namespace nxr::manifold;
using namespace nxr::manifold::registry;

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << (msg) << "\n"; ++g_failures; } } while (0)

static void test_scalar_skeleton() {
    const OperatorVariant* lb = operatorById("laplaceBeltrami");
    CHECK(lb != nullptr, "laplaceBeltrami present");
    if (lb) {
        CHECK(lb->bundle == Bundle::scalar, "laplaceBeltrami bundle=scalar");
        CHECK(lb->order  == Order::second,  "laplaceBeltrami order=second");
        CHECK(lb->op_id  == OperatorId::LaplacianCotan, "laplaceBeltrami op_id");
    }
    CHECK(operatorById("doesNotExist") == nullptr, "unknown id → nullptr");
}

int main() {
    test_scalar_skeleton();
    std::cout << (g_failures ? "REGISTRY TESTS FAILED\n" : "ALL REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
