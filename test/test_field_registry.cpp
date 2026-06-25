/**
 * test_field_registry.cpp — Field type system: catalogue, operator coupling,
 * conversion graph, routing.
 * Build: cmake --build build --target test_field_registry
 * Run:   ./build/test_field_registry
 */
#include "nxr/field_registry.h"
#include "nxr/operator_registry.h"
#include "nxr/errors.h"
#include <iostream>
#include <string>

using namespace nxr::manifold;
using namespace nxr::manifold::registry;

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << (msg) << "\n"; ++g_failures; } } while (0)

static void test_scalar_skeleton() {
    const FieldVariant* sv = fieldById("scalarVertex");
    CHECK(sv != nullptr, "scalarVertex present");
    if (sv) {
        CHECK(sv->descriptor.domain == Domain::vertex, "scalarVertex domain=vertex");
        CHECK(sv->descriptor.bundle == Bundle::scalar, "scalarVertex bundle=scalar");
        CHECK(sv->descriptor.n_form == NForm::zero,    "scalarVertex n_form=zero");
    }
    CHECK(fieldById("doesNotExist") == nullptr, "unknown id -> nullptr");
}

int main() {
    test_scalar_skeleton();
    std::cout << (g_failures ? "FIELD REGISTRY TESTS FAILED\n" : "ALL FIELD REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
