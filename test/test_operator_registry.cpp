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

static void test_full_population() {
    const OperatorVariant* lc = operatorById("leviCivitaConnectionLaplacian");
    const OperatorVariant* tc = operatorById("trivialConnectionLaplacian");
    CHECK(lc && tc, "both connection-L variants present");
    if (lc && tc) {
        CHECK(lc->op_id == tc->op_id, "same OperatorId (LaplacianConnection)");
        CHECK(lc->holonomy == Holonomy::intrinsic_curved, "LC holonomy");
        CHECK(tc->holonomy == Holonomy::flat,             "trivial holonomy");
        CHECK(tc->singular == Singular::chi_defects,      "trivial chi-defects");
        CHECK(lc->gauge == Gauge::levi_civita && tc->gauge == Gauge::trivial, "gauges");
    }
    const OperatorVariant* fa = operatorById("flatCovariantLaplacian");
    const OperatorVariant* pr = operatorById("productCovariantLaplacian");
    CHECK(fa && pr && fa->op_id == pr->op_id, "both covariant variants, same OperatorId");
    if (fa && pr) {
        CHECK(fa->coupling == Coupling::ambient && fa->holonomy == Holonomy::flat, "flat covariant");
        CHECK(pr->coupling == Coupling::product && pr->holonomy == Holonomy::intrinsic_curved, "product covariant");
    }
    const OperatorVariant* rd = operatorById("relativeDirac");
    CHECK(rd && rd->graded && rd->holonomy == Holonomy::graded, "relativeDirac graded");
    CHECK(rd && rd->tau_presets.find("squared") != std::string::npos, "squared preset present");
    const OperatorVariant* gl = operatorById("graphLaplacian");
    CHECK(gl && gl->holonomy == Holonomy::combinatorial, "graph L combinatorial");
    const OperatorVariant* ew = operatorById("extrinsicWeitzenbockLaplacian");
    CHECK(ew && ew->status == Status::planned, "Weitzenboeck planned");
    CHECK(ew && ew->bundle == Bundle::ambient && ew->holonomy == Holonomy::extrinsic_curved, "Weitzenboeck facets");
}

int main() {
    test_scalar_skeleton();
    test_full_population();
    std::cout << (g_failures ? "REGISTRY TESTS FAILED\n" : "ALL REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
