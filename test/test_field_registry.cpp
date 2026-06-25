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

static void test_full_catalogue() {
    const char* ids[] = {"scalarVertex","oneFormEdge","twoFormFace","tangentVertex",
                         "tangentFace","ambientVertexWorld","ambientVertexLocal",
                         "ambientEdge","ambientFaceWorld","immersionVertex","immersionFace"};
    for (const char* id : ids) CHECK(fieldById(id) != nullptr, std::string("variant present: ") + id);

    const FieldVariant* tv = fieldById("tangentVertex");
    CHECK(tv && tv->descriptor.bundle == Bundle::tangent && tv->descriptor.field_type == FieldType::complex,
          "tangentVertex tangent+complex");
    CHECK(tv && tv->descriptor.representation == Representation::intrinsic_complex, "tangentVertex repr");
    const FieldVariant* aw = fieldById("ambientVertexWorld");
    CHECK(aw && aw->descriptor.representation == Representation::world, "ambientVertexWorld repr=world");
    const FieldVariant* al = fieldById("ambientVertexLocal");
    CHECK(al && al->descriptor.representation == Representation::local_frame, "ambientVertexLocal repr=local_frame");
    const FieldVariant* iv = fieldById("immersionVertex");
    CHECK(iv && iv->descriptor.bundle == Bundle::immersion && iv->descriptor.field_type == FieldType::quaternion,
          "immersionVertex immersion+quaternion");
    const FieldVariant* of = fieldById("oneFormEdge");
    CHECK(of && of->descriptor.domain == Domain::edge && of->descriptor.n_form == NForm::one, "oneFormEdge edge 1-form");
}

int main() {
    test_scalar_skeleton();
    test_full_catalogue();
    std::cout << (g_failures ? "FIELD REGISTRY TESTS FAILED\n" : "ALL FIELD REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
