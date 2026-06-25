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

static void test_operator_io_integrity() {
    for (const auto& op : operatorRegistry()) {
        CHECK(!op.input_field.empty(),  std::string("operator has input_field: ")  + op.id);
        CHECK(!op.output_field.empty(), std::string("operator has output_field: ") + op.id);
        CHECK(fieldById(op.input_field)  != nullptr, std::string("input_field resolves: ")  + op.id + " -> " + op.input_field);
        CHECK(fieldById(op.output_field) != nullptr, std::string("output_field resolves: ") + op.id + " -> " + op.output_field);
    }
    const OperatorVariant* f2 = operatorById("faceLaplacian2Form");
    const OperatorVariant* rf = operatorById("relativeFaceDirac");
    CHECK(f2 && f2->input_field == "twoFormFace",   "faceLaplacian2Form input twoFormFace");
    CHECK(rf && rf->input_field == "immersionFace", "relativeFaceDirac input immersionFace");
}

static void test_routing() {
    const FieldDescriptor& scalarV = fieldById("scalarVertex")->descriptor;
    const FieldDescriptor& tangV   = fieldById("tangentVertex")->descriptor;
    const FieldDescriptor& immV    = fieldById("immersionVertex")->descriptor;

    bool ok = true;
    try { requireField(scalarV, "laplaceBeltrami"); } catch (const nxr::core::Error&) { ok = false; }
    CHECK(ok, "scalarVertex accepted by laplaceBeltrami");

    bool threw = false;
    try { requireField(tangV, "laplaceBeltrami"); } catch (const nxr::core::Error&) { threw = true; }
    CHECK(threw, "tangentVertex rejected by laplaceBeltrami");

    threw = false;
    try { requireField(immV, "flatCovariantLaplacian"); } catch (const nxr::core::Error&) { threw = true; }
    CHECK(threw, "immersionVertex rejected by ambient operator");

    auto ops = operatorsAccepting(scalarV);
    bool hasLB = false, hasGL = false;
    for (auto& id : ops) { if (id == "laplaceBeltrami") hasLB = true; if (id == "graphLaplacian") hasGL = true; }
    CHECK(hasLB && hasGL, "operatorsAccepting(scalarVertex) includes laplaceBeltrami and graphLaplacian");

    CHECK(componentsPerElement(scalarV) == 1, "scalar components=1");
    CHECK(componentsPerElement(fieldById("ambientVertexWorld")->descriptor) == 3, "ambient components=3");
    CHECK(componentsPerElement(immV) == 4, "immersion components=4");

    bool shapeOk = true;
    try { validateFieldShape(scalarV, 12, /*nV*/12, /*nE*/30, /*nF*/20); } catch (const nxr::core::Error&) { shapeOk = false; }
    CHECK(shapeOk, "scalarVertex 12 rows valid on 12-vertex mesh");
    bool shapeThrew = false;
    try { validateFieldShape(scalarV, 11, 12, 30, 20); } catch (const nxr::core::Error&) { shapeThrew = true; }
    CHECK(shapeThrew, "scalarVertex wrong row count rejected");
}

static void test_conversion_graph() {
    const auto& edges = conversionGraph();
    CHECK(!edges.empty(), "conversion graph non-empty");
    for (const auto& e : edges) {
        CHECK(fieldById(e.from) != nullptr, std::string("conversion from resolves: ") + e.from);
        CHECK(fieldById(e.to)   != nullptr, std::string("conversion to resolves: ")   + e.to);
        if (e.implemented) CHECK(!e.impl.empty(), std::string("implemented edge names impl: ") + e.from + "->" + e.to);
    }
    bool hasLift = false;
    for (const auto& e : edges)
        if (e.from == "ambientVertexWorld" && e.to == "ambientVertexLocal" && e.implemented) hasLift = true;
    CHECK(hasLift, "world->local lift edge present and implemented");
}

int main() {
    test_scalar_skeleton();
    test_full_catalogue();
    test_operator_io_integrity();
    test_routing();
    test_conversion_graph();
    std::cout << (g_failures ? "FIELD REGISTRY TESTS FAILED\n" : "ALL FIELD REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
