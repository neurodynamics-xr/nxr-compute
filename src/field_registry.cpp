#include "nxr/field_registry.h"
#include "nxr/errors.h"

namespace nxr::manifold::registry {

using nxr::core::Error;
using nxr::core::ErrorCode;

static FieldDescriptor desc(Domain d, Bundle b, FieldType ft, NForm nf,
                            Representation rep, Gauge g = Gauge::na, int nSym = 1) {
    return FieldDescriptor{ d, b, ft, nf, rep, g, nSym };
}

const std::vector<FieldVariant>& fieldRegistry() {
    static const std::vector<FieldVariant> table = {
        // scalar / DEC forms
        { "scalarVertex", "Scalar (0-form, vertex)",
          desc(Domain::vertex, Bundle::scalar, FieldType::real, NForm::zero, Representation::na),
          "vertex function / 0-form" },
        { "oneFormEdge", "1-form (edge)",
          desc(Domain::edge, Bundle::scalar, FieldType::real, NForm::one, Representation::na),
          "DEC 1-form" },
        { "twoFormFace", "2-form (face)",
          desc(Domain::face, Bundle::scalar, FieldType::real, NForm::two, Representation::na),
          "per-face scalar / 2-form" },
        // tangent
        { "tangentVertex", "Tangent field (vertex)",
          desc(Domain::vertex, Bundle::tangent, FieldType::complex, NForm::na, Representation::intrinsic_complex, Gauge::levi_civita),
          "n-RoSy via nSym" },
        { "tangentFace", "Tangent field (face)",
          desc(Domain::face, Bundle::tangent, FieldType::complex, NForm::na, Representation::intrinsic_complex, Gauge::levi_civita),
          "n-RoSy via nSym" },
        // ambient (R^3)
        { "ambientVertexWorld", "Ambient vector (vertex, world)",
          desc(Domain::vertex, Bundle::ambient, FieldType::real, NForm::na, Representation::world),
          "R^3 global Cartesian (e.g. leadfield)" },
        { "ambientVertexLocal", "Ambient vector (vertex, local frame)",
          desc(Domain::vertex, Bundle::ambient, FieldType::real, NForm::na, Representation::local_frame, Gauge::levi_civita),
          "R^3 in per-vertex frame [a;b;c]" },
        { "ambientEdge", "Ambient vector (edge, local frame)",
          desc(Domain::edge, Bundle::ambient, FieldType::real, NForm::na, Representation::local_frame, Gauge::levi_civita),
          "covariant-difference output (3E)" },
        { "ambientFaceWorld", "Ambient vector (face, world)",
          desc(Domain::face, Bundle::ambient, FieldType::real, NForm::na, Representation::world),
          "R^3 per face (gradient/whitney output)" },
        // immersion (quaternion)
        { "immersionVertex", "Immersion spinor (vertex)",
          desc(Domain::vertex, Bundle::immersion, FieldType::quaternion, NForm::na, Representation::quaternion_interleaved),
          "4v+c shape-spinor" },
        { "immersionFace", "Immersion spinor (face)",
          desc(Domain::face, Bundle::immersion, FieldType::quaternion, NForm::na, Representation::quaternion_interleaved),
          "4f+c shape-spinor" },
    };
    return table;
}

const FieldVariant* fieldById(std::string_view id) {
    for (const auto& v : fieldRegistry()) if (v.id == id) return &v;
    return nullptr;
}

std::vector<const FieldVariant*>
fieldsWhere(const std::function<bool(const FieldVariant&)>& pred) {
    std::vector<const FieldVariant*> out;
    for (const auto& v : fieldRegistry()) if (pred(v)) out.push_back(&v);
    return out;
}

bool fieldMatches(const FieldDescriptor& f, const FieldDescriptor& expected) {
    // gauge and nSym are advisory parameters, not match keys.
    return f.domain       == expected.domain
        && f.bundle       == expected.bundle
        && f.field_type   == expected.field_type
        && f.n_form       == expected.n_form
        && f.representation == expected.representation;
}

void requireField(const FieldDescriptor& f, std::string_view operatorId) {
    const OperatorVariant* op = operatorById(operatorId);
    if (!op)
        throw Error(ErrorCode::InvalidInput, "unknown operator id: " + std::string(operatorId),
                    "check operatorRegistry()");
    const FieldVariant* expected = fieldById(op->input_field);
    if (!expected)
        throw Error(ErrorCode::InvalidInput,
                    "operator '" + std::string(operatorId) + "' names unknown input_field '" + op->input_field + "'",
                    "field registry integrity error");
    if (!fieldMatches(f, expected->descriptor))
        throw Error(ErrorCode::InvalidInput,
                    "field not admissible as input of '" + std::string(operatorId) + "' (expected " + op->input_field + ")",
                    "field bundle/domain/degree/representation mismatch");
}

std::vector<std::string> operatorsAccepting(const FieldDescriptor& f) {
    std::vector<std::string> out;
    for (const auto& op : operatorRegistry()) {
        const FieldVariant* in = fieldById(op.input_field);
        if (in && fieldMatches(f, in->descriptor)) out.push_back(op.id);
    }
    return out;
}

int componentsPerElement(const FieldDescriptor& f) {
    switch (f.bundle) {
        case Bundle::scalar:    return 1;
        case Bundle::tangent:   return 1;   // complex coordinate per element (VectorXcd)
        case Bundle::ambient:   return 3;
        case Bundle::immersion: return 4;
    }
    return 1;
}

void validateFieldShape(const FieldDescriptor& f, int rows, int nV, int nE, int nF) {
    int n = (f.domain == Domain::vertex) ? nV : (f.domain == Domain::edge) ? nE : nF;
    int expected = n * componentsPerElement(f);
    if (rows != expected)
        throw Error(ErrorCode::InvalidInput,
                    "field row count " + std::to_string(rows) + " != expected " + std::to_string(expected),
                    "rows must equal nElements(domain) * componentsPerElement");
}

const std::vector<ConversionEdge>& conversionGraph() {
    static const std::vector<ConversionEdge> edges = {
        { "ambientVertexWorld", "ambientVertexLocal", "differential::liftToFrame", true },
        { "ambientVertexLocal", "ambientVertexWorld", "differential::liftToWorld", true },
        { "ambientVertexWorld", "ambientVertexLocal", "G.c^T (covariantGradient correspondence)", true },
        { "oneFormEdge",        "ambientFaceWorld",   "field::interp::whitney",      true },
        { "scalarVertex",       "ambientFaceWorld",   "field::op::gradient",         true },
        { "tangentVertex",      "tangentVertex",      "lowerToReal2N (complex<->real2N)", true },
    };
    return edges;
}

} // namespace nxr::manifold::registry
