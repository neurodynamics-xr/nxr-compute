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

// fieldMatches / requireField / operatorsAccepting / componentsPerElement /
// validateFieldShape / conversionGraph are implemented in later tasks (not defined yet).

} // namespace nxr::manifold::registry
