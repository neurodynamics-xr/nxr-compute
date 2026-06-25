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
        { "scalarVertex", "Scalar (0-form, vertex)",
          desc(Domain::vertex, Bundle::scalar, FieldType::real, NForm::zero, Representation::na),
          "vertex function / 0-form" },
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
