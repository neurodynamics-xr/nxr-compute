#include "nxr/operator_registry.h"
#include "nxr/errors.h"

namespace nxr::manifold::registry {

using nxr::core::Error;
using nxr::core::ErrorCode;

static CrossLink squaresTo(std::string t, Relation r = Relation::exact)  { return {true,  true,  std::move(t), r}; }
static CrossLink squareOf (std::string t, Relation r = Relation::exact)  { return {true,  false, std::move(t), r}; }

const std::vector<OperatorVariant>& operatorRegistry() {
    static const std::vector<OperatorVariant> table = {
        { "laplaceBeltrami", "Laplace–Beltrami (cotan)",
          Bundle::scalar, Holonomy::intrinsic_curved, Order::second, Role::laplacian,
          FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
          squareOf("intrinsicDirac", Relation::principal_part),
          "massGalerkin", false, "", Status::built, OperatorId::LaplacianCotan, "" },
    };
    return table;
}

const OperatorVariant* operatorById(std::string_view id) {
    for (const auto& v : operatorRegistry()) if (v.id == id) return &v;
    return nullptr;
}

std::vector<const OperatorVariant*>
operatorsWhere(const std::function<bool(const OperatorVariant&)>& pred) {
    std::vector<const OperatorVariant*> out;
    for (const auto& v : operatorRegistry()) if (pred(v)) out.push_back(&v);
    return out;
}

void requireBundle(std::string_view id, Bundle required) {
    const OperatorVariant* v = operatorById(id);
    if (!v)
        throw Error(ErrorCode::InvalidInput,
                    "unknown operator id: " + std::string(id),
                    "check nxr::manifold::registry::operatorRegistry()");
    if (v->bundle != required)
        throw Error(ErrorCode::InvalidInput,
                    std::string("operator '") + std::string(id) + "' has bundle " +
                        toString(v->bundle) + ", required " + toString(required),
                    "immersion operators are not valid ambient differentiators (LJC trap)");
}

// variantIdsFor is intentionally NOT defined in Task 1 — no call site references it yet.

} // namespace nxr::manifold::registry
