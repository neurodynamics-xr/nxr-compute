#pragma once
#include "nxr/compute.h"          // OperatorId, Manifold
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace nxr::manifold::registry {

// ── Controlled vocabularies (machine-checkable enums) ────────────────
enum class Bundle    { scalar, tangent, ambient, immersion };
enum class Holonomy  { flat, combinatorial, intrinsic_curved, extrinsic_curved, graded };
enum class Order     { zeroth, first, second };
enum class Role      { laplacian, connection_laplacian, dirac, gradient,
                       exterior_derivative, mass, hodge_star };
enum class FieldType { real, complex, quaternion };
enum class Domain    { vertex, face, edge };
enum class Gauge     { na, euclidean, levi_civita, trivial };
enum class Coupling  { na, product, ambient };
enum class Singular  { none, chi_defects };
enum class Status    { built, planned };
enum class Relation  { exact, principal_part };

inline const char* toString(Bundle v)    { static const char* s[]{"scalar","tangent","ambient","immersion"}; return s[(int)v]; }
inline const char* toString(Holonomy v)  { static const char* s[]{"flat","combinatorial","intrinsic_curved","extrinsic_curved","graded"}; return s[(int)v]; }
inline const char* toString(Order v)     { static const char* s[]{"zeroth","first","second"}; return s[(int)v]; }
inline const char* toString(Role v)      { static const char* s[]{"laplacian","connection_laplacian","dirac","gradient","exterior_derivative","mass","hodge_star"}; return s[(int)v]; }
inline const char* toString(FieldType v) { static const char* s[]{"real","complex","quaternion"}; return s[(int)v]; }
inline const char* toString(Domain v)    { static const char* s[]{"vertex","face","edge"}; return s[(int)v]; }
inline const char* toString(Gauge v)     { static const char* s[]{"n/a","euclidean","levi-civita","trivial"}; return s[(int)v]; }
inline const char* toString(Coupling v)  { static const char* s[]{"n/a","product","ambient"}; return s[(int)v]; }
inline const char* toString(Singular v)  { static const char* s[]{"none","chi_defects"}; return s[(int)v]; }
inline const char* toString(Status v)    { static const char* s[]{"built","planned"}; return s[(int)v]; }
inline const char* toString(Relation v)  { static const char* s[]{"exact","principal_part"}; return s[(int)v]; }

// Cross-link: this operator's root (for a second-order op) or square (first-order).
struct CrossLink {
    bool        present  = false;
    bool        isSquaresTo = false;   // true: this→square ; false: this←root (square_of)
    std::string target;                // id of the linked operator (may be empty for none)
    Relation    relation = Relation::exact;
};

// One catalogued operator. The `id` string is the stable key; op_id + the
// gauge/coupling/domain/tau address complete the resolution to an assembly path.
struct OperatorVariant {
    std::string id;
    std::string label;
    Bundle      bundle;
    Holonomy    holonomy;
    Order       order;
    Role        role;
    FieldType   field_type;
    Domain      domain;
    Singular    singular   = Singular::none;
    Gauge       gauge      = Gauge::na;
    Coupling    coupling   = Coupling::na;
    CrossLink   square;
    std::string natural_mass;          // "massGalerkin", "identity", "massGalerkin⊗I4", …
    bool        graded     = false;    // τ-family
    std::string tau_presets;           // e.g. "intrinsic=0;extrinsic=1;squared=0.5x2"
    Status      status     = Status::built;
    OperatorId  op_id;                 // the (possibly shared) enum slot
    std::string notes;
};

// ── API ──────────────────────────────────────────────────────────────
const std::vector<OperatorVariant>& operatorRegistry();
const OperatorVariant*  operatorById(std::string_view id);
std::vector<const OperatorVariant*>
                        operatorsWhere(const std::function<bool(const OperatorVariant&)>& pred);

// LJC-trap guard: throw Error(InvalidInput) if `id`'s bundle != required.
void requireBundle(std::string_view id, Bundle required);

// Completeness: the curated id(s) catalogued for an OperatorId. No `default`
// case — adding an OperatorId without a branch fails to compile (-Wswitch).
std::vector<std::string> variantIdsFor(OperatorId op);

} // namespace nxr::manifold::registry
