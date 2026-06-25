#include "nxr/operator_registry.h"
#include "nxr/errors.h"

namespace nxr::manifold::registry {

using nxr::core::Error;
using nxr::core::ErrorCode;

static CrossLink squaresTo(std::string t, Relation r = Relation::exact)  { return {true,  true,  std::move(t), r}; }
static CrossLink squareOf (std::string t, Relation r = Relation::exact)  { return {true,  false, std::move(t), r}; }

const std::vector<OperatorVariant>& operatorRegistry() {
    static const std::vector<OperatorVariant> table = [] {
        std::vector<OperatorVariant> t = {
            // ── scalar ──
            { "laplaceBeltrami", "Laplace-Beltrami (cotan)", Bundle::scalar, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squareOf("intrinsicDirac", Relation::principal_part), "massGalerkin", false, "", Status::built, OperatorId::LaplacianCotan, "" },
            { "graphLaplacian", "Graph Laplacian (d0Td0)", Bundle::scalar, Holonomy::combinatorial,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squareOf("d0", Relation::exact), "identity", false, "", Status::built, OperatorId::LaplacianGraph, "" },
            { "faceLaplacianGreenGauss", "Face Laplacian (Green-Gauss dual)", Bundle::scalar, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squareOf("faceGradient", Relation::exact), "", false, "", Status::built, OperatorId::LapFace, "" },
            { "faceLaplacian2Form", "Face Laplacian (DEC 2-form, d1*1inv*d1T)", Bundle::scalar, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
              {}, "", false, "", Status::built, OperatorId::DiracFace, "relativeFaceDirac tau=0 anchor; internal cacheTwoFormLaplacian_" },

            // ── tangent (complex, nSym) ──
            { "leviCivitaConnectionLaplacian", "Levi-Civita connection (Bochner) Laplacian", Bundle::tangent, Holonomy::intrinsic_curved,
              Order::second, Role::connection_laplacian, FieldType::complex, Domain::vertex, Singular::none, Gauge::levi_civita, Coupling::na,
              {}, "", false, "", Status::built, OperatorId::LaplacianConnection, "domain in {vertex,face,edge}" },
            { "trivialConnectionLaplacian", "Trivial connection Laplacian", Bundle::tangent, Holonomy::flat,
              Order::second, Role::connection_laplacian, FieldType::complex, Domain::vertex, Singular::chi_defects, Gauge::trivial, Coupling::na,
              {}, "", false, "", Status::built, OperatorId::LaplacianConnection, "Sum singularity index == chi (Gauss-Bonnet)" },

            // ── tangent gradient (complex, nSym) ──
            { "leviCivitaConnectionGradient", "Levi-Civita connection gradient (d^nabla)", Bundle::tangent, Holonomy::intrinsic_curved,
              Order::first, Role::gradient, FieldType::complex, Domain::edge, Singular::none, Gauge::levi_civita, Coupling::na,
              squaresTo("leviCivitaConnectionLaplacian", Relation::exact), "", false, "", Status::built, OperatorId::ConnectionGradient, "edge<-vertex; nSym" },
            { "trivialConnectionGradient", "Trivial connection gradient (d^nabla)", Bundle::tangent, Holonomy::flat,
              Order::first, Role::gradient, FieldType::complex, Domain::edge, Singular::chi_defects, Gauge::trivial, Coupling::na,
              squaresTo("trivialConnectionLaplacian", Relation::exact), "", false, "", Status::built, OperatorId::ConnectionGradient, "edge<-vertex; nSym" },

            // ── ambient (real, 3-comp) ──
            { "flatCovariantLaplacian", "Flat covariant Laplacian (ambient)", Bundle::ambient, Holonomy::flat,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::ambient,
              squareOf("covariantGradient", Relation::exact), "", false, "", Status::built, OperatorId::LaplacianCovariant, "" },
            { "productCovariantLaplacian", "Product covariant Laplacian (tan+nor)", Bundle::ambient, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::product,
              {}, "", false, "", Status::built, OperatorId::LaplacianCovariant, "" },
            { "covariantGradient", "Covariant gradient (flat transport)", Bundle::ambient, Holonomy::flat,
              Order::first, Role::gradient, FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na,
              squaresTo("flatCovariantLaplacian", Relation::exact), "", false, "", Status::built, OperatorId::Gradient3D, "edge<-vertex" },
            { "faceGradient", "Face gradient (Green-Gauss)", Bundle::ambient, Holonomy::intrinsic_curved,
              Order::first, Role::gradient, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squaresTo("faceLaplacianGreenGauss", Relation::exact), "", false, "", Status::built, OperatorId::GradFace, "" },
            { "extrinsicWeitzenbockLaplacian", "Extrinsic Weitzenbock Laplacian (D3+D_N)", Bundle::ambient, Holonomy::extrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              {}, "", false, "", Status::planned, OperatorId::Gradient3D,
              "PLANNED: flatCovariantLaplacian + W_extrinsic; assembly throws NotImplemented until built (follow-on plan). op_id placeholder until its own OperatorId is added." },

            // ── immersion (quaternion, 4-comp) ──
            { "intrinsicDirac", "Intrinsic Dirac (1st-order)", Bundle::immersion, Holonomy::intrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squaresTo("laplaceBeltrami", Relation::principal_part), "", false, "", Status::built, OperatorId::DiracIntrinsicD, "face<-vertex; scalar part of square == cotanL" },
            { "extrinsicDirac", "Extrinsic Dirac (1st-order, Gauss map)", Bundle::immersion, Holonomy::extrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squaresTo("relativeDirac", Relation::exact), "", false, "", Status::built, OperatorId::DiracD, "face<-vertex; squares to relativeDirac@tau=1 (E)" },
            { "relativeDirac", "Relative Dirac (vertex tau-family)", Bundle::immersion, Holonomy::graded,
              Order::second, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squareOf("extrinsicDirac", Relation::exact), "massGalerkin*I4", true, "intrinsic=0;extrinsic=1;squared=0.5x2", Status::built, OperatorId::Dirac, "" },
            { "intrinsicFaceDirac", "Intrinsic face Dirac (1st-order)", Bundle::immersion, Holonomy::intrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squaresTo("faceLaplacian2Form", Relation::principal_part), "", false, "", Status::built, OperatorId::DiracFaceIntrinsicD, "vertex<-face; closed-mesh v1" },
            { "extrinsicFaceDirac", "Extrinsic face Dirac (1st-order)", Bundle::immersion, Holonomy::extrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squaresTo("relativeFaceDirac", Relation::exact), "", false, "", Status::built, OperatorId::DiracFaceD, "vertex<-face; closed-mesh v1" },
            { "relativeFaceDirac", "Relative face Dirac (tau-family)", Bundle::immersion, Holonomy::graded,
              Order::second, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squareOf("extrinsicFaceDirac", Relation::exact), "diag(faceArea)*I4", true, "intrinsic=0;extrinsic=1;squared=0.5x2", Status::built, OperatorId::DiracFace, "closed-mesh v1" },

            // ── metrics & exterior calculus ──
            { "massLumped", "Lumped (barycentric) mass", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::mass,
              FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::MassLumped, "holonomy n/a (metric)" },
            { "massGalerkin", "Galerkin (FEM) mass", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::mass,
              FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::MassGalerkin, "holonomy n/a (metric)" },
            { "d0", "Exterior derivative d0 (grad)", Bundle::scalar, Holonomy::combinatorial, Order::first, Role::exterior_derivative,
              FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "edge<-vertex" },
            { "d1", "Exterior derivative d1 (curl)", Bundle::scalar, Holonomy::combinatorial, Order::first, Role::exterior_derivative,
              FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "face<-edge" },
            { "hodge0", "Hodge star star0", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
            { "hodge1", "Hodge star star1", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
            { "hodge2", "Hodge star star2", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
            { "hodge1inv", "Hodge star star1 inverse", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, OperatorId::Dec, "holonomy n/a (metric)" },
        };
        // Field I/O declared adjacent, keyed by curated id (NOT OperatorId: DiracFace
        // backs both faceLaplacian2Form and relativeFaceDirac with different I/O).
        struct IO { const char* id; const char* in; const char* out; };
        static const IO io[] = {
            { "laplaceBeltrami",                "scalarVertex",      "scalarVertex" },
            { "graphLaplacian",                 "scalarVertex",      "scalarVertex" },
            { "faceLaplacianGreenGauss",        "twoFormFace",       "twoFormFace" },
            { "faceLaplacian2Form",             "twoFormFace",       "twoFormFace" },
            { "leviCivitaConnectionLaplacian",  "tangentVertex",     "tangentVertex" },
            { "trivialConnectionLaplacian",     "tangentVertex",     "tangentVertex" },
            { "leviCivitaConnectionGradient",   "tangentVertex",     "tangentEdge" },
            { "trivialConnectionGradient",      "tangentVertex",     "tangentEdge" },
            { "flatCovariantLaplacian",         "ambientVertexLocal","ambientVertexLocal" },
            { "productCovariantLaplacian",      "ambientVertexLocal","ambientVertexLocal" },
            { "covariantGradient",              "ambientVertexLocal","ambientEdge" },
            { "faceGradient",                   "twoFormFace",       "ambientFaceWorld" },
            { "extrinsicWeitzenbockLaplacian",  "ambientVertexLocal","ambientVertexLocal" },
            { "intrinsicDirac",                 "immersionVertex",   "immersionFace" },
            { "extrinsicDirac",                 "immersionVertex",   "immersionFace" },
            { "relativeDirac",                  "immersionVertex",   "immersionVertex" },
            { "intrinsicFaceDirac",             "immersionFace",     "immersionVertex" },
            { "extrinsicFaceDirac",             "immersionFace",     "immersionVertex" },
            { "relativeFaceDirac",              "immersionFace",     "immersionFace" },
            { "massLumped",                     "scalarVertex",      "scalarVertex" },
            { "massGalerkin",                   "scalarVertex",      "scalarVertex" },
            { "d0",                             "scalarVertex",      "oneFormEdge" },
            { "d1",                             "oneFormEdge",       "twoFormFace" },
            { "hodge0",                         "scalarVertex",      "scalarVertex" },
            { "hodge1",                         "oneFormEdge",       "oneFormEdge" },
            { "hodge2",                         "twoFormFace",       "twoFormFace" },
            { "hodge1inv",                      "oneFormEdge",       "oneFormEdge" },
        };
        for (auto& e : t)
            for (const auto& x : io)
                if (e.id == x.id) { e.input_field = x.in; e.output_field = x.out; break; }
        return t;
    }();
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

// No `default:` — adding an OperatorId enumerator without a branch triggers
// -Wswitch (promoted to error for this TU) so metadata can't be forgotten.
std::vector<std::string> variantIdsFor(OperatorId op) {
    switch (op) {
        case OperatorId::LaplacianCotan:      return {"laplaceBeltrami"};
        case OperatorId::LaplacianGraph:      return {"graphLaplacian"};
        case OperatorId::LaplacianConnection: return {"leviCivitaConnectionLaplacian", "trivialConnectionLaplacian"};
        case OperatorId::LaplacianCovariant:  return {"flatCovariantLaplacian", "productCovariantLaplacian"};
        case OperatorId::Dec:                 return {"d0", "d1", "hodge0", "hodge1", "hodge2", "hodge1inv"};
        case OperatorId::MassLumped:          return {"massLumped"};
        case OperatorId::MassGalerkin:        return {"massGalerkin"};
        // NOTE: when the follow-on plan adds OperatorId::ExtrinsicWeitzenbock, add its
        // case here AND repoint the extrinsicWeitzenbockLaplacian table entry's op_id
        // off the Gradient3D placeholder (not compiler-enforced — see its notes field).
        case OperatorId::Gradient3D:          return {"covariantGradient"};
        case OperatorId::Dirac:               return {"relativeDirac"};
        case OperatorId::DiracFace:           return {"relativeFaceDirac", "faceLaplacian2Form"};
        case OperatorId::DiracD:              return {"extrinsicDirac"};
        case OperatorId::DiracFaceD:          return {"extrinsicFaceDirac"};
        case OperatorId::DiracIntrinsicD:     return {"intrinsicDirac"};
        case OperatorId::DiracFaceIntrinsicD: return {"intrinsicFaceDirac"};
        case OperatorId::GradFace:            return {"faceGradient"};
        case OperatorId::LapFace:             return {"faceLaplacianGreenGauss"};
        case OperatorId::ConnectionGradient:  return {"leviCivitaConnectionGradient", "trivialConnectionGradient"};
    }
    return {};   // unreachable; silences control-reaches-end warning
}

} // namespace nxr::manifold::registry
