#include "nxr/operator_registry.h"
#include "nxr/errors.h"

namespace nxr::manifold::registry {

using nxr::core::Error;
using nxr::core::ErrorCode;

static CrossLink squaresTo(std::string t, Relation r = Relation::exact)  { return {true,  true,  std::move(t), r}; }
static CrossLink squareOf (std::string t, Relation r = Relation::exact)  { return {true,  false, std::move(t), r}; }

const std::vector<OperatorVariant>& operatorRegistry() {
    static const std::vector<OperatorVariant> table = {
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

        // ── ambient (real, 3-comp) ──
        { "flatCovariantLaplacian", "Flat covariant Laplacian (ambient)", Bundle::ambient, Holonomy::flat,
          Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::ambient,
          squareOf("covariantGradient", Relation::exact), "", false, "", Status::built, OperatorId::LaplacianCovariant, "" },
        { "productCovariantLaplacian", "Product covariant Laplacian (tan+nor)", Bundle::ambient, Holonomy::intrinsic_curved,
          Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::product,
          {}, "", false, "", Status::built, OperatorId::LaplacianCovariant, "" },
        { "covariantGradient", "Covariant gradient (flat transport)", Bundle::ambient, Holonomy::flat,
          Order::first, Role::gradient, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
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
