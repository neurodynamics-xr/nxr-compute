/**
 * test_operator_registry.cpp — registry completeness + cross-link identities.
 * Build: cmake --build build --target test_operator_registry
 * Run:   ./build/test_operator_registry
 */
#include "nxr/operator_registry.h"
#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace nxr::manifold;
using namespace nxr::manifold::registry;
using namespace nxr::manifold::ops;
using namespace nxr::manifold::solve;

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << (msg) << "\n"; ++g_failures; } } while (0)

// ── Icosphere helper (icosahedron, 12 verts / 20 faces, on unit sphere) ──────
static void makeIcosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    V.clear();
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        V.push_back(raw[i] / len);
        V.push_back(raw[i+1] / len);
        V.push_back(raw[i+2] / len);
    }
    F = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
}

// ── Sparse-matrix infinity-norm difference helper ─────────────────────────────
static double maxAbsDiff(const Eigen::SparseMatrix<double>& A,
                         const Eigen::SparseMatrix<double>& B) {
    Eigen::SparseMatrix<double> D = A - B;
    double m = 0.0;
    for (int k = 0; k < D.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(D, k); it; ++it)
            m = std::max(m, std::abs(it.value()));
    return m;
}

// ── Numerical cross-link identity checks ──────────────────────────────────────
// Verifies the four squares_to / square_of identities declared in the registry.
// Each check prints the actual maxAbsDiff so the caller can see the magnitude.
// Relation::exact  → tolerance 1e-9
// Relation::principal_part → expected to be larger; any failure is reported
//   (do not loosen the tolerance here — report to the caller instead).
static void test_cross_link_identities() {
    std::cout << "\n── Cross-link identity checks ──\n";

    std::vector<double>  V;
    std::vector<int32_t> F;
    makeIcosphere(V, F);
    int nV = static_cast<int>(V.size() / 3);
    int nF_ = static_cast<int>(F.size() / 3);
    Manifold m(V.data(), nV, F.data(), nF_);

    // 1. graphLaplacian square_of d0 (exact):  d0^T d0 == graphL
    {
        const Eigen::SparseMatrix<double>& D0 = d0(m);
        Eigen::SparseMatrix<double> sq = (D0.transpose() * D0).pruned();
        Eigen::SparseMatrix<double> gl = m.operators().laplacian().graph();
        double diff = maxAbsDiff(sq, gl);
        std::cout << "  [1] graphLaplacian == d0^T d0 (exact):      maxAbsDiff = " << diff << "\n";
        CHECK(diff < 1e-9, "graphLaplacian == d0^T d0 (exact)");
    }

    // 2. relativeDirac@tau=0 == cotanL ⊗ I4  (exact built-in anchor)
    {
        Eigen::SparseMatrix<double> d0tau = m.operators().dirac(0.0);
        Eigen::SparseMatrix<double> kron  = blockKron(m.operators().laplacian().cotan(), 4);
        double diff = maxAbsDiff(d0tau, kron);
        std::cout << "  [2] relativeDirac(0) == cotanL⊗I4 (exact): maxAbsDiff = " << diff << "\n";
        CHECK(diff < 1e-9, "relativeDirac(0) == cotanL⊗I4 (exact)");
    }

    // 3. squared preset: 2*relativeDirac(1/2) == relativeDirac(0)+relativeDirac(1) (exact)
    {
        Eigen::SparseMatrix<double> half    = m.operators().dirac(0.5);
        Eigen::SparseMatrix<double> sum     = m.operators().dirac(0.0) + m.operators().dirac(1.0);
        Eigen::SparseMatrix<double> twoHalf = 2.0 * half;
        double diff = maxAbsDiff(twoHalf, sum);
        std::cout << "  [3] 2*relativeDirac(1/2)==D(0)+D(1) (exact):maxAbsDiff = " << diff << "\n";
        CHECK(diff < 1e-9, "2*relativeDirac(1/2) == D(0)+D(1) (squared preset, exact)");
    }

    // 4. intrinsicDirac squares_to laplaceBeltrami (principal_part):
    //    scalar block (w-component) of D_int^T (diag(faceArea)⊗I4) D_int == cotanL
    {
        const Eigen::SparseMatrix<double>& Dint = m.operators().diracIntrinsicD(); // [4F x 4V]
        auto& geom = m.geometry();
        geom.requireFaceAreas();
        // Build W_F = diag(faceArea) ⊗ I4  [4F x 4F]
        std::vector<Eigen::Triplet<double>> T;
        T.reserve(nF_);
        for (auto f : m.mesh().faces())
            T.emplace_back(static_cast<int>(f.getIndex()),
                           static_cast<int>(f.getIndex()), geom.faceAreas[f]);
        Eigen::SparseMatrix<double> Af(nF_, nF_);
        Af.setFromTriplets(T.begin(), T.end());
        Eigen::SparseMatrix<double> WF = blockKron(Af, 4);               // [4F x 4F]

        Eigen::SparseMatrix<double> sq = (Dint.transpose() * WF * Dint).pruned(); // [4V x 4V]

        // Extract the scalar (w) block: entries where row%4==0 and col%4==0
        std::vector<Eigen::Triplet<double>> Ts;
        for (int k = 0; k < sq.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(sq, k); it; ++it)
                if (it.row() % 4 == 0 && it.col() % 4 == 0)
                    Ts.emplace_back(it.row() / 4, it.col() / 4, it.value());
        Eigen::SparseMatrix<double> scalarPart(nV, nV);
        scalarPart.setFromTriplets(Ts.begin(), Ts.end());

        Eigen::SparseMatrix<double> cotanL = m.operators().laplacian().cotan();
        double diff = maxAbsDiff(scalarPart, cotanL);
        std::cout << "  [4] intrinsicDirac scalar block==cotanL (principal_part): maxAbsDiff = "
                  << diff << "\n";
        if (diff >= 1e-9) {
            std::cout << "      NOTE: relation is principal_part — "
                      << "non-zero diff is expected on a curved mesh (curvature correction).\n"
                      << "      Reporting magnitude for caller review; NOT failing the build.\n";
            // principal_part: report but do not increment g_failures
        }
    }
}

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

static void test_query_and_guard() {
    auto firstImmersion = operatorsWhere([](const OperatorVariant& v) {
        return v.bundle == Bundle::immersion && v.order == Order::first;
    });
    if (firstImmersion.size() != 4) {
        std::cerr << "DEBUG firstImmersion count=" << firstImmersion.size() << ": ";
        for (auto* v : firstImmersion) std::cerr << v->id << " ";
        std::cerr << "\n";
    }
    CHECK(firstImmersion.size() == 4, "4 first-order immersion ops");

    // LJC-trap guard: immersion op rejected where ambient required.
    bool threw = false;
    try { requireBundle("relativeDirac", Bundle::ambient); }
    catch (const nxr::core::Error&) { threw = true; }
    CHECK(threw, "requireBundle throws on immersion!=ambient");

    bool ok = true;
    try { requireBundle("flatCovariantLaplacian", Bundle::ambient); }
    catch (const nxr::core::Error&) { ok = false; }
    CHECK(ok, "requireBundle passes on ambient==ambient");
}

static void test_completeness() {
    const OperatorId all[] = {
        OperatorId::LaplacianCotan, OperatorId::LaplacianGraph, OperatorId::LaplacianConnection,
        OperatorId::LaplacianCovariant, OperatorId::Dec, OperatorId::MassLumped,
        OperatorId::MassGalerkin, OperatorId::Gradient3D, OperatorId::Dirac, OperatorId::DiracFace,
        OperatorId::DiracD, OperatorId::DiracFaceD, OperatorId::DiracIntrinsicD,
        OperatorId::DiracFaceIntrinsicD, OperatorId::GradFace, OperatorId::LapFace,
    };
    for (OperatorId op : all) {
        auto ids = variantIdsFor(op);
        CHECK(!ids.empty(), "OperatorId has >=1 variant");
        for (const auto& id : ids)
            CHECK(operatorById(id) != nullptr, "variantIdsFor id resolves: " + id);
    }
}

int main() {
    test_scalar_skeleton();
    test_full_population();
    test_completeness();
    test_query_and_guard();
    test_cross_link_identities();
    std::cout << (g_failures ? "REGISTRY TESTS FAILED\n" : "ALL REGISTRY TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
