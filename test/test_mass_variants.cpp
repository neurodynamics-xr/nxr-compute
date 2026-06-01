/**
 * test_mass_variants.cpp — correctness checks for the two mass-matrix
 * variants (Lumped, Galerkin) on a fixed icosphere mesh.
 *
 * Names match geometry-central's vocabulary exactly:
 *   Lumped    ↔  GC vertexLumpedMassMatrix    (diagonal, A/3-per-vertex)
 *   Galerkin  ↔  GC vertexGalerkinMassMatrix  (sparse, P1 hat L² products)
 *
 * Verifies (each variant):
 *   - matrix is symmetric to machine precision
 *   - structural pattern is correct (diagonal vs sparse off-diagonal)
 *   - total mass is conserved (Σᵢⱼ Mᵢⱼ == surface area, exact equality)
 *   - vertexDualAreas (always GC-sourced) matches the reference run
 *   - eigenvalues are non-negative (modulo floating-point noise near λ₀)
 *   - eigenvalues differ between variants (both Lumped and Galerkin
 *     must each produce a distinct λ₁ — catches accidental aliasing)
 *
 * Build: cmake --build build --target test_mass_variants
 * Run:   ./build/test_mass_variants
 */

#include "nxr/compute.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

using namespace nxr::manifold;
using namespace nxr::manifold::solve;
using namespace nxr::manifold::ops;
using namespace nxr::manifold::ops::laplacian::connection;
using namespace nxr::manifold::transport;
using namespace nxr::manifold::connection;
using namespace nxr::manifold::parametrization;
using namespace nxr::manifold::parametrization::stripes;
using namespace nxr::manifold::geometry;
using namespace nxr::manifold::query;
using namespace nxr::field::generate;
using namespace nxr::field::interp;
using namespace nxr::field::op;
using namespace nxr::field::extract;

using nxr::manifold::ops::MassMatrixVariant;

static void generateIcosphere(std::vector<double>& V, std::vector<int32_t>& F) {
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

static const char* variantName(MassMatrixVariant v) {
    switch (v) {
        case MassMatrixVariant::Lumped:   return "lumped";
        case MassMatrixVariant::Galerkin: return "galerkin";
    }
    return "?";
}

static int failures = 0;

static void check(bool cond, const std::string& label) {
    if (cond) {
        std::cout << "    ✓ " << label << std::endl;
    } else {
        std::cout << "    ✗ " << label << std::endl;
        ++failures;
    }
}

int main() {
    std::cout << "==============================================\n"
              << "  Mass-matrix variant correctness suite\n"
              << "==============================================\n";

    std::vector<double> V;
    std::vector<int32_t> F;
    generateIcosphere(V, F);
    int nV = static_cast<int>(V.size() / 3);
    int nF = static_cast<int>(F.size() / 3);
    std::cout << "\nMesh: icosahedron, " << nV << " vertices, "
              << nF << " faces\n";

    // For an icosahedron with unit-radius vertices on the sphere, the
    // total surface area is fixed by the embedding — compute it from
    // the lumped run as the reference and require all variants match
    // it exactly (within float roundoff).
    nxr::manifold::Manifold m(V.data(), nV, F.data(), nF);
    auto refOps = nxr::manifold::ops::assembleManifoldOperators(
        m, MassMatrixVariant::Lumped);
    const double refArea = refOps.totalArea;
    std::cout << "Reference surface area (lumped) = "
              << std::setprecision(12) << refArea << "\n";

    // Non-trivial eigenvalues from each variant — to confirm they
    // produce numerically distinct spectra.
    Eigen::VectorXd evalsByVariant[2];
    int variantIdx = 0;

    for (auto v : {MassMatrixVariant::Lumped,
                   MassMatrixVariant::Galerkin}) {
        std::cout << "\n── variant: " << variantName(v) << " ─────────\n";

        // Fresh context per variant — avoid any state leak between runs.
        nxr::manifold::Manifold localCtx(V.data(), nV, F.data(), nF);
        auto ops = nxr::manifold::ops::assembleManifoldOperators(localCtx, v);

        check(ops.massVariant == v,                       "ops.massVariant tag");
        check(ops.mass.rows() == nV && ops.mass.cols() == nV,
              "M shape == nV×nV");

        // Symmetry to machine precision.
        Eigen::SparseMatrix<double> diff = ops.mass - Eigen::SparseMatrix<double>(ops.mass.transpose());
        const double symErr = diff.norm();
        check(symErr < 1e-12,                             "M is symmetric (||M - Mᵀ||_F)");

        // Structural pattern: diagonal vs sparse off-diagonal.
        int nnz = static_cast<int>(ops.mass.nonZeros());
        if (v == MassMatrixVariant::Galerkin) {
            // Each face contributes 9 entries; deduplicated by setFromTriplets.
            // Lower bound: at least nV diagonal + 2 * nE off-diagonal.
            // Upper bound: 9 * nF (no dedup).
            check(nnz > nV,                               "galerkin: nnz > nV (has off-diagonal)");
        } else {
            check(nnz == nV,                              "lumped: nnz == nV (diagonal)");
        }

        // Total mass conservation: Σᵢⱼ Mᵢⱼ == totalArea, exactly.
        // For diagonal variants this collapses to Σᵢ Mᵢᵢ == totalArea.
        double sumAll = 0.0;
        for (int k = 0; k < ops.mass.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(ops.mass, k); it; ++it) {
                sumAll += it.value();
            }
        }
        const double areaErr = std::abs(sumAll - ops.totalArea);
        check(areaErr < 1e-10 * std::max(1.0, ops.totalArea),
              "Σᵢⱼ Mᵢⱼ == totalArea");

        // Total area equals the lumped reference.
        const double refErr = std::abs(ops.totalArea - refArea);
        check(refErr < 1e-10 * std::max(1.0, refArea),
              "totalArea matches lumped reference");

        // vertexDualAreas is always sourced from GC's vertexDualAreas
        // (A/3-per-vertex) regardless of mass variant → match ref run.
        const double vaErr = (ops.vertexDualAreas - refOps.vertexDualAreas).norm();
        check(vaErr < 1e-12,                              "vertexDualAreas == refOps.vertexDualAreas");

        // Run the eigensolver. Smallest 6 modes; should converge cheaply
        // on 12 vertices.
        const int k = 6;
        auto eig = nxr::manifold::solve::eigen(ops.cotanLaplacian, ops.mass, k);
        check(eig.k == k,                                 "k modes returned");
        check(eig.nConverged == k,                        "all modes converged");

        bool nonneg = true;
        for (int i = 0; i < eig.k; ++i) {
            if (eig.eigenvalues(i) < -1e-9) { nonneg = false; break; }
        }
        check(nonneg,                                     "λ_i >= 0 (modulo float roundoff)");

        std::cout << "    eigenvalues: [";
        for (int i = 0; i < eig.k; ++i) {
            if (i) std::cout << ", ";
            std::cout << std::scientific << std::setprecision(3) << eig.eigenvalues(i);
        }
        std::cout << "]\n";

        evalsByVariant[variantIdx++] = eig.eigenvalues;
    }

    // Cross-variant sanity: at icosahedron resolution Lumped and Galerkin
    // produce numerically distinct λ₁. They converge on refinement, but
    // not at coarse resolution. This check guards against accidental
    // aliasing where both variants compute the same matrix — e.g., the
    // historical bug where MassMatrixVariant::Voronoi and ::Barycentric
    // were both sourced from the same GC code path and produced
    // bit-identical output.
    std::cout << "\n── variant cross-comparison ─────────\n";
    auto& ev0 = evalsByVariant[0];   // Lumped
    auto& ev1 = evalsByVariant[1];   // Galerkin
    if (ev0.size() >= 2 && ev1.size() >= 2) {
        const double d = std::abs(ev0(1) - ev1(1));
        const double tol = 1e-6 * std::max(std::abs(ev0(1)), std::abs(ev1(1)));
        std::cout << "    |λ₁(lumped) - λ₁(galerkin)| = " << d
                  << "  (tol = " << tol << ")\n";
        check(d > tol,
              "lumped and galerkin produce distinct λ₁ (catches aliasing)");
    }

    // Parser round-trip.
    std::cout << "\n── parseMassMatrixVariant() ─────────\n";
    check(nxr::manifold::ops::parseMassMatrixVariant("lumped")   == MassMatrixVariant::Lumped,   "lumped");
    check(nxr::manifold::ops::parseMassMatrixVariant("galerkin") == MassMatrixVariant::Galerkin, "galerkin");

    // Hard break: old names must throw.
    auto throwsOnName = [](const std::string& s) {
        try {
            (void) nxr::manifold::ops::parseMassMatrixVariant(s);
        } catch (const nxr::core::Error&) {
            return true;
        }
        return false;
    };
    check(throwsOnName("voronoi"),     "old name 'voronoi' throws (hard break)");
    check(throwsOnName("barycentric"), "old name 'barycentric' throws (hard break)");
    check(throwsOnName("full"),        "old name 'full' throws (hard break)");
    check(throwsOnName("nope"),        "unknown name throws Error");

    std::cout << "\n==============================================\n";
    if (failures == 0) {
        std::cout << "  ALL CHECKS PASSED.\n";
    } else {
        std::cout << "  " << failures << " CHECK(S) FAILED.\n";
    }
    std::cout << "==============================================\n";
    return failures == 0 ? 0 : 1;
}
