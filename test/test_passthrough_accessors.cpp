/**
 * test_passthrough_accessors.cpp — Verify zero-copy DEC / vertexDualAreas
 * accessors are bit-identical to assembleDECOperators / the corresponding
 * VertexData<double>, and that repeated calls return the same reference.
 *
 *   1. Each accessor returns a matrix bit-equal to the field on the
 *      DECOperators struct returned by assembleDECOperators (i.e. zero
 *      drift vs. the legacy API).
 *   2. Two calls to the same accessor return the same reference (the
 *      address of the underlying field) — proves no copy, no rebuild.
 *   3. vertexDualAreas returns an Eigen::VectorXd whose entries match
 *      geometry-central's VertexData<double> read element-by-element.
 */

#include "nxr/compute.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

static void buildIcosahedron(std::vector<double>& V, std::vector<int32_t>& F) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        V.push_back(raw[i]   / len);
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

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << #cond << ")" << std::endl; \
        std::exit(1); \
    } \
} while (0)

// Frobenius norm of (A - B) for sparse matrices.
static double sparseDiffNorm(const Eigen::SparseMatrix<double>& A,
                             const Eigen::SparseMatrix<double>& B) {
    Eigen::SparseMatrix<double> D = A - B;
    return D.norm();
}

int main() {
    std::cout << "[test_passthrough_accessors] starting" << std::endl;

    std::vector<double>  V;
    std::vector<int32_t> F;
    buildIcosahedron(V, F);
    const int nV = static_cast<int>(V.size() / 3);
    const int nF = static_cast<int>(F.size() / 3);
    std::cout << "  mesh: " << nV << " vertices, " << nF << " faces" << std::endl;

    Manifold m(V.data(), nV, F.data(), nF);

    // ── 1. Bit-identical to assembleDECOperators ────────────────
    {
        DECOperators dec = assembleDECOperators(m);

        REQUIRE(sparseDiffNorm(d0(m),            dec.d0)            < 1e-15, "d0 matches struct field");
        REQUIRE(sparseDiffNorm(d1(m),            dec.d1)            < 1e-15, "d1 matches struct field");
        REQUIRE(sparseDiffNorm(hodge0(m),        dec.hodge0)        < 1e-15, "hodge0 matches struct field");
        REQUIRE(sparseDiffNorm(hodge1(m),        dec.hodge1)        < 1e-15, "hodge1 matches struct field");
        REQUIRE(sparseDiffNorm(hodge2(m),        dec.hodge2)        < 1e-15, "hodge2 matches struct field");
        REQUIRE(sparseDiffNorm(hodge1Inverse(m), dec.hodge1Inverse) < 1e-15, "hodge1Inverse matches struct field");

        std::cout << "  ✓ DEC accessors bit-identical to assembleDECOperators" << std::endl;
    }

    // ── 2. Repeat calls return the same reference (no copy) ────
    {
        const auto& a = d0(m);
        const auto& b = d0(m);
        REQUIRE(&a == &b, "d0 returns same reference on repeat call");

        const auto& h1a = hodge1(m);
        const auto& h1b = hodge1(m);
        REQUIRE(&h1a == &h1b, "hodge1 returns same reference on repeat call");

        const auto& va_a = vertexDualAreas(m);
        const auto& va_b = vertexDualAreas(m);
        REQUIRE(&va_a == &va_b, "vertexDualAreas returns same reference on repeat call");

        std::cout << "  ✓ Repeat calls return identical reference (zero-copy)" << std::endl;
    }

    // ── 3. vertexDualAreas matches assembleManifoldOperators' vertexAreas ──
    //
    // assembleManifoldOperators currently builds its `vertexAreas` VectorXd
    // by copying out of `geometry.vertexDualAreas` element-by-element
    // (see mesh_operators.cpp). The accessor here returns GC's raw
    // VectorXd directly — so the two should be bit-identical.
    {
        const Eigen::VectorXd& dual = vertexDualAreas(m);
        REQUIRE(static_cast<int>(dual.size()) == nV, "vertexDualAreas size == nV");

        // All entries strictly positive (Voronoi dual areas).
        for (int i = 0; i < nV; ++i) {
            REQUIRE(dual(i) > 0.0, "vertexDualAreas entry > 0");
        }

        // Bit-identical to assembleManifoldOperators(m).vertexAreas.
        ManifoldOperators ops = assembleManifoldOperators(m);
        REQUIRE(ops.vertexDualAreas.size() == dual.size(), "vertexAreas size match");
        const double diff = (ops.vertexDualAreas - dual).cwiseAbs().maxCoeff();
        REQUIRE(diff < 1e-15, "vertexDualAreas == assembleManifoldOperators(m).vertexAreas");

        std::cout << "  ✓ vertexDualAreas matches assembleManifoldOperators "
                  << "(sum = " << dual.sum() << ", max diff = " << diff << ")" << std::endl;
    }

    std::cout << "[test_passthrough_accessors] OK" << std::endl;
    return 0;
}
