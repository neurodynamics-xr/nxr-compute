/**
 * test_query_measure.cpp — Verify the query / measure locus APIs.
 *
 * Asserts:
 *   1. query::point + measure::point round-trip a vertex index to its
 *      3D position.
 *   2. query::line returns a non-empty polyline; measure::line is
 *      positive and >= the Euclidean distance between endpoints.
 *   3. measure::line(locus) and measure::line(m, a, b) agree.
 *   4. query::area(v, level) shrinks/grows monotonically with level.
 *   5. measure::area at a huge level recovers the full mesh area;
 *      at zero level is empty.
 *   6. measure::area(locus) and measure::area(m, v, l) agree.
 *   7. query::point / area validate inputs and throw on out-of-range.
 */

#include "nxr/compute.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nxr::manifold;

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

int main() {
    std::cout << "[test_query_measure] starting" << std::endl;

    std::vector<double>  V;
    std::vector<int32_t> F;
    buildIcosahedron(V, F);
    const int nV = static_cast<int>(V.size() / 3);
    const int nF = static_cast<int>(F.size() / 3);

    Manifold m(V.data(), nV, F.data(), nF);
    auto ops = ops::assembleManifoldOperators(m);
    const double totalArea = ops.totalArea;
    std::cout << "  mesh: " << nV << " V, " << nF << " F, total area " << totalArea << std::endl;

    // ── 1. query::point / measure::point round-trip ──────────
    {
        const int v = 5;
        auto pl = query::point(m, v);
        REQUIRE(pl.vertex == v, "query::point echoes input vertex");

        Eigen::Vector3d p = measure::point(m, v);
        const double dx = p.x() - V[3*v + 0];
        const double dy = p.y() - V[3*v + 1];
        const double dz = p.z() - V[3*v + 2];
        const double err = std::sqrt(dx*dx + dy*dy + dz*dz);
        REQUIRE(err < 1e-15, "measure::point matches vertex position");
        std::cout << "  [1] point round-trip ok" << std::endl;
    }

    // ── 2. query::line + measure::line ───────────────────────
    {
        const int a = 0;
        const int b = 3;
        auto pl = query::line(m, a, b);
        REQUIRE(pl.points.rows() >= 2, "polyline has at least 2 points");
        REQUIRE(pl.points.cols() == 3, "polyline points are 3D");

        // First and last polyline points should be the endpoint vertices.
        Eigen::Vector3d pa = measure::point(m, a);
        Eigen::Vector3d pb = measure::point(m, b);
        REQUIRE((pl.points.row(0).transpose()             - pa).norm() < 1e-10,
                "polyline starts at vertex a");
        REQUIRE((pl.points.row(pl.points.rows() - 1).transpose() - pb).norm() < 1e-10,
                "polyline ends at vertex b");

        const double L = measure::line(pl);
        const double euclid = (pa - pb).norm();
        REQUIRE(L > 0, "polyline length positive");
        REQUIRE(L >= euclid - 1e-12,
                "polyline length >= Euclidean (geodesic >= straight-line)");
        std::cout << "  [2] line length=" << L << " (>= euclid " << euclid << ")" << std::endl;

        // measure::line(m, a, b) agrees with measure::line(locus).
        const double L2 = measure::line(m, a, b);
        REQUIRE(std::abs(L - L2) < 1e-12, "measure::line overloads agree");
        std::cout << "  [3] line overloads agree" << std::endl;
    }

    // ── 3. query::area / measure::area extremes ──────────────
    {
        const int v = 0;

        // Tiny ball — possibly empty (icosahedron edge length ≈ 1.05).
        // The conservative all-vertices-inside criterion allows empty regions.
        auto rEmpty = query::area(m, v, 0.01);
        REQUIRE(rEmpty.faces.size() == 0, "tiny ball is empty");
        REQUIRE(measure::area(m, rEmpty) == 0.0, "empty region has zero area");
        std::cout << "  [4] tiny ball empty" << std::endl;

        // Huge ball — should sweep the entire mesh.
        auto rFull = query::area(m, v, 100.0);
        REQUIRE(static_cast<int>(rFull.faces.size()) == nF,
                "huge ball covers all faces");
        const double afull = measure::area(m, rFull);
        REQUIRE(std::abs(afull - totalArea) < 1e-12,
                "huge ball area == totalArea");
        std::cout << "  [5] huge ball area=" << afull << " == totalArea " << totalArea << std::endl;

        // measure::area(m, v, level) agrees with the locus overload.
        const double aA = measure::area(m, v, 100.0);
        REQUIRE(std::abs(aA - afull) < 1e-12, "measure::area overloads agree");

        // Monotonicity: more level → more area.
        auto r05 = query::area(m, v, 0.5);
        auto r10 = query::area(m, v, 1.0);
        const double a05 = measure::area(m, r05);
        const double a10 = measure::area(m, r10);
        REQUIRE(a05 <= a10 + 1e-12, "area is monotonic in level");
        REQUIRE(a10 <= afull + 1e-12, "intermediate level <= full area");
        std::cout << "  [6] area(level=0.5)=" << a05 << " <= area(level=1.0)=" << a10 << std::endl;
    }

    // ── 4. Input validation ──────────────────────────────────
    {
        bool threw = false;
        try { (void)query::point(m, -1); }
        catch (const nxr::core::Error& e) {
            threw = (e.code() == nxr::core::ErrorCode::InvalidInput);
        }
        REQUIRE(threw, "query::point throws on negative index");

        threw = false;
        try { (void)query::point(m, nV); }
        catch (const nxr::core::Error& e) {
            threw = (e.code() == nxr::core::ErrorCode::InvalidInput);
        }
        REQUIRE(threw, "query::point throws on >= nV");

        threw = false;
        try { (void)query::area(m, 0, -1.0); }
        catch (const nxr::core::Error& e) {
            threw = (e.code() == nxr::core::ErrorCode::InvalidInput);
        }
        REQUIRE(threw, "query::area throws on level <= 0");

        threw = false;
        try { (void)measure::point(m, nV + 100); }
        catch (const nxr::core::Error& e) {
            threw = (e.code() == nxr::core::ErrorCode::InvalidInput);
        }
        REQUIRE(threw, "measure::point throws on out-of-range");

        std::cout << "  [7] input validation ok" << std::endl;
    }

    std::cout << "[test_query_measure] all assertions passed ✓" << std::endl;
    return 0;
}
