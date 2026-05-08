/**
 * test_visualization_primitives.cpp — verifies the three new
 * visualization-layer primitives:
 *
 *   1. computeFaceFrames    — per-face orthonormal (e1, e2, n)
 *   2. tracePath            — geodesic path via flip-out
 *   3. computeUVCoordinates — BFF parametrization
 *
 * Face frames + geodesic path: tested on the icosahedron fixture
 * shared with the other tests. BFF: needs an open mesh, so we
 * build a small triangulated grid (a flat disc topology).
 */

#include "nxr/compute.h"

#include <Eigen/Geometry>   // Vector3d::cross — not pulled in by compute.h's Core-only include

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nxr::compute;

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << #cond << ")" << std::endl; \
        std::exit(1); \
    } \
} while (0)

// ── Icosahedron fixture (closed mesh, no boundary) ────────────

static void buildIcosahedron(std::vector<double>& V, std::vector<int32_t>& F) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
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

// ── Flat triangulated grid (open mesh, has boundary) ──────────
//
// nGrid x nGrid vertices in the unit square at z=0, triangulated
// into 2(nGrid-1)^2 right triangles. Boundary is a single loop
// of length 4(nGrid-1) along the square perimeter.

static void buildGrid(int nGrid, std::vector<double>& V, std::vector<int32_t>& F) {
    V.clear();
    F.clear();
    for (int j = 0; j < nGrid; j++) {
        for (int i = 0; i < nGrid; i++) {
            V.push_back(static_cast<double>(i) / (nGrid - 1));  // x
            V.push_back(static_cast<double>(j) / (nGrid - 1));  // y
            V.push_back(0.0);                                    // z
        }
    }
    auto idx = [nGrid](int i, int j) { return j * nGrid + i; };
    for (int j = 0; j < nGrid - 1; j++) {
        for (int i = 0; i < nGrid - 1; i++) {
            F.push_back(idx(i,     j));
            F.push_back(idx(i + 1, j));
            F.push_back(idx(i + 1, j + 1));
            F.push_back(idx(i,     j));
            F.push_back(idx(i + 1, j + 1));
            F.push_back(idx(i,     j + 1));
        }
    }
}

int main() try {
    std::cout << "[viz_primitives] starting" << std::endl;

    // ── 1. Face frames on icosahedron ────────────────────────
    {
        std::vector<double> V; std::vector<int32_t> F;
        buildIcosahedron(V, F);
        int nV = static_cast<int>(V.size() / 3);
        int nF = static_cast<int>(F.size() / 3);

        ComputeContext ctx(V.data(), nV, F.data(), nF);
        auto frames = computeFaceFrames(ctx);

        REQUIRE(frames.e1.rows()      == nF, "e1 row count");
        REQUIRE(frames.e2.rows()      == nF, "e2 row count");
        REQUIRE(frames.normals.rows() == nF, "normals row count");

        // For each face: check orthonormality {e1, e2, n}.
        double maxOrthErr = 0.0;
        double maxNormErr = 0.0;
        double maxRightHandErr = 0.0;
        for (int fi = 0; fi < nF; fi++) {
            Eigen::Vector3d e1 = frames.e1.row(fi).transpose();
            Eigen::Vector3d e2 = frames.e2.row(fi).transpose();
            Eigen::Vector3d n  = frames.normals.row(fi).transpose();

            maxOrthErr = std::max(maxOrthErr, std::abs(e1.dot(e2)));
            maxOrthErr = std::max(maxOrthErr, std::abs(e1.dot(n)));
            maxOrthErr = std::max(maxOrthErr, std::abs(e2.dot(n)));

            maxNormErr = std::max(maxNormErr, std::abs(e1.norm() - 1.0));
            maxNormErr = std::max(maxNormErr, std::abs(e2.norm() - 1.0));
            maxNormErr = std::max(maxNormErr, std::abs(n.norm()  - 1.0));

            // Right-handed: e1 × e2 ≈ n.
            Eigen::Vector3d crossE = e1.cross(e2);
            maxRightHandErr = std::max(maxRightHandErr, (crossE - n).norm());
        }
        REQUIRE(maxOrthErr      < 1e-12, "frames are orthogonal");
        REQUIRE(maxNormErr      < 1e-12, "frames are unit-length");
        REQUIRE(maxRightHandErr < 1e-12, "{e1, e2, n} is right-handed");
        std::cout << "  [1] face frames on icosahedron ✓ (max|e1·e2|="
                  << maxOrthErr << ")" << std::endl;
    }

    // ── 2. Geodesic path on icosahedron ──────────────────────
    {
        std::vector<double> V; std::vector<int32_t> F;
        buildIcosahedron(V, F);
        int nV = static_cast<int>(V.size() / 3);
        int nF = static_cast<int>(F.size() / 3);

        ComputeContext ctx(V.data(), nV, F.data(), nF);

        // First sanity: adjacent vertices (0 → 1, share an edge).
        std::cout << "  [2a] tracing 0→1 (adjacent)..." << std::endl;
        auto pathAdj = tracePath(ctx, 0, 1);
        REQUIRE(pathAdj.rows() >= 2, "adjacent path has at least 2 points");
        std::cout << "  [2a] adjacent path ✓ (" << pathAdj.rows() << " points)"
                  << std::endl;

        // Vertices 0 and 3 are antipodal on the unit sphere. The
        // continuous geodesic between antipodes on the sphere has
        // length π. On the piecewise-flat icosahedron *inscribed*
        // in the sphere, the discrete geodesic is shorter (paths
        // cut across flat faces rather than tracing arcs). For this
        // 12-vertex mesh we expect length in (chord=2, arc=π) ≈
        // (2.0, 3.14). The flip-out result is ~2.78, comfortably
        // inside that envelope.
        auto path = tracePath(ctx, 0, 3);

        REQUIRE(path.rows() >= 2, "path has at least 2 points");
        REQUIRE(path.cols() == 3, "path is 3D");

        double pathLength = 0.0;
        for (int i = 0; i + 1 < path.rows(); i++) {
            pathLength += (path.row(i + 1) - path.row(i)).norm();
        }
        std::cout << "  [2] geodesic path antipodes: length="
                  << pathLength << " (chord=2.0, sphere-arc=π≈"
                  << M_PI << ")" << std::endl;
        REQUIRE(pathLength > 2.0,
            "discrete geodesic must exceed the straight-line chord");
        REQUIRE(pathLength < M_PI,
            "piecewise-flat geodesic must be shorter than the sphere arc");

        // Endpoints should match the fixture vertex positions.
        Eigen::Vector3d v0 = path.row(0).transpose();
        Eigen::Vector3d v3 = path.row(path.rows() - 1).transpose();
        Eigen::Vector3d expected0(V[0],   V[1],   V[2]);
        Eigen::Vector3d expected3(V[9],   V[10],  V[11]);
        REQUIRE((v0 - expected0).norm() < 1e-9, "path starts at vStart");
        REQUIRE((v3 - expected3).norm() < 1e-9, "path ends at vEnd");
        std::cout << "  [2b] path endpoints land on the fixture vertices ✓"
                  << std::endl;
    }

    // ── 3. BFF parametrization on a flat grid ────────────────
    //
    // The grid has a single boundary loop (the square perimeter).
    // BFF should produce UV coordinates that respect the natural
    // (i, j) layout — no degenerate triangles, finite values.
    {
        const int nGrid = 5;
        std::vector<double> V; std::vector<int32_t> F;
        buildGrid(nGrid, V, F);
        int nV = static_cast<int>(V.size() / 3);
        int nF = static_cast<int>(F.size() / 3);
        std::cout << "  grid: nV=" << nV << " nF=" << nF << std::endl;

        ComputeContext ctx(V.data(), nV, F.data(), nF);
        auto uvs = computeUVCoordinates(ctx);

        REQUIRE(uvs.rows() == nV, "UV row count");
        REQUIRE(uvs.cols() == 2,  "UV is 2D");

        // All UVs should be finite.
        for (int i = 0; i < nV; i++) {
            REQUIRE(std::isfinite(uvs(i, 0)), "u finite");
            REQUIRE(std::isfinite(uvs(i, 1)), "v finite");
        }

        // BBox of UVs should be non-degenerate.
        double uMin = uvs.col(0).minCoeff(), uMax = uvs.col(0).maxCoeff();
        double vMin = uvs.col(1).minCoeff(), vMax = uvs.col(1).maxCoeff();
        REQUIRE((uMax - uMin) > 1e-3, "u range non-degenerate");
        REQUIRE((vMax - vMin) > 1e-3, "v range non-degenerate");
        std::cout << "  [3] BFF UVs on flat grid ✓ (u∈["
                  << uMin << "," << uMax << "], v∈[" << vMin << "," << vMax << "])"
                  << std::endl;

        // Closed mesh: should error with a clear message.
        std::vector<double> Vico; std::vector<int32_t> Fico;
        buildIcosahedron(Vico, Fico);
        ComputeContext ctxClosed(Vico.data(), 12, Fico.data(), 20);
        bool threw = false;
        try {
            (void)computeUVCoordinates(ctxClosed);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        REQUIRE(threw, "BFF on closed mesh should throw");
        std::cout << "  [3b] BFF rejects closed mesh ✓" << std::endl;
    }

    std::cout << "[viz_primitives] all assertions passed ✓" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "[viz_primitives] UNCAUGHT EXCEPTION: " << e.what() << std::endl;
    return 1;
} catch (...) {
    std::cerr << "[viz_primitives] UNKNOWN EXCEPTION" << std::endl;
    return 1;
}
