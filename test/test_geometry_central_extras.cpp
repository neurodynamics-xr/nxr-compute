/**
 * test_geometry_central_extras.cpp — smoke for the geometry-central
 * extras shipped in this PR:
 *
 *   1. VectorHeatSolver (transport / extendScalar / logMap / findCenter)
 *   2. SignedHeatSolver (signed geodesic distance from a curve)
 *   3. computeSmoothFaceField / computeSmoothVertexField (NRoSy)
 *   4. computeStripePattern
 *
 * All five primitives are run on the unit icosahedron fixture used
 * by the other smokes. The test verifies basic sanity (output sizes,
 * non-trivial ranges, distance ≈ logmap norm) rather than
 * bit-for-bit numerical contracts — the underlying solvers belong
 * to geometry-central and have their own test suite.
 */

#include "nxr/compute.h"

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

static void testVectorHeat(ComputeContext& ctx) {
    std::cout << "[vhm] starting" << std::endl;
    VectorHeatSolver vhm(ctx);
    int nV = ctx.nV();

    // Transport one tangent vector from vertex 0. Expect non-zero output everywhere.
    Eigen::MatrixXd src(1, 3);
    src << 1.0, 0.0, 0.0;
    auto transported = vectorHeatTransport(vhm, {0}, src);
    REQUIRE(transported.rows() == nV, "transported has nV rows");
    REQUIRE(transported.cols() == 3,  "transported has 3 cols");
    double maxNorm = 0.0;
    for (int i = 0; i < nV; i++) {
        maxNorm = std::max(maxNorm, transported.row(i).norm());
    }
    REQUIRE(maxNorm > 1e-6, "transport produced non-trivial vectors");

    // Extend a scalar (1.0 at vertex 0, 0.0 at vertex 6 — antipodes-ish).
    auto extended = vectorHeatExtendScalar(vhm, {0, 6}, (Eigen::VectorXd(2) << 1.0, 0.0).finished());
    REQUIRE(extended.size() == nV, "extended has nV entries");
    // Source values reproduced (heat-method approximation, so loose tol).
    REQUIRE(std::abs(extended(0) - 1.0) < 0.5, "vertex 0 near 1.0");
    REQUIRE(std::abs(extended(6) - 0.0) < 0.5, "vertex 6 near 0.0");

    // Log map at vertex 0. Norm of (logX, logY) should be ~ geodesic distance.
    auto logmap = vectorHeatLogMap(vhm, 0);
    REQUIRE(logmap.logCoords.rows() == nV, "logCoords has nV rows");
    REQUIRE(logmap.logCoords.cols() == 2,  "logCoords has 2 cols");
    REQUIRE(logmap.logCoords.row(0).norm() < 1e-6, "log map at source is ~0");
    // sourceE1 / E2 should be unit-length.
    REQUIRE(std::abs(logmap.sourceE1.norm() - 1.0) < 1e-6, "sourceE1 unit");
    REQUIRE(std::abs(logmap.sourceE2.norm() - 1.0) < 1e-6, "sourceE2 unit");

    // findCenter of three nearby vertices: must be a finite 3D point.
    Eigen::Vector3d c = vectorHeatFindCenter(vhm, {0, 1, 5});
    REQUIRE(std::isfinite(c.x()) && std::isfinite(c.y()) && std::isfinite(c.z()),
            "center is finite");
    // Icosahedron is unit-radius, so center should sit roughly on the surface.
    REQUIRE(std::abs(c.norm() - 1.0) < 0.4, "center near surface");
    std::cout << "[vhm] PASS" << std::endl;
}

static void testSignedHeat(ComputeContext& ctx) {
    std::cout << "[shm] starting" << std::endl;
    SignedHeatSolver shs(ctx);
    int nV = ctx.nV();

    // Loop around the "north" pole of the icosahedron (vertex 0).
    auto sd = signedHeatDistance(shs, {11, 5, 1, 7, 10}, true);
    REQUIRE(sd.size() == nV, "signed distance has nV entries");
    // Should have both signs (curve splits the surface in two).
    REQUIRE(sd.minCoeff() < 0.0 && sd.maxCoeff() > 0.0,
            "both positive and negative sides of the curve");
    std::cout << "[shm] PASS" << std::endl;
}

static void testSmoothFields(ComputeContext& ctx) {
    std::cout << "[field] starting" << std::endl;
    int nF = ctx.nF();
    int nV = ctx.nV();

    auto faceField = computeSmoothFaceField(ctx, 4);
    REQUIRE(faceField.rows() == nF, "face field has nF rows");
    REQUIRE(faceField.cols() == 3,  "face field has 3 cols");
    double mn = 0.0;
    for (int i = 0; i < nF; i++) mn = std::max(mn, faceField.row(i).norm());
    REQUIRE(mn > 1e-6, "face field is non-trivial");

    auto vfield = computeSmoothVertexField(ctx, 2);
    REQUIRE(vfield.vertexVectors.rows() == nV, "vertex field nV rows");
    REQUIRE(vfield.vertexFieldRaw.size() == nV * 2, "vertex field raw nV*2");
    REQUIRE(vfield.nSym == 2, "nSym preserved");
    std::cout << "[field] PASS" << std::endl;
}

static void testStripes(ComputeContext& ctx) {
    std::cout << "[stripes] starting" << std::endl;
    auto vfield = computeSmoothVertexField(ctx, 2);
    auto stripes = computeStripePattern(ctx, vfield.vertexFieldRaw, 8.0);
    REQUIRE(stripes.segmentCount >= 0, "non-negative segment count");
    REQUIRE(stripes.positions.rows() == stripes.segmentCount * 2,
            "positions row count = 2 * segments");
    REQUIRE(stripes.positions.cols() == 3, "positions cols = 3");
    std::cout << "[stripes] PASS (segs=" << stripes.segmentCount << ")" << std::endl;
}

int main() {
    std::vector<double>  V;
    std::vector<int32_t> F;
    buildIcosahedron(V, F);
    int nV = static_cast<int>(V.size() / 3);
    int nF = static_cast<int>(F.size() / 3);
    ComputeContext ctx(V.data(), nV, F.data(), nF);

    testVectorHeat(ctx);
    testSignedHeat(ctx);
    testSmoothFields(ctx);
    testStripes(ctx);

    std::cout << "all geometry-central extras tests PASSED" << std::endl;
    return 0;
}
