#include "nxr/compute.h"
#include <Eigen/Geometry>
#include <iostream>
#include <cmath>
#include <complex>

using nxr::manifold::Manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { \
    std::cerr << "  [FAIL] " << msg << "\n"; ++g_failures; } \
    else { std::cout << "  [PASS] " << msg << "\n"; } } while (0)

// Unit icosahedron (12 verts, 20 faces) — same fixture as the MATLAB tests.
static void makeIcosahedron(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double verts[12][3] = {
        {-1, t, 0},{1, t, 0},{-1,-t, 0},{1,-t, 0},
        {0,-1, t},{0, 1, t},{0,-1,-t},{0, 1,-t},
        {t, 0,-1},{t, 0, 1},{-t, 0,-1},{-t, 0, 1}};
    for (auto& r : verts) {
        double n = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        r[0]/=n; r[1]/=n; r[2]/=n;
    }
    V.assign(&verts[0][0], &verts[0][0]+36);
    int faces[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
    F.assign(&faces[0][0], &faces[0][0]+60);
}

static void testVertexGrid() {
    std::cout << "\n=== vertexGrid ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    Eigen::MatrixXcd c = nxr::manifold::geometry::vertexGrid(m);
    EXPECT(c.rows() == 12 && c.cols() == 3, "vertexGrid is [nV, 3]");

    nxr::manifold::geometry::VertexFrames vf = nxr::manifold::geometry::vertexFrames(m);

    double maxE1Err = 0, maxE2Err = 0, maxDot = 0, maxCrossErr = 0;
    for (int v = 0; v < 12; ++v) {
        Eigen::Vector3d e1 = c.row(v).real(), e2 = c.row(v).imag();
        maxE1Err = std::max(maxE1Err, std::abs(e1.norm() - 1.0));
        maxE2Err = std::max(maxE2Err, std::abs(e2.norm() - 1.0));
        maxDot   = std::max(maxDot, std::abs(e1.dot(e2)));
        Eigen::Vector3d n_gc(vf.normals(v,0), vf.normals(v,1), vf.normals(v,2));
        maxCrossErr = std::max(maxCrossErr, (e1.cross(e2) - n_gc).norm());
    }
    EXPECT(maxE1Err < 1e-9, "real(c) unit length");
    EXPECT(maxE2Err < 1e-9, "imag(c) unit length");
    EXPECT(maxDot   < 1e-9, "real(c) ⟂ imag(c)");
    EXPECT(maxCrossErr < 1e-9, "real(c) × imag(c) matches GC vertex normal");
}

static void testFaceGrid() {
    std::cout << "\n=== faceGrid ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    Eigen::MatrixXcd c = nxr::manifold::geometry::faceGrid(m);
    EXPECT(c.rows() == 20 && c.cols() == 3, "faceGrid is [nF, 3]");

    nxr::manifold::geometry::FaceFrames ff = nxr::manifold::geometry::frames(m);

    double maxE1Err = 0, maxE2Err = 0, maxDot = 0, maxCrossErr = 0;
    for (int f = 0; f < 20; ++f) {
        Eigen::Vector3d e1 = c.row(f).real(), e2 = c.row(f).imag();
        maxE1Err = std::max(maxE1Err, std::abs(e1.norm() - 1.0));
        maxE2Err = std::max(maxE2Err, std::abs(e2.norm() - 1.0));
        maxDot   = std::max(maxDot, std::abs(e1.dot(e2)));
        Eigen::Vector3d n_gc(ff.normals(f,0), ff.normals(f,1), ff.normals(f,2));
        maxCrossErr = std::max(maxCrossErr, (e1.cross(e2) - n_gc).norm());
    }
    EXPECT(maxE1Err < 1e-9, "real(c) unit length");
    EXPECT(maxE2Err < 1e-9, "imag(c) unit length");
    EXPECT(maxDot   < 1e-9, "real(c) ⟂ imag(c)");
    EXPECT(maxCrossErr < 1e-9, "real(c) × imag(c) matches GC face normal");
}

static void testVertexCurvature() {
    std::cout << "\n=== vertexCurvature ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    auto cv = nxr::manifold::geometry::vertexCurvature(m);
    EXPECT(cv.deviatoric.size() == 12, "deviatoric length == nV");
    EXPECT(cv.mean.size() == 12, "mean length == nV");

    // Unit icosahedron approximates a unit sphere: nearly umbilic, so the
    // deviatoric magnitude is small relative to the mean, and the mean
    // (≈ principal curvature ≈ 1/R = 1) is positive (convex).
    double maxDev = cv.deviatoric.cwiseAbs().maxCoeff();
    double minMean = cv.mean.minCoeff();
    EXPECT(minMean > 0.5, "mean curvature positive (convex), ~1 on unit sphere");
    EXPECT(maxDev < 0.5 * minMean, "deviatoric small vs mean (near-umbilic)");

    // Extrinsic Gaussian K = H² − |q|² must be positive on a convex surface.
    bool allKpos = true;
    for (int v = 0; v < 12; ++v) {
        double K = cv.mean(v) * cv.mean(v) - std::norm(cv.deviatoric(v));
        if (K <= 0) allKpos = false;
    }
    EXPECT(allKpos, "extrinsic Gaussian H² − |q|² > 0 (convex)");

    // Determinism.
    auto cv2 = nxr::manifold::geometry::vertexCurvature(m);
    EXPECT((cv.deviatoric - cv2.deviatoric).cwiseAbs().maxCoeff() < 1e-15 &&
           (cv.mean - cv2.mean).cwiseAbs().maxCoeff() < 1e-15,
           "vertexCurvature deterministic");
}

static void testTrivialGaugeRotations() {
    std::cout << "\n=== integrateTrivialGaugeRotations ===\n";
    std::vector<double> V; std::vector<int32_t> F; makeIcosahedron(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    auto dec   = nxr::manifold::ops::assembleDECOperators(m);
    auto cache = nxr::manifold::ops::CholeskyCache{};
    // χ(sphere) = 2: two index-1 singularities (Gauss-Bonnet satisfied).
    std::map<int, double> sing = {{0, 1.0}, {1, 1.0}};

    auto g = nxr::manifold::connection::integrateTrivialGaugeRotations(m, dec, cache, sing);
    EXPECT(g.vertex.size() == 12, "rotation length == nV");

    double maxModErr = (g.vertex.cwiseAbs().array() - 1.0).abs().maxCoeff();
    EXPECT(maxModErr < 1e-9, "all rotations unit modulus");

    double spread = (g.vertex.array() - g.vertex(0)).abs().maxCoeff();
    EXPECT(spread > 1e-6, "trivial gauge differs from Levi-Civita");

    auto g2 = nxr::manifold::connection::integrateTrivialGaugeRotations(m, dec, cache, sing);
    EXPECT((g.vertex - g2.vertex).cwiseAbs().maxCoeff() < 1e-15,
           "rotation field deterministic");
}

int main() {
    testVertexGrid();
    testFaceGrid();
    testVertexCurvature();
    testTrivialGaugeRotations();
    if (g_failures) { std::cerr << "\n" << g_failures << " failure(s)\n"; return 1; }
    std::cout << "\nALL PASSED\n"; return 0;
}
