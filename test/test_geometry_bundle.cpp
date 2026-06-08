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

    double maxE1Err = 0, maxE2Err = 0, maxDot = 0, maxCrossErr = 0;
    for (int v = 0; v < 12; ++v) {
        Eigen::Vector3d e1 = c.row(v).real(), e2 = c.row(v).imag();
        maxE1Err = std::max(maxE1Err, std::abs(e1.norm() - 1.0));
        maxE2Err = std::max(maxE2Err, std::abs(e2.norm() - 1.0));
        maxDot   = std::max(maxDot, std::abs(e1.dot(e2)));
        maxCrossErr = std::max(maxCrossErr, std::abs(e1.cross(e2).norm() - 1.0));
    }
    EXPECT(maxE1Err < 1e-9, "real(c) unit length");
    EXPECT(maxE2Err < 1e-9, "imag(c) unit length");
    EXPECT(maxDot   < 1e-9, "real(c) ⟂ imag(c)");
    EXPECT(maxCrossErr < 1e-9, "real(c) × imag(c) is unit normal");
}

int main() {
    testVertexGrid();
    if (g_failures) { std::cerr << "\n" << g_failures << " failure(s)\n"; return 1; }
    std::cout << "\nALL PASSED\n"; return 0;
}
