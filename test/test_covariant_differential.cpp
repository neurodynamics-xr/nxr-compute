#include "nxr/compute.h"
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testFrameTransport() {
    std::cout << "\n=== covariant-differential: frame transport ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    // orthogonality: P_ij^T P_ij = I (P is a product of rotations)
    Eigen::Matrix3d P = differential::frameTransport(m, 0, 3);
    EXPECT((P.transpose() * P - Eigen::Matrix3d::Identity()).norm() < 1e-12, "P_ij orthogonal");

    // flatness / path-independence: around face {0,11,5}, P_k i · P_j k · P_i j = I
    int i = 0, j = 11, k = 5;
    Eigen::Matrix3d loop = differential::frameTransport(m, k, i)
                         * differential::frameTransport(m, j, k)
                         * differential::frameTransport(m, i, j);
    EXPECT((loop - Eigen::Matrix3d::Identity()).norm() < 1e-12, "holonomy around triangle is identity (flat)");

    // self-transport is identity
    EXPECT((differential::frameTransport(m, 4, 4) - Eigen::Matrix3d::Identity()).norm() < 1e-12,
           "P_ii = I");

    // vertexFrameMatrices: [nV,9], each row an orthonormal frame
    Eigen::MatrixXd Fm = differential::vertexFrameMatrices(m);
    EXPECT(Fm.rows() == 12 && Fm.cols() == 9, "vertexFrameMatrices [12,9]");
    Eigen::Matrix3d F0;  // row 0 reshaped row-major
    F0 << Fm(0,0),Fm(0,1),Fm(0,2), Fm(0,3),Fm(0,4),Fm(0,5), Fm(0,6),Fm(0,7),Fm(0,8);
    EXPECT((F0.transpose()*F0 - Eigen::Matrix3d::Identity()).norm() < 1e-12, "Fv orthonormal");
}

int main() {
    testFrameTransport();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
