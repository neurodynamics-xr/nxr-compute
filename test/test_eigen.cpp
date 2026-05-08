/**
 * test_eigen.cpp — Standalone test for the cortical compute library.
 *
 * Tests: operator assembly, eigensolver, normalization, DC removal,
 *        Poisson solve, geodesic distance, Hodge decomposition.
 *
 * Build: cmake --build build --target test_eigen
 * Run:   ./build/Release/test_eigen.exe
 */

#include "nxr/compute.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

// Generate a subdivided icosphere for a slightly larger test mesh
void generateIcosphere(std::vector<double>& vertices, std::vector<int32_t>& faces) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    std::vector<double> verts;
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        verts.push_back(raw[i] / len);
        verts.push_back(raw[i+1] / len);
        verts.push_back(raw[i+2] / len);
    }
    vertices = verts;
    faces = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Cortical Compute Library Test Suite" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::vector<double> vertices;
    std::vector<int32_t> faces;
    generateIcosphere(vertices, faces);
    int nV = static_cast<int>(vertices.size() / 3);
    int nF = static_cast<int>(faces.size() / 3);

    std::cout << "\nTest mesh: icosahedron (" << nV << " V, " << nF << " F)" << std::endl;

    // ── Create compute context + factor cache ────────────────
    nxr::compute::ComputeContext ctx(vertices.data(), nV, faces.data(), nF);
    nxr::compute::CholeskyCache cache;

    // ── Test 1: Operator assembly ────────────────────────────
    std::cout << "\n[Test 1] Operator Assembly" << std::endl;
    auto ops = nxr::compute::assembleMeshOperators(ctx);
    std::cout << "  Stiffness: " << ops.stiffness.nonZeros() << " nnz" << std::endl;
    std::cout << "  Total area: " << ops.totalArea << std::endl;

    // ── Test 2: DEC operators ────────────────────────────────
    std::cout << "\n[Test 2] DEC Operators" << std::endl;
    auto dec = nxr::compute::assembleDECOperators(ctx);
    std::cout << "  d0: " << dec.d0.rows() << "x" << dec.d0.cols() << " (" << dec.d0.nonZeros() << " nnz)" << std::endl;
    std::cout << "  d1: " << dec.d1.rows() << "x" << dec.d1.cols() << " (" << dec.d1.nonZeros() << " nnz)" << std::endl;
    std::cout << "  hodge1: " << dec.hodge1.rows() << "x" << dec.hodge1.cols() << std::endl;

    // ── Test 3: Eigensolver ──────────────────────────────────
    std::cout << "\n[Test 3] Eigensolver" << std::endl;
    int k = 6;
    auto result = nxr::compute::solveEigenmodes(ops.stiffness, ops.mass, k);
    std::cout << "  Eigenvalues:" << std::endl;
    for (int i = 0; i < result.k; i++) {
        std::cout << "    λ_" << i << " = " << std::setprecision(6) << result.eigenvalues(i) << std::endl;
    }

    // ── Test 4: Normalization ────────────────────────────────
    std::cout << "\n[Test 4] Normalization" << std::endl;
    auto U_norm = nxr::compute::normalizeEigenmodes(result.eigenvectors, ops.mass);
    result.eigenvectors = U_norm;

    // ── Test 5: DC removal ───────────────────────────────────
    std::cout << "\n[Test 5] DC Removal" << std::endl;
    auto noDC = nxr::compute::removeDC(result);
    std::cout << "  Modes remaining: " << noDC.k << std::endl;

    // ── Test 6: Poisson solver ───────────────────────────────
    std::cout << "\n[Test 6] Poisson Solver" << std::endl;
    std::map<int, double> densityMap;
    densityMap[0] = 1.0;   // source at vertex 0
    densityMap[3] = -1.0;  // sink at vertex 3 (opposite side of icosahedron)
    auto phi = nxr::compute::solvePoisson(ops, cache, densityMap);
    std::cout << "  φ at source (v=0): " << phi(0) << std::endl;
    std::cout << "  φ at sink (v=3):   " << phi(3) << std::endl;

    // ── Test 7: Geodesic distance ────────────────────────────
    // The geodesic API now takes a stateful HeatGeodesicSolver — the
    // factor pair (M+tA, A) is built at construction and reused across
    // calls. Mirrors the VectorHeatSolver / SignedHeatSolver pattern.
    std::cout << "\n[Test 7] Geodesic Distance" << std::endl;
    std::vector<int> sources = {0};
    nxr::compute::HeatGeodesicSolver heatGeo(ctx);
    auto dists = nxr::compute::computeGeodesicDistance(heatGeo, sources);
    std::cout << "  Distance at source (v=0): " << dists(0) << std::endl;
    std::cout << "  Distance at v=3:          " << dists(3) << std::endl;
    std::cout << "  Max distance:             " << dists.maxCoeff() << std::endl;

    // ── Test 8: Hodge decomposition ──────────────────────────
    std::cout << "\n[Test 8] Hodge Decomposition" << std::endl;
    auto omega = nxr::compute::generateRandomOmega(ctx.nE());
    auto hodge = nxr::compute::hodgeDecompose(ctx, dec, cache, omega);
    std::cout << "  ω norm:       " << omega.norm() << std::endl;
    std::cout << "  α norm:       " << hodge.exactPotential.norm() << std::endl;
    std::cout << "  β norm (V):   " << hodge.coExactPotentialV.norm() << std::endl;
    std::cout << "  dα 1-form:    " << hodge.dAlpha.norm() << std::endl;
    std::cout << "  δβ 1-form:    " << hodge.deltaBeta.norm() << std::endl;
    std::cout << "  γ 1-form:     " << hodge.gamma.norm() << std::endl;
    std::cout << "  ω vectors:    " << hodge.omegaVectors.rows() << " faces" << std::endl;

    // ── Test 9: Curvatures ───────────────────────────────────
    std::cout << "\n[Test 9] Curvatures" << std::endl;
    auto curv = nxr::compute::computeCurvatures(ctx);
    std::cout << "  K range: [" << curv.gaussian.minCoeff() << ", " << curv.gaussian.maxCoeff() << "]" << std::endl;
    std::cout << "  H range: [" << curv.mean.minCoeff() << ", " << curv.mean.maxCoeff() << "]" << std::endl;
    std::cout << "  κ_min range: [" << curv.kMin.minCoeff() << ", " << curv.kMin.maxCoeff() << "]" << std::endl;
    std::cout << "  κ_max range: [" << curv.kMax.minCoeff() << ", " << curv.kMax.maxCoeff() << "]" << std::endl;

    // ── Test 10: Vertex normals (all 6 variants) ─────────────
    std::cout << "\n[Test 10] Vertex Normals" << std::endl;
    auto n_angle = nxr::compute::computeVertexNormals(ctx, nxr::compute::NormalType::AngleWeighted);
    auto n_area = nxr::compute::computeVertexNormals(ctx, nxr::compute::NormalType::AreaWeighted);
    auto n_sphere = nxr::compute::computeVertexNormals(ctx, nxr::compute::NormalType::SphereInscribed);
    std::cout << "  First normal (angle):  (" << n_angle(0, 0) << ", " << n_angle(0, 1) << ", " << n_angle(0, 2) << ")" << std::endl;
    std::cout << "  First normal (area):   (" << n_area(0, 0) << ", " << n_area(0, 1) << ", " << n_area(0, 2) << ")" << std::endl;
    std::cout << "  First normal (sphere): (" << n_sphere(0, 0) << ", " << n_sphere(0, 1) << ", " << n_sphere(0, 2) << ")" << std::endl;

    // ── Test 11: Scalar gradient on Poisson solution ─────────
    std::cout << "\n[Test 11] Scalar Gradient" << std::endl;
    auto grad = nxr::compute::scalarGradient(ctx, phi);
    std::cout << "  Gradient norm at face 0: " << grad.row(0).norm() << std::endl;

    // ── Test 12: Isolines on geodesic distances ──────────────
    std::cout << "\n[Test 12] Isolines" << std::endl;
    auto iso = nxr::compute::computeIsolines(ctx, dists, 10);
    std::cout << "  Segment count: " << iso.segmentCount << std::endl;

    // ── Test 13: Direction field (icosahedron: χ=2) ──────────
    std::cout << "\n[Test 13] Direction Field" << std::endl;
    std::map<int, double> singularities;
    singularities[0] = 1.0;
    singularities[3] = 1.0;  // Antipodal vertices on icosahedron, sum = 2 = χ
    auto dirField = nxr::compute::computeDirectionField(ctx, dec, cache, singularities);
    std::cout << "  χ = " << dirField.eulerCharacteristic << std::endl;
    std::cout << "  Gauss-Bonnet: " << (dirField.gaussBonnetSatisfied ? "OK" : "FAIL") << std::endl;
    std::cout << "  Direction at face 0: (" << dirField.directionVectors(0, 0) << ", "
              << dirField.directionVectors(0, 1) << ", " << dirField.directionVectors(0, 2) << ")" << std::endl;

    // ── Test 14: Streamlines on the direction field ──────────
    std::cout << "\n[Test 14] Streamlines" << std::endl;
    auto streams = nxr::compute::traceStreamlines(ctx, dirField.directionVectors, 5, 0.2, 100);
    std::cout << "  Streamline segments: " << streams.segmentCount << std::endl;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "  All tests complete ✓" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
}
