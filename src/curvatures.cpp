#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>
#include <cmath>

namespace nxr::manifold::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

CurvatureResult curvatures(Manifold& m) {
    auto& mesh = m.mesh();
    auto& geometry = m.geometry();
    int nV = m.nV();

    CurvatureResult result;
    result.gaussian.resize(nV);
    result.mean.resize(nV);
    result.kMin.resize(nV);
    result.kMax.resize(nV);
    result.principalDirMax.resize(nV, 3);
    result.principalDirMax.setZero();

    // ── Gaussian curvature (always available) ───────────────
    geometry.requireVertexGaussianCurvatures();
    for (Vertex v : mesh.vertices()) {
        result.gaussian(static_cast<int>(v.getIndex())) = geometry.vertexGaussianCurvatures[v];
    }

    // ── Mean curvature ──────────────────────────────────────
    try {
        geometry.requireVertexMeanCurvatures();
        for (Vertex v : mesh.vertices()) {
            result.mean(static_cast<int>(v.getIndex())) = geometry.vertexMeanCurvatures[v];
        }
    } catch (const std::exception& e) {
        std::cerr << "[curvatures] Mean curvature unavailable: " << e.what() << std::endl;
        result.mean.setZero();
    }

    // ── Principal curvatures (min, max) ─────────────────────
    try {
        geometry.requireVertexMinPrincipalCurvatures();
        geometry.requireVertexMaxPrincipalCurvatures();
        for (Vertex v : mesh.vertices()) {
            int i = static_cast<int>(v.getIndex());
            result.kMin(i) = geometry.vertexMinPrincipalCurvatures[v];
            result.kMax(i) = geometry.vertexMaxPrincipalCurvatures[v];
        }
    } catch (const std::exception& e) {
        std::cerr << "[curvatures] Principal curvatures unavailable: " << e.what() << std::endl;
        result.kMin.setZero();
        result.kMax.setZero();
    }

    // ── Principal direction (lifted to 3D) ──────────────────
    // Only available for manifold meshes and requires tangent basis; may fail.
    try {
        geometry.requireVertexPrincipalCurvatureDirections();
        geometry.requireVertexTangentBasis();
        for (Vertex v : mesh.vertices()) {
            int i = static_cast<int>(v.getIndex());
            Vector2 pd = geometry.vertexPrincipalCurvatureDirections[v];
            // 4-symmetric: unwrap with atan2 / 4
            double theta = std::atan2(pd.y, pd.x) / 4.0;
            const auto& basis = geometry.vertexTangentBasis[v];
            Vector3 dir3 = basis[0] * std::cos(theta) + basis[1] * std::sin(theta);
            result.principalDirMax(i, 0) = dir3.x;
            result.principalDirMax(i, 1) = dir3.y;
            result.principalDirMax(i, 2) = dir3.z;
        }
    } catch (const std::exception& e) {
        std::cerr << "[curvatures] Principal directions unavailable: " << e.what() << std::endl;
    }

    std::cout << "[curvatures] Computed: K range ["
              << result.gaussian.minCoeff() << ", " << result.gaussian.maxCoeff() << "], H range ["
              << result.mean.minCoeff() << ", " << result.mean.maxCoeff() << "]" << std::endl;

    return result;
}

} // namespace nxr::manifold::geometry