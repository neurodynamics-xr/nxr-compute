#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/heat_method_distance.h"

#include <iostream>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

Eigen::VectorXd computeGeodesicDistance(
    ComputeContext& ctx,
    const std::vector<int>& sourceVertices
) {
    auto& mesh = ctx.mesh();
    auto& geometry = ctx.geometry();

    // Build source vertex list for geometry-central
    std::vector<Vertex> sources;
    sources.reserve(sourceVertices.size());
    for (int idx : sourceVertices) {
        sources.push_back(mesh.vertex(static_cast<size_t>(idx)));
    }

    // Solve via heat method
    HeatMethodDistanceSolver solver(geometry);
    VertexData<double> distances = solver.computeDistance(sources);

    // Convert to Eigen::VectorXd indexed by vertex index
    int nV = ctx.nV();
    Eigen::VectorXd result(nV);
    for (Vertex v : mesh.vertices()) {
        result(static_cast<int>(v.getIndex())) = distances[v];
    }

    std::cout << "[geodesic] Computed distances from " << sourceVertices.size()
              << " sources. Max distance: " << result.maxCoeff() << std::endl;

    return result;
}

} // namespace nxr::compute
