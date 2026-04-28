#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"

#include <iostream>
#include <stdexcept>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Geodesic path via the flip-out algorithm (Sharp & Crane 2020).
//
// 1. Initialize a coarse path from vStart to vEnd via Dijkstra
//    over mesh edges (gives a sequence of halfedges).
// 2. iterativeShorten() flips edges around each interior vertex
//    of the path until every wedge is locally straight (locally
//    a geodesic).
// 3. getPathPolyline3D() expands the intrinsic edge sequence into
//    a 3D polyline lying on the original mesh.
Eigen::MatrixXd tracePath(ComputeContext& ctx, int vStart, int vEnd) {
    auto& mesh = ctx.mesh();
    auto& geom = ctx.geometry();

    int nV = ctx.nV();
    if (vStart < 0 || vStart >= nV || vEnd < 0 || vEnd >= nV) {
        throw Error(ErrorCode::InvalidInput,
            "tracePath: vertex index out of range (nV=" + std::to_string(nV) + ")");
    }

    if (vStart == vEnd) {
        // Degenerate: zero-length path is the single point itself.
        Eigen::MatrixXd out(1, 3);
        Vector3 p = geom.inputVertexPositions[mesh.vertex(vStart)];
        out(0, 0) = p.x; out(0, 1) = p.y; out(0, 2) = p.z;
        return out;
    }

    // FlipEdgeNetwork's SignpostIntrinsicTriangulation needs intrinsic
    // edge lengths + corner angles to operate. They're derivable from
    // VertexPositionGeometry but must be `required` before use.
    geom.requireEdgeLengths();
    geom.requireCornerAngles();

    auto network = FlipEdgeNetwork::constructFromDijkstraPath(
        mesh, geom,
        mesh.vertex(vStart),
        mesh.vertex(vEnd));

    if (!network) {
        throw Error(ErrorCode::InvalidInput,
            "tracePath: failed to initialize Dijkstra path",
            "vStart and vEnd may not be in the same connected component");
    }

    // FlipEdgeNetwork carries an intrinsic-mesh path; getPathPolyline3D()
    // needs the embedded VertexPositionGeometry to lift it back to 3D.
    network->posGeom = &geom;

    network->iterativeShorten();

    auto polylines = network->getPathPolyline3D();
    if (polylines.empty() || polylines[0].empty()) {
        throw Error(ErrorCode::InternalError,
            "tracePath: flip-out produced an empty path");
    }
    const auto& path = polylines[0];

    Eigen::MatrixXd out(path.size(), 3);
    for (std::size_t i = 0; i < path.size(); i++) {
        out(static_cast<int>(i), 0) = path[i].x;
        out(static_cast<int>(i), 1) = path[i].y;
        out(static_cast<int>(i), 2) = path[i].z;
    }

    std::cout << "[geodesic_path] " << path.size()
              << " points, length=" << network->length() << std::endl;
    return out;
}

} // namespace nxr::compute
