#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"

#include <stdexcept>

namespace nxr::manifold::query {

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
GeodesicPath tracePathSurface(Manifold& m, int vStart, int vEnd) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();

    int nV = m.nV();
    if (vStart < 0 || vStart >= nV || vEnd < 0 || vEnd >= nV) {
        throw Error(ErrorCode::InvalidInput,
            "tracePath: vertex index out of range (nV=" + std::to_string(nV) + ")");
    }

    GeodesicPath out;
    if (vStart == vEnd) {
        // Degenerate: zero-length path is the single point itself.
        Vector3 p = geom.inputVertexPositions[mesh.vertex(vStart)];
        out.positions.resize(1, 3);
        out.positions << p.x, p.y, p.z;
        out.vertices.setConstant(1, 3, vStart);
        out.weights.setZero(1, 3);
        out.weights(0, 0) = 1.0;
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

    network->posGeom = &geom;
    network->iterativeShorten();

    // The intrinsic path traced back onto the INPUT mesh: each point a
    // vertex, an edge crossing or (rarely) a face point.
    auto polylines = network->getPathPolyline();
    if (polylines.empty() || polylines[0].empty()) {
        throw Error(ErrorCode::InternalError,
            "tracePath: flip-out produced an empty path");
    }
    const auto& path = polylines[0];
    const int n = static_cast<int>(path.size());
    out.positions.resize(n, 3);
    out.vertices.resize(n, 3);
    out.weights.setZero(n, 3);
    for (int i = 0; i < n; i++) {
        const SurfacePoint& sp = path[i];
        Vector3 p = sp.interpolate(geom.inputVertexPositions);
        out.positions(i, 0) = p.x; out.positions(i, 1) = p.y; out.positions(i, 2) = p.z;
        switch (sp.type) {
            case SurfacePointType::Vertex: {
                int v = static_cast<int>(sp.vertex.getIndex());
                out.vertices.row(i) << v, v, v;
                out.weights(i, 0) = 1.0;
                break;
            }
            case SurfacePointType::Edge: {
                int a = static_cast<int>(sp.edge.halfedge().tailVertex().getIndex());
                int b = static_cast<int>(sp.edge.halfedge().tipVertex().getIndex());
                out.vertices.row(i) << a, b, a;
                out.weights(i, 0) = 1.0 - sp.tEdge;
                out.weights(i, 1) = sp.tEdge;
                break;
            }
            case SurfacePointType::Face: {
                Halfedge he = sp.face.halfedge();
                // faceCoords follow the face's halfedge order
                int a = static_cast<int>(he.vertex().getIndex());
                int b = static_cast<int>(he.next().vertex().getIndex());
                int c = static_cast<int>(he.next().next().vertex().getIndex());
                out.vertices.row(i) << a, b, c;
                out.weights(i, 0) = sp.faceCoords.x;
                out.weights(i, 1) = sp.faceCoords.y;
                out.weights(i, 2) = sp.faceCoords.z;
                break;
            }
        }
    }
    out.length = network->length();
    return out;
}

Eigen::MatrixXd tracePath(Manifold& m, int vStart, int vEnd) {
    return tracePathSurface(m, vStart, vEnd).positions;
}

} // namespace nxr::manifold::query