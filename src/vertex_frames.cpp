#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>

namespace nxr::manifold::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// See VertexFrames in compute.h for the gauge convention. This is exactly
// geometry-central's vertexTangentBasis (basisX, basisY = n × basisX) plus the
// vertex normal, so it matches the tangent space the connection Laplacian uses.
VertexFrames vertexFrames(Manifold& m) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();

    geom.requireVertexTangentBasis();
    geom.requireVertexNormals();

    int nV = m.nV();
    VertexFrames out;
    out.e1.resize(nV, 3);
    out.e2.resize(nV, 3);
    out.normals.resize(nV, 3);

    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(v.getIndex());
        Vector3 b0 = geom.vertexTangentBasis[v][0];
        Vector3 b1 = geom.vertexTangentBasis[v][1];
        Vector3 N  = geom.vertexNormals[v];
        out.e1     (vi, 0) = b0.x; out.e1     (vi, 1) = b0.y; out.e1     (vi, 2) = b0.z;
        out.e2     (vi, 0) = b1.x; out.e2     (vi, 1) = b1.y; out.e2     (vi, 2) = b1.z;
        out.normals(vi, 0) = N.x;  out.normals(vi, 1) = N.y;  out.normals(vi, 2) = N.z;
    }

    std::cout << "[vertex_frames] " << nV << " vertex frames computed" << std::endl;
    return out;
}

} // namespace nxr::manifold::geometry
