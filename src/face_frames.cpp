#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Per-face orthonormal frame:
//   e1 = unit vector along the face's first halfedge
//   n  = face normal
//   e2 = n × e1   (so {e1, e2, n} is right-handed)
//
// Convention matches direction_field.cpp::faceOrthonormalBasis, so
// tangent vectors transported via the trivial-connection direction
// field are interpretable in the same coordinate system.
FaceFrames computeFaceFrames(ComputeContext& ctx) {
    auto& mesh = ctx.mesh();
    auto& geom = ctx.geometry();

    geom.requireFaceNormals();

    int nF = ctx.nF();
    FaceFrames out;
    out.e1.resize(nF, 3);
    out.e2.resize(nF, 3);
    out.normals.resize(nF, 3);

    for (Face f : mesh.faces()) {
        int fi = static_cast<int>(f.getIndex());

        Halfedge h = f.halfedge();
        Vector3 e0 = geom.inputVertexPositions[h.tipVertex()] -
                     geom.inputVertexPositions[h.tailVertex()];
        e0 = e0.normalize();

        Vector3 N  = geom.faceNormals[f];
        Vector3 e1 = cross(N, e0).normalize();

        out.e1     (fi, 0) = e0.x;  out.e1     (fi, 1) = e0.y;  out.e1     (fi, 2) = e0.z;
        out.e2     (fi, 0) = e1.x;  out.e2     (fi, 1) = e1.y;  out.e2     (fi, 2) = e1.z;
        out.normals(fi, 0) = N.x;   out.normals(fi, 1) = N.y;   out.normals(fi, 2) = N.z;
    }

    std::cout << "[face_frames] " << nF << " face frames computed" << std::endl;
    return out;
}

} // namespace nxr::compute
