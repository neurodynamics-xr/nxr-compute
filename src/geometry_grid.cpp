#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace nxr::manifold::geometry {

// c = e1 + i·e2, built directly on the existing frame assembly so the
// tangent basis matches everything else (connection Laplacian, transport).
Eigen::MatrixXcd vertexGrid(Manifold& m) {
    VertexFrames vf = vertexFrames(m);
    int nV = m.nV();
    Eigen::MatrixXcd c(nV, 3);
    for (int v = 0; v < nV; ++v)
        for (int k = 0; k < 3; ++k)
            c(v, k) = std::complex<double>(vf.e1(v, k), vf.e2(v, k));
    return c;
}

Eigen::MatrixXcd faceGrid(Manifold& m) {
    FaceFrames ff = frames(m);
    int nF = m.nF();
    Eigen::MatrixXcd c(nF, 3);
    for (int f = 0; f < nF; ++f)
        for (int k = 0; k < 3; ++k)
            c(f, k) = std::complex<double>(ff.e1(f, k), ff.e2(f, k));
    return c;
}

} // namespace nxr::manifold::geometry
