#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <complex>
#include <cmath>

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

// Build q from the validated principal-curvature data. We take the magnitude
// from (κmax − κmin)/2 and the phase from the SAME 3D max-principal-direction
// the existing curvatures() lift produces, projected back into the (e1,e2)
// tangent basis. Going through the 3D direction makes this independent of
// geometry-central's internal 2-RoSy power convention.
VertexCurvature2RoSy vertexCurvature(Manifold& m) {
    CurvatureResult cr = curvatures(m);
    VertexFrames    vf = vertexFrames(m);
    int nV = m.nV();

    VertexCurvature2RoSy out;
    out.deviatoric.resize(nV);
    out.mean.resize(nV);
    for (int v = 0; v < nV; ++v) {
        Eigen::RowVector3d dir = cr.principalDirMax.row(v);
        Eigen::RowVector3d e1  = vf.e1.row(v);
        Eigen::RowVector3d e2  = vf.e2.row(v);
        double a = dir.dot(e1);
        double b = dir.dot(e2);
        double theta = std::atan2(b, a);
        double mag   = 0.5 * std::abs(cr.kMax(v) - cr.kMin(v));
        out.deviatoric(v) = std::polar(mag, 2.0 * theta);
        out.mean(v)       = 0.5 * (cr.kMax(v) + cr.kMin(v));
    }
    return out;
}

} // namespace nxr::manifold::geometry
