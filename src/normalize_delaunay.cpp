#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/remeshing.h"

namespace nxr::manifold {

using namespace geometrycentral;
using namespace geometrycentral::surface;

DelaunayNormalization fixDelaunay(
    const double* vertices, int nV, const int32_t* faces, int nF) {

    std::vector<std::vector<size_t>> polygons(nF);
    for (int i = 0; i < nF; ++i)
        polygons[i] = { static_cast<size_t>(faces[3*i]),
                        static_cast<size_t>(faces[3*i+1]),
                        static_cast<size_t>(faces[3*i+2]) };

    ManifoldSurfaceMesh mesh(polygons);
    VertexData<Vector3> positions(mesh);
    for (size_t i = 0; i < static_cast<size_t>(nV); ++i)
        positions[mesh.vertex(i)] = Vector3{ vertices[3*i], vertices[3*i+1], vertices[3*i+2] };
    VertexPositionGeometry geom(mesh, positions);

    DelaunayNormalization out;
    out.flips = static_cast<int>(fixDelaunay(mesh, geom));

    mesh.compress();  // ensure dense indexing before reading back
    std::vector<std::vector<size_t>> fvl = mesh.getFaceVertexList();
    out.faces.resize(static_cast<int>(fvl.size()), 3);
    for (size_t i = 0; i < fvl.size(); ++i)
        for (int k = 0; k < 3; ++k)
            out.faces(static_cast<int>(i), k) = static_cast<int>(fvl[i][k]);
    return out;
}

} // namespace nxr::manifold
