#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/boundary_first_flattening.h"

#include <iostream>
#include <stdexcept>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Boundary First Flattening — produces low-distortion conformal
// UV coordinates for an open (boundary-bearing) manifold mesh.
// Reference: Sawhney & Crane, "Boundary First Flattening" (2017).
//
// We use geometry-central's bundled implementation
// (parameterizeBFF) — no additional dependency. The default
// version flattens with a uniform-curvature boundary (the
// authors' standard "free boundary" mode); the alternative
// `parameterizeBFFfromScaleFactors` / `…fromExteriorAngles`
// entry points are not exposed in this v0 wrapper.
//
// For closed meshes the caller must cut the surface to a disc
// topology first; this function throws otherwise.
Eigen::MatrixXd computeUVCoordinates(ComputeContext& ctx) {
    auto& mesh = ctx.mesh();
    auto& geom = ctx.geometry();

    if (mesh.nBoundaryLoops() == 0) {
        throw Error(ErrorCode::OpenMeshRequired,
            "computeUVCoordinates: mesh has no boundary",
            "BFF requires an open mesh. Cut the surface to disc topology before calling.");
    }

    VertexData<Vector2> uvs = parameterizeBFF(mesh, geom);

    int nV = ctx.nV();
    Eigen::MatrixXd out(nV, 2);
    for (Vertex v : mesh.vertices()) {
        int idx = static_cast<int>(v.getIndex());
        Vector2 uv = uvs[v];
        out(idx, 0) = uv.x;
        out(idx, 1) = uv.y;
    }

    std::cout << "[parametrization] BFF: " << nV
              << " UV coords (" << mesh.nBoundaryLoops()
              << " boundary loop(s))" << std::endl;
    return out;
}

} // namespace nxr::compute
