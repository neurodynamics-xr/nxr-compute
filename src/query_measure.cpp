#include "nxr/compute.h"

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <iostream>

namespace {

using namespace nxr::manifold;

inline void requireVertexInRange(Manifold& m, int v, const char* fnName) {
    if (v < 0 || v >= m.nV()) {
        throw nxr::core::Error(
            nxr::core::ErrorCode::InvalidInput,
            std::string(fnName) + ": vertex index out of range",
            "expected 0 <= v < " + std::to_string(m.nV()));
    }
}

}  // namespace

namespace nxr::manifold::query {

PointLocus point(Manifold& m, int v) {
    requireVertexInRange(m, v, "query::point");
    return PointLocus{v};
}

PolylineLocus line(Manifold& m, int vStart, int vEnd) {
    // tracePath validates its own inputs and runs the edge-flip
    // geodesic. Wrap the [N, 3] matrix in the locus type.
    return PolylineLocus{tracePath(m, vStart, vEnd)};
}

RegionLocus area(Manifold& m, int v, double level) {
    requireVertexInRange(m, v, "query::area");
    if (!(level > 0.0)) {
        throw nxr::core::Error(
            nxr::core::ErrorCode::InvalidInput,
            "query::area: level must be > 0");
    }

    // Compute heat-method geodesic distance from v. Each call pays
    // a HeatGeodesicSolver Cholesky factor — see the docstring on the
    // header for the perf note.
    solve::HeatGeodesicSolver solver(m);
    Eigen::VectorXd distances = solve::heat(solver, {v});

    // Walk faces; a face is in the region iff all 3 of its vertices
    // satisfy distance <= level. Conservative (under-counts faces that
    // straddle the boundary) but well-defined.
    auto& mesh = m.mesh();
    RegionLocus out;
    out.faces.reserve(mesh.nFaces());
    for (geometrycentral::surface::Face f : mesh.faces()) {
        bool allInside = true;
        for (geometrycentral::surface::Vertex u : f.adjacentVertices()) {
            const int uIdx = static_cast<int>(u.getIndex());
            if (distances(uIdx) > level) {
                allInside = false;
                break;
            }
        }
        if (allInside) {
            out.faces.push_back(static_cast<int>(f.getIndex()));
        }
    }
    return out;
}

}  // namespace nxr::manifold::query

namespace nxr::manifold::measure {

Eigen::Vector3d point(Manifold& m, int v) {
    requireVertexInRange(m, v, "measure::point");
    auto& geom = m.geometry();
    geom.requireVertexPositions();
    auto& mesh = m.mesh();
    const auto pos = geom.vertexPositions[mesh.vertex(static_cast<size_t>(v))];
    return Eigen::Vector3d(pos.x, pos.y, pos.z);
}

double line(const query::PolylineLocus& locus) {
    const auto& P = locus.points;
    if (P.rows() < 2) return 0.0;
    double L = 0.0;
    for (int i = 1; i < P.rows(); ++i) {
        L += (P.row(i) - P.row(i - 1)).norm();
    }
    return L;
}

double line(Manifold& m, int vStart, int vEnd) {
    return line(query::line(m, vStart, vEnd));
}

double area(Manifold& m, const query::RegionLocus& locus) {
    auto& geom = m.geometry();
    geom.requireFaceAreas();
    auto& mesh = m.mesh();
    double total = 0.0;
    for (int fIdx : locus.faces) {
        total += geom.faceAreas[mesh.face(static_cast<size_t>(fIdx))];
    }
    return total;
}

double area(Manifold& m, int v, double level) {
    return area(m, query::area(m, v, level));
}

}  // namespace nxr::manifold::measure
