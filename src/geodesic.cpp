#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/heat_method_distance.h"

#include <iostream>

namespace nxr::manifold::solve {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Solver impl (PIMPL) ───────────────────────────────────
//
// Owns the geometry-central HeatMethodDistanceSolver plus a
// reference to the Manifold. The solver factorizes both
// Cholesky systems (M + tA, and A) at construction; subsequent
// computeDistance() calls back-substitute only. Mirrors the
// VectorHeatSolver / SignedHeatSolver pattern.

class HeatGeodesicSolverImpl {
public:
    HeatGeodesicSolverImpl(Manifold& c, double tCoef)
        : m(c), solver(c.geometry(), tCoef) {}
    Manifold& m;
    HeatMethodDistanceSolver solver;
};

HeatGeodesicSolver::HeatGeodesicSolver(Manifold& m, double tCoef)
    : impl_(std::make_unique<HeatGeodesicSolverImpl>(m, tCoef)) {}

HeatGeodesicSolver::~HeatGeodesicSolver() = default;

HeatGeodesicSolverImpl& HeatGeodesicSolver::impl() { return *impl_; }

// ── Free function ─────────────────────────────────────────

Eigen::VectorXd heat(
    HeatGeodesicSolver& solver,
    const std::vector<int>& sourceVertices
) {
    auto& s = solver.impl();
    auto& mesh = s.m.mesh();
    int nV = s.m.nV();

    // Build source vertex list for geometry-central
    std::vector<Vertex> sources;
    sources.reserve(sourceVertices.size());
    for (int idx : sourceVertices) {
        sources.push_back(mesh.vertex(static_cast<size_t>(idx)));
    }

    // Solve via the cached heat-method solver — back-substitution only
    // on subsequent calls; no per-call factorization.
    VertexData<double> distances = s.solver.computeDistance(sources);

    // Convert to Eigen::VectorXd indexed by vertex index
    Eigen::VectorXd result(nV);
    for (Vertex v : mesh.vertices()) {
        result(static_cast<int>(v.getIndex())) = distances[v];
    }

    std::cout << "[geodesic] Computed distances from " << sourceVertices.size()
              << " sources. Max distance: " << result.maxCoeff() << std::endl;

    return result;
}

} // namespace nxr::manifold::solve