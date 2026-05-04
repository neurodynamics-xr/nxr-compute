#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/heat_method_distance.h"

#include <iostream>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Solver impl (PIMPL) ───────────────────────────────────
//
// Owns the geometry-central HeatMethodDistanceSolver plus a
// reference to the ComputeContext. The solver factorizes both
// Cholesky systems (M + tA, and A) at construction; subsequent
// computeDistance() calls back-substitute only. Mirrors the
// VectorHeatSolver / SignedHeatSolver pattern.

class HeatGeodesicSolverImpl {
public:
    HeatGeodesicSolverImpl(ComputeContext& c, double tCoef)
        : ctx(c), solver(c.geometry(), tCoef) {}
    ComputeContext& ctx;
    HeatMethodDistanceSolver solver;
};

HeatGeodesicSolver::HeatGeodesicSolver(ComputeContext& ctx, double tCoef)
    : impl_(std::make_unique<HeatGeodesicSolverImpl>(ctx, tCoef)) {}

HeatGeodesicSolver::~HeatGeodesicSolver() = default;

HeatGeodesicSolverImpl& HeatGeodesicSolver::impl() { return *impl_; }

// ── Free function ─────────────────────────────────────────

Eigen::VectorXd computeGeodesicDistance(
    HeatGeodesicSolver& solver,
    const std::vector<int>& sourceVertices
) {
    auto& s = solver.impl();
    auto& mesh = s.ctx.mesh();
    int nV = s.ctx.nV();

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

} // namespace nxr::compute
