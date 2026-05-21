#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/signed_heat_method.h"
#include "geometrycentral/surface/surface_point.h"

#include <iostream>
#include <vector>

namespace nxr::manifold::solve {

using namespace geometrycentral;
using namespace geometrycentral::surface;

class SignedHeatSolverImpl {
public:
    SignedHeatSolverImpl(Manifold& c, double tCoef)
        : m(c), solver(c.geometry(), tCoef) {}
    Manifold& m;
    geometrycentral::surface::SignedHeatSolver solver;
};

SignedHeatSolver::SignedHeatSolver(Manifold& m, double tCoef)
    : impl_(std::make_unique<SignedHeatSolverImpl>(m, tCoef)) {}

SignedHeatSolver::~SignedHeatSolver() = default;

SignedHeatSolverImpl& SignedHeatSolver::impl() { return *impl_; }

Eigen::VectorXd signedHeat(
    SignedHeatSolver& solver,
    const std::vector<int>& curveVertices,
    bool isLoop,
    SignedHeatLevelSet levelSet
) {
    auto& s = solver.impl();
    auto& mesh = s.m.mesh();
    int nV = s.m.nV();

    if (curveVertices.size() < 2) {
        throw Error(ErrorCode::InvalidInput,
            "signedHeat: curve must have at least 2 vertices");
    }

    Curve curve;
    curve.isSigned = true;
    curve.nodes.reserve(curveVertices.size() + (isLoop ? 1 : 0));
    for (int idx : curveVertices) {
        if (idx < 0 || idx >= nV) {
            throw Error(ErrorCode::InvalidInput,
                "curve vertex index out of range");
        }
        curve.nodes.emplace_back(mesh.vertex(static_cast<size_t>(idx)));
    }
    if (isLoop) {
        curve.nodes.emplace_back(
            mesh.vertex(static_cast<size_t>(curveVertices.front())));
    }

    SignedHeatOptions opts;
    switch (levelSet) {
        case SignedHeatLevelSet::None:
            opts.levelSetConstraint = LevelSetConstraint::None; break;
        case SignedHeatLevelSet::ZeroSet:
            opts.levelSetConstraint = LevelSetConstraint::ZeroSet; break;
        case SignedHeatLevelSet::Multiple:
            opts.levelSetConstraint = LevelSetConstraint::Multiple; break;
    }

    std::vector<Curve> curves{std::move(curve)};
    VertexData<double> distances = s.solver.computeDistance(curves, opts);

    Eigen::VectorXd out(nV);
    for (Vertex v : mesh.vertices()) {
        out(static_cast<int>(v.getIndex())) = distances[v];
    }

    std::cout << "[signed_heat] " << curveVertices.size()
              << "-vertex curve, isLoop=" << isLoop
              << ", range [" << out.minCoeff() << ", " << out.maxCoeff() << "]"
              << std::endl;
    return out;
}

} // namespace nxr::manifold::solve