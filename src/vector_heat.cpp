#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/surface_centers.h"
#include "geometrycentral/surface/surface_point.h"

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Solver impl (PIMPL) ───────────────────────────────────
//
// Owns the geometry-central VectorHeatMethodSolver plus a
// reference to the ComputeContext so wrappers can resolve
// vertex / face indices and reach the embedded geometry for
// tangent-basis lifting.

class VectorHeatSolverImpl {
public:
    VectorHeatSolverImpl(ComputeContext& c, double tCoef)
        : ctx(c), solver(c.geometry(), tCoef) {
        c.geometry().requireVertexTangentBasis();
    }
    ComputeContext& ctx;
    VectorHeatMethodSolver solver;
};

VectorHeatSolver::VectorHeatSolver(ComputeContext& ctx, double tCoef)
    : impl_(std::make_unique<VectorHeatSolverImpl>(ctx, tCoef)) {}

VectorHeatSolver::~VectorHeatSolver() = default;

VectorHeatSolverImpl& VectorHeatSolver::impl()  { return *impl_; }
ComputeContext&       VectorHeatSolver::ctx()   { return impl_->ctx; }

// ── Helpers ───────────────────────────────────────────────

namespace {

void checkVertexInRange(int idx, int nV) {
    if (idx < 0 || idx >= nV) {
        throw Error(ErrorCode::InvalidInput,
            "vertex index out of range",
            "expected 0 <= idx < " + std::to_string(nV));
    }
}

// Project a 3D world vector into a vertex's intrinsic tangent
// basis. Anything in the normal direction is dropped.
Vector2 worldToTangent(const Vector3& w,
                       const std::array<Vector3, 2>& basis) {
    return {dot(w, basis[0]), dot(w, basis[1])};
}

// Lift a 2D tangent vector back to 3D world using the same basis.
Vector3 tangentToWorld(const Vector2& t,
                       const std::array<Vector3, 2>& basis) {
    return t.x * basis[0] + t.y * basis[1];
}

} // namespace

// ── Tangent vector transport ──────────────────────────────

Eigen::MatrixXd vectorHeatTransport(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::MatrixXd& sourceVectors
) {
    auto& s = solver.impl();
    auto& mesh = s.ctx.mesh();
    auto& geom = s.ctx.geometry();
    int nV = s.ctx.nV();

    if (static_cast<int>(sourceVertices.size()) != sourceVectors.rows()) {
        throw Error(ErrorCode::InvalidInput,
            "sourceVertices length must match sourceVectors row count");
    }
    if (sourceVectors.cols() != 3) {
        throw Error(ErrorCode::InvalidInput,
            "sourceVectors must be Nx3 (world-space tangent vectors)");
    }

    std::vector<std::tuple<Vertex, Vector2>> sources;
    sources.reserve(sourceVertices.size());
    for (std::size_t i = 0; i < sourceVertices.size(); i++) {
        checkVertexInRange(sourceVertices[i], nV);
        Vertex v = mesh.vertex(static_cast<size_t>(sourceVertices[i]));
        Vector3 w{sourceVectors(i, 0), sourceVectors(i, 1), sourceVectors(i, 2)};
        Vector2 t = worldToTangent(w, geom.vertexTangentBasis[v]);
        sources.emplace_back(v, t);
    }

    VertexData<Vector2> transported = s.solver.transportTangentVectors(sources);

    Eigen::MatrixXd out(nV, 3);
    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(v.getIndex());
        Vector3 w = tangentToWorld(transported[v], geom.vertexTangentBasis[v]);
        out(vi, 0) = w.x;
        out(vi, 1) = w.y;
        out(vi, 2) = w.z;
    }

    std::cout << "[vector_heat] transported " << sourceVertices.size()
              << " sources to " << nV << " vertices" << std::endl;
    return out;
}

// ── Scalar extension ──────────────────────────────────────

Eigen::VectorXd vectorHeatExtendScalar(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::VectorXd& sourceValues
) {
    auto& s = solver.impl();
    auto& mesh = s.ctx.mesh();
    int nV = s.ctx.nV();

    if (static_cast<int>(sourceVertices.size()) != sourceValues.size()) {
        throw Error(ErrorCode::InvalidInput,
            "sourceVertices and sourceValues must have the same length");
    }

    std::vector<std::tuple<Vertex, double>> sources;
    sources.reserve(sourceVertices.size());
    for (std::size_t i = 0; i < sourceVertices.size(); i++) {
        checkVertexInRange(sourceVertices[i], nV);
        sources.emplace_back(
            mesh.vertex(static_cast<size_t>(sourceVertices[i])),
            sourceValues[static_cast<Eigen::Index>(i)]);
    }

    VertexData<double> extended = s.solver.extendScalar(sources);

    Eigen::VectorXd out(nV);
    for (Vertex v : mesh.vertices()) {
        out(static_cast<int>(v.getIndex())) = extended[v];
    }
    return out;
}

// ── Log map ───────────────────────────────────────────────

LogMapResult vectorHeatLogMap(
    VectorHeatSolver& solver,
    int sourceVertex,
    LogMapStrategy strategy
) {
    auto& s = solver.impl();
    auto& mesh = s.ctx.mesh();
    auto& geom = s.ctx.geometry();
    int nV = s.ctx.nV();
    checkVertexInRange(sourceVertex, nV);

    Vertex src = mesh.vertex(static_cast<size_t>(sourceVertex));

    geometrycentral::surface::LogMapStrategy gcStrategy;
    switch (strategy) {
        case LogMapStrategy::VectorHeat:
            gcStrategy = geometrycentral::surface::LogMapStrategy::VectorHeat;
            break;
        case LogMapStrategy::AffineLocal:
            gcStrategy = geometrycentral::surface::LogMapStrategy::AffineLocal;
            break;
        case LogMapStrategy::AffineAdaptive:
            gcStrategy = geometrycentral::surface::LogMapStrategy::AffineAdaptive;
            break;
        default:
            throw Error(ErrorCode::InvalidInput, "unknown LogMapStrategy");
    }

    VertexData<Vector2> logMap = s.solver.computeLogMap(src, gcStrategy);

    LogMapResult out;
    out.logCoords.resize(nV, 2);
    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(v.getIndex());
        out.logCoords(vi, 0) = logMap[v].x;
        out.logCoords(vi, 1) = logMap[v].y;
    }
    const auto& srcBasis = geom.vertexTangentBasis[src];
    out.sourceE1 << srcBasis[0].x, srcBasis[0].y, srcBasis[0].z;
    out.sourceE2 << srcBasis[1].x, srcBasis[1].y, srcBasis[1].z;
    return out;
}

// ── Karcher mean / find center ────────────────────────────

Eigen::Vector3d vectorHeatFindCenter(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    int p
) {
    auto& s = solver.impl();
    auto& mesh = s.ctx.mesh();
    auto& geom = s.ctx.geometry();
    int nV = s.ctx.nV();

    if (sourceVertices.empty()) {
        throw Error(ErrorCode::InvalidInput,
            "findCenter requires at least one source vertex");
    }

    std::vector<Vertex> srcs;
    srcs.reserve(sourceVertices.size());
    for (int idx : sourceVertices) {
        checkVertexInRange(idx, nV);
        srcs.push_back(mesh.vertex(static_cast<size_t>(idx)));
    }

    // findCenter requires a ManifoldSurfaceMesh — ComputeContext stores
    // exactly that, but the base reference is SurfaceMesh. Cast down.
    auto* manifold = dynamic_cast<ManifoldSurfaceMesh*>(&mesh);
    if (!manifold) {
        throw Error(ErrorCode::NonManifold,
            "vectorHeatFindCenter requires a manifold surface mesh");
    }

    SurfacePoint center = findCenter(*manifold, geom, s.solver, srcs, p);

    // Resolve SurfacePoint to a 3D world position via barycentric blend.
    Vector3 pos = center.interpolate(geom.inputVertexPositions);
    Eigen::Vector3d out;
    out << pos.x, pos.y, pos.z;
    return out;
}

} // namespace nxr::compute
