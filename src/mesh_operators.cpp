#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>
#include <iostream>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── ComputeContext implementation ────────────────────────────

ComputeContext::ComputeContext(const double* vertices, int nV,
                                const int32_t* faces, int nF) {
    // Build polygon list from flat face array
    std::vector<std::vector<size_t>> polygons(nF);
    for (int i = 0; i < nF; i++) {
        polygons[i] = {
            static_cast<size_t>(faces[3 * i]),
            static_cast<size_t>(faces[3 * i + 1]),
            static_cast<size_t>(faces[3 * i + 2])
        };
    }

    mesh_ = std::make_unique<ManifoldSurfaceMesh>(polygons);

    VertexData<Vector3> positions(*mesh_);
    for (size_t i = 0; i < static_cast<size_t>(nV); i++) {
        positions[mesh_->vertex(i)] = Vector3{
            vertices[3*i], vertices[3*i+1], vertices[3*i+2]
        };
    }
    geometry_ = std::make_unique<VertexPositionGeometry>(*mesh_, positions);
}

ComputeContext::~ComputeContext() = default;

int ComputeContext::nV() const { return static_cast<int>(mesh_->nVertices()); }
int ComputeContext::nE() const { return static_cast<int>(mesh_->nEdges()); }
int ComputeContext::nF() const { return static_cast<int>(mesh_->nFaces()); }

ManifoldSurfaceMesh& ComputeContext::mesh() { return *mesh_; }
VertexPositionGeometry& ComputeContext::geometry() { return *geometry_; }

// ── Operator Assembly ────────────────────────────────────────

MeshOperators assembleMeshOperators(ComputeContext& ctx) {
    auto& mesh = ctx.mesh();
    auto& geometry = ctx.geometry();

    MeshOperators ops;
    ops.nV = ctx.nV();
    ops.nE = ctx.nE();
    ops.nF = ctx.nF();

    // Cotangent Laplacian (already positive semidefinite in geometry-central)
    geometry.requireCotanLaplacian();
    ops.stiffness = geometry.cotanLaplacian;
    // Symmetrize for numerical safety
    Eigen::SparseMatrix<double> Lt = ops.stiffness.transpose();
    ops.stiffness = (ops.stiffness + Lt) * 0.5;

    // Lumped mass matrix (Voronoi dual areas on diagonal)
    geometry.requireVertexDualAreas();
    ops.vertexAreas.resize(ops.nV);
    std::vector<Eigen::Triplet<double>> massTriplets;
    massTriplets.reserve(ops.nV);
    ops.totalArea = 0.0;
    for (Vertex v : mesh.vertices()) {
        int idx = static_cast<int>(v.getIndex());
        double area = geometry.vertexDualAreas[v];
        ops.vertexAreas(idx) = area;
        ops.totalArea += area;
        massTriplets.emplace_back(idx, idx, area);
    }
    ops.mass.resize(ops.nV, ops.nV);
    ops.mass.setFromTriplets(massTriplets.begin(), massTriplets.end());

    // Vertex normals
    geometry.requireVertexNormals();
    ops.normals.resize(ops.nV, 3);
    for (Vertex v : mesh.vertices()) {
        int idx = static_cast<int>(v.getIndex());
        Vector3 n = geometry.vertexNormals[v];
        ops.normals(idx, 0) = n.x;
        ops.normals(idx, 1) = n.y;
        ops.normals(idx, 2) = n.z;
    }

    std::cout << "[mesh_operators] Assembled: "
              << ops.nV << " V, " << ops.nE << " E, " << ops.nF << " F "
              << "(total area = " << ops.totalArea << ")" << std::endl;

    return ops;
}

MeshOperators assembleMeshOperators(
    const double* vertices, int nV,
    const int32_t* faces, int nF
) {
    ComputeContext ctx(vertices, nV, faces, nF);
    return assembleMeshOperators(ctx);
}

// ── DEC Operators ────────────────────────────────────────────

DECOperators assembleDECOperators(ComputeContext& ctx) {
    auto& geometry = ctx.geometry();

    geometry.requireDECOperators();

    DECOperators dec;
    dec.d0 = geometry.d0;
    dec.d1 = geometry.d1;
    dec.hodge0 = geometry.hodge0;
    dec.hodge1 = geometry.hodge1;
    dec.hodge2 = geometry.hodge2;
    dec.hodge1Inverse = geometry.hodge1Inverse;

    return dec;
}

} // namespace nxr::compute
