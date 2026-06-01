#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>
#include <iostream>

namespace nxr::manifold {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Manifold implementation ────────────────────────────

Manifold::Manifold(const double* vertices, int nV,
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

Manifold::~Manifold() = default;

int Manifold::nV() const { return static_cast<int>(mesh_->nVertices()); }
int Manifold::nE() const { return static_cast<int>(mesh_->nEdges()); }
int Manifold::nF() const { return static_cast<int>(mesh_->nFaces()); }

ManifoldSurfaceMesh& Manifold::mesh() { return *mesh_; }
VertexPositionGeometry& Manifold::geometry() { return *geometry_; }

} // namespace nxr::manifold

namespace nxr::manifold::ops {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Mass-matrix variant dispatch ─────────────────────────────
//
// Two options, names matching geometry-central's vocabulary exactly:
//   Lumped   — diagonal, sourced from GC's vertexLumpedMassMatrix
//              (== diag(vertexDualAreas), where each vertexDualAreas[v]
//              is the sum of A_T/3 over incident triangles). NOT a
//              circumcentric or mixed-Voronoi construction.
//   Galerkin — sparse, sourced from GC's vertexGalerkinMassMatrix.
//              Per-triangle element matrix is (A_T/12)·[[2 1 1][1 2 1][1 1 2]],
//              the L² integrals of P1 hat functions. Eigenvalues converge
//              to the continuous LB spectrum faster than lumped on coarse
//              meshes; matches lapy's default and gptoolbox's `'full'`.
//
// Both variants conserve total area (Σᵢⱼ Mᵢⱼ ≡ totalArea) and are symmetric.
// Both are sourced directly from geometry-central rather than rebuilt here.

// ── Operator Assembly ────────────────────────────────────────

ManifoldOperators assembleManifoldOperators(Manifold& m,
                                    MassMatrixVariant variant) {
    auto& mesh = m.mesh();
    auto& geometry = m.geometry();

    int nV = m.nV();
    int nE = m.nE();
    int nF = m.nF();

    // Pin the cotangent Laplacian — it's a view in the returned struct.
    // cotanLaplacian is already symmetric (cotan weights symmetric,
    // diagonal = negative row sum); the previous (L+Lᵀ)/2 step has been
    // dropped (verified bit-identical across all numerical test suites).
    geometry.requireCotanLaplacian();

    // Mass matrix per requested variant — both source directly from
    // geometry-central. Mass is stored by VALUE (not a view) so the
    // struct shape stays uniform across variants; for Lumped this
    // implies one extra sparse-matrix copy on assembly, which is
    // negligible vs. the eigensolve / Poisson costs.
    Eigen::SparseMatrix<double> mass;
    double totalArea = 0.0;
    if (variant == MassMatrixVariant::Lumped) {
        geometry.requireVertexLumpedMassMatrix();
        mass = geometry.vertexLumpedMassMatrix;
        totalArea = mass.diagonal().sum();
    } else if (variant == MassMatrixVariant::Galerkin) {
        geometry.requireVertexGalerkinMassMatrix();
        mass = geometry.vertexGalerkinMassMatrix;
        // Σᵢⱼ Mᵢⱼ ≡ totalArea for the Galerkin matrix by construction
        // (each face contributes A_T · sum_of_element_matrix/12 = A_T).
        totalArea = mass.sum();
    } else {
        throw Error(ErrorCode::InvalidInput,
            "assembleManifoldOperators: unknown MassMatrixVariant");
    }

    // vertexDualAreas reports GC's per-vertex A/3 dual area regardless
    // of the chosen mass variant — downstream consumers (curvature,
    // gradients, diffusion) want a per-vertex weight even when the L²
    // inner product uses a non-diagonal M.
    geometry.requireVertexDualAreas();
    Eigen::VectorXd vertexDualAreas(nV);
    for (Vertex v : mesh.vertices()) {
        vertexDualAreas(static_cast<int>(v.getIndex())) =
            geometry.vertexDualAreas[v];
    }

    // Vertex normals — V×3 lifted from VertexData<Vector3>. Stored
    // row-major (VertexNormalsMatrix alias) so the §11 row-major
    // output buffer at the binding edge is a direct memcpy of
    // .data(); no transpose pass needed. Value-copy is unavoidable
    // either way: geometry-central stores normals as a labeled
    // array, not a contiguous matrix.
    geometry.requireVertexNormals();
    VertexNormalsMatrix vertexNormals(nV, 3);
    for (Vertex v : mesh.vertices()) {
        int idx = static_cast<int>(v.getIndex());
        Vector3 n = geometry.vertexNormals[v];
        vertexNormals(idx, 0) = n.x;
        vertexNormals(idx, 1) = n.y;
        vertexNormals(idx, 2) = n.z;
    }

    const char* variantName =
        variant == MassMatrixVariant::Lumped   ? "lumped"   :
        variant == MassMatrixVariant::Galerkin ? "galerkin" : "?";
    std::cout << "[mesh_operators] Assembled: "
              << nV << " V, " << nE << " E, " << nF << " F "
              << "(total area = " << totalArea
              << ", mass = " << variantName
              << ", mass nnz = " << mass.nonZeros() << ")" << std::endl;

    return ManifoldOperators(
        geometry.cotanLaplacian,
        std::move(mass),
        variant,
        std::move(vertexDualAreas),
        std::move(vertexNormals),
        totalArea
    );
}

ManifoldOperators assembleManifoldOperators(
    const double* vertices, int nV,
    const int32_t* faces, int nF,
    MassMatrixVariant variant
) {
    Manifold m(vertices, nV, faces, nF);
    return assembleManifoldOperators(m, variant);
}

MassMatrixVariant parseMassMatrixVariant(const std::string& name) {
    if (name == "lumped")   return MassMatrixVariant::Lumped;
    if (name == "galerkin") return MassMatrixVariant::Galerkin;
    throw Error(ErrorCode::InvalidInput,
        "Unknown mass-matrix variant '" + name +
        "' — expected 'lumped' or 'galerkin' (matching geometry-central's "
        "vertexLumpedMassMatrix and vertexGalerkinMassMatrix).");
}

// ── DEC Operators ────────────────────────────────────────────

DECOperators assembleDECOperators(Manifold& m) {
    auto& geometry = m.geometry();


    geometry.requireDECOperators();
    return DECOperators(
        geometry.d0,
        geometry.d1,
        geometry.hodge0,
        geometry.hodge1,
        geometry.hodge2,
        geometry.hodge1Inverse
    );
}

// ── Zero-copy passthrough accessors ──────────────────────────
//
// Each calls GC's require* once (no-op if cached) and returns the
// cached field by const reference. Bit-identical to the matching
// field on the struct returned by assembleDECOperators.

const Eigen::SparseMatrix<double>& d0(Manifold& m) {
    auto& geom = m.geometry();
    geom.requireDECOperators();
    return geom.d0;
}

const Eigen::SparseMatrix<double>& d1(Manifold& m) {
    auto& geom = m.geometry();
    geom.requireDECOperators();
    return geom.d1;
}

const Eigen::SparseMatrix<double>& hodge0(Manifold& m) {
    auto& geom = m.geometry();
    geom.requireDECOperators();
    return geom.hodge0;
}

const Eigen::SparseMatrix<double>& hodge1(Manifold& m) {
    auto& geom = m.geometry();
    geom.requireDECOperators();
    return geom.hodge1;
}

const Eigen::SparseMatrix<double>& hodge2(Manifold& m) {
    auto& geom = m.geometry();
    geom.requireDECOperators();
    return geom.hodge2;
}

const Eigen::SparseMatrix<double>& hodge1Inverse(Manifold& m) {
    auto& geom = m.geometry();
    geom.requireDECOperators();
    return geom.hodge1Inverse;
}

const Eigen::VectorXd& vertexDualAreas(Manifold& m) {
    auto& geom = m.geometry();
    geom.requireVertexDualAreas();
    return geom.vertexDualAreas.raw();
}

} // namespace nxr::manifold::ops