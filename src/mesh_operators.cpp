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

// ── Mass-matrix variant dispatch ─────────────────────────────
//
// Three options, all symmetric and area-conserving (Σᵢⱼ Mᵢⱼ ≡ totalArea):
//   Voronoi      — diagonal, mixed Voronoi-barycentric (Meyer 2003).
//                  Default; what geometry-central's vertexDualAreas
//                  returns. Robust on obtuse triangles.
//   Barycentric  — diagonal, M_ii = Σ_{T ∋ i} (A_T / 3). Pure lumped.
//   ConsistentFEM — sparse non-diagonal, full FEM mass: per-triangle
//                   element matrix is (A_T/12)·[[2 1 1][1 2 1][1 1 2]],
//                   from the L² integrals of P1 hat functions.
//                   Eigenvalues converge faster to the continuous LB
//                   spectrum on coarse meshes; matches lapy's default
//                   and gptoolbox's `'full'` variant.

namespace {

// Reusable helper: emit triplets onto `out` and accumulate area into
// `totalArea`. Caller is responsible for resizing M and calling
// setFromTriplets. Iterates each face once, accumulating either a
// diagonal entry per vertex or a 3×3 element per face.
void assembleMassTriplets(
    ManifoldSurfaceMesh& mesh,
    VertexPositionGeometry& geometry,
    MassMatrixVariant variant,
    std::vector<Eigen::Triplet<double>>& out,
    double& totalArea
) {
    totalArea = 0.0;
    out.clear();

    if (variant == MassMatrixVariant::Voronoi) {
        // Mixed Voronoi-barycentric per Meyer 2003. geometry-central's
        // vertexDualAreas implements the obtuse-aware mixed scheme.
        geometry.requireVertexDualAreas();
        out.reserve(mesh.nVertices());
        for (Vertex v : mesh.vertices()) {
            const int idx = static_cast<int>(v.getIndex());
            const double a = geometry.vertexDualAreas[v];
            out.emplace_back(idx, idx, a);
            totalArea += a;
        }
        return;
    }

    geometry.requireFaceAreas();

    if (variant == MassMatrixVariant::Barycentric) {
        // Σ_{T ∋ i} (A_T / 3). Each triangle distributes A/3 to each of
        // its three vertices on the diagonal; total mass per face = A,
        // so ΣM = totalArea exactly.
        out.reserve(3 * mesh.nFaces());  // setFromTriplets dedupes per (i,j)
        for (Face f : mesh.faces()) {
            const double third = geometry.faceAreas[f] / 3.0;
            totalArea += geometry.faceAreas[f];
            for (Vertex v : f.adjacentVertices()) {
                const int idx = static_cast<int>(v.getIndex());
                out.emplace_back(idx, idx, third);
            }
        }
        return;
    }

    if (variant == MassMatrixVariant::ConsistentFEM) {
        // Per-triangle element matrix M_e = (A/12) · [[2 1 1][1 2 1][1 1 2]].
        // Off-diagonal entries auto-sum across shared edges via
        // setFromTriplets. Total per triangle = A·12/12 = A → ΣM = totalArea.
        out.reserve(9 * mesh.nFaces());
        for (Face f : mesh.faces()) {
            const double a12 = geometry.faceAreas[f] / 12.0;
            totalArea += geometry.faceAreas[f];
            int idx[3];
            int k = 0;
            for (Vertex v : f.adjacentVertices()) {
                idx[k++] = static_cast<int>(v.getIndex());
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    out.emplace_back(idx[i], idx[j],
                                     (i == j ? 2.0 : 1.0) * a12);
                }
            }
        }
        return;
    }

    throw Error(ErrorCode::InvalidInput,
        "assembleMassTriplets: unknown MassMatrixVariant");
}

} // namespace

// ── Operator Assembly ────────────────────────────────────────

MeshOperators assembleMeshOperators(ComputeContext& ctx,
                                    MassMatrixVariant variant) {
    auto& mesh = ctx.mesh();
    auto& geometry = ctx.geometry();

    int nV = ctx.nV();
    int nE = ctx.nE();
    int nF = ctx.nF();

    // Pin the cotangent Laplacian — it's a view in the returned struct.
    // cotanLaplacian is already symmetric (cotan weights symmetric,
    // diagonal = negative row sum); the previous (L+Lᵀ)/2 step has been
    // dropped (verified bit-identical across all numerical test suites).
    geometry.requireCotanLaplacian();

    // Mass matrix per requested variant.
    //   * Voronoi: source from geometry.vertexLumpedMassMatrix (avoids the
    //     redundant triplet rebuild — geometry-central already maintains
    //     this matrix and pinning is cheap).
    //   * Barycentric / ConsistentFEM: build via assembleMassTriplets
    //     since geometry-central doesn't cache them.
    //
    // Mass is stored by VALUE in MeshOperators (not a view) because the
    // non-Voronoi variants don't have a single backing geometry-central
    // matrix to view; uniform value-storage keeps the struct shape
    // variant-independent.
    Eigen::SparseMatrix<double> mass;
    double totalArea = 0.0;
    if (variant == MassMatrixVariant::Voronoi) {
        geometry.requireVertexLumpedMassMatrix();
        mass = geometry.vertexLumpedMassMatrix;
        totalArea = mass.diagonal().sum();
    } else {
        std::vector<Eigen::Triplet<double>> massTriplets;
        assembleMassTriplets(mesh, geometry, variant, massTriplets, totalArea);
        mass.resize(nV, nV);
        mass.setFromTriplets(massTriplets.begin(), massTriplets.end());
    }

    // vertexDualAreas always carries mixed-Voronoi areas regardless of
    // mass variant — downstream consumers (curvature, gradients,
    // diffusion) expect per-vertex weights even when the L² inner
    // product uses a non-diagonal M.
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
        variant == MassMatrixVariant::Voronoi      ? "voronoi"      :
        variant == MassMatrixVariant::Barycentric  ? "barycentric"  :
        variant == MassMatrixVariant::ConsistentFEM? "full"         : "?";
    std::cout << "[mesh_operators] Assembled: "
              << nV << " V, " << nE << " E, " << nF << " F "
              << "(total area = " << totalArea
              << ", mass = " << variantName
              << ", mass nnz = " << mass.nonZeros() << ")" << std::endl;

    return MeshOperators(
        geometry.cotanLaplacian,
        std::move(mass),
        variant,
        std::move(vertexDualAreas),
        std::move(vertexNormals),
        totalArea
    );
}

MeshOperators assembleMeshOperators(
    const double* vertices, int nV,
    const int32_t* faces, int nF,
    MassMatrixVariant variant
) {
    ComputeContext ctx(vertices, nV, faces, nF);
    return assembleMeshOperators(ctx, variant);
}

MassMatrixVariant parseMassMatrixVariant(const std::string& name) {
    if (name == "voronoi")     return MassMatrixVariant::Voronoi;
    if (name == "barycentric") return MassMatrixVariant::Barycentric;
    if (name == "full")        return MassMatrixVariant::ConsistentFEM;
    throw Error(ErrorCode::InvalidInput,
        "Unknown mass-matrix variant '" + name +
        "' — expected 'voronoi', 'barycentric', or 'full'");
}

// ── DEC Operators ────────────────────────────────────────────

DECOperators assembleDECOperators(ComputeContext& ctx) {
    auto& geometry = ctx.geometry();
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

} // namespace nxr::compute
