#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>

namespace nxr::manifold::ops::facegrad {

// Barycentric dual-mesh gradient of a per-face scalar: per-face ambient 3-vector
// field [3F × F]. Green–Gauss / DEC construction (the dual of the vertex FEM
// gradient): for face f with outward normal n_f, area A_f, and boundary edges
//   l_k (tail→tip, CCW) whose twin lies in the edge-neighbor face g_k,
//   grad ψ|_f = (1/2A_f) Σ_k (ψ_{g_k} − ψ_f) (l_k × n_f).
// l_k × n_f is the OUTWARD in-plane edge normal of magnitude |l_k| (rotation of
// l_k by −90° about n_f). Σ_k l_k = 0 ⇒ the self (ψ_f) contributions cancel and
// constants are annihilated; each output 3-vector is tangent to its face.
// Closed-mesh v1: every interior face has exactly 3 edge-neighbors — fail loud on
// an open boundary (a boundary face's missing twin would corrupt the stencil).
Eigen::SparseMatrix<double> gradient(Manifold& m) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    if (mesh.hasBoundary())
        throw Error(ErrorCode::InvalidInput,
            "facegrad::gradient: open boundary unsupported (closed-mesh v1)",
            "Every face must have three edge-neighbors; pass a closed mesh "
            "(e.g. a FreeSurfer hemisphere).");
    geom.requireFaceNormals();
    geom.requireFaceAreas();

    const int Fn = m.nF();
    auto vec3 = [](const Vector3& u) { return Eigen::Vector3d(u.x, u.y, u.z); };

    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(Fn) * 3 * 3 * 2);  // 3 edges × 3 components × {neighbor,self}

    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        if (A <= 0.0)
            throw Error(ErrorCode::InvalidInput,
                "facegrad::gradient: degenerate (zero-area) face",
                "Face index " + std::to_string(fi) + "; fix mesh quality first.");
        const Eigen::Vector3d nf = vec3(geom.faceNormals[f]);
        const double s = 1.0 / (2.0 * A);

        for (Halfedge he : f.adjacentHalfedges()) {
            const Eigen::Vector3d l =
                vec3(geom.inputVertexPositions[he.tipVertex()]) -
                vec3(geom.inputVertexPositions[he.tailVertex()]);
            const Eigen::Vector3d w = s * l.cross(nf);          // outward edge normal / 2A
            const int gi = static_cast<int>(he.twin().face().getIndex());
            for (int d = 0; d < 3; ++d) {
                if (w[d] != 0.0) {
                    T.emplace_back(3 * fi + d, gi, w[d]);        // +w on the neighbor column
                    T.emplace_back(3 * fi + d, fi, -w[d]);       // −w on the self column
                }
            }
        }
    }
    Eigen::SparseMatrix<double> G(3 * Fn, Fn);
    G.setFromTriplets(T.begin(), T.end());
    G.makeCompressed();
    return G;
}

// Face Laplacian K̃ = gradientᵀ ⋆_F gradient [F × F], ⋆_F = per-face area mass
// (A_f on each of the 3 components). Built from THE SAME gradient triplets so the
// pair never drifts (mirrors extrinsicBlockFace building Ẽ from matrixFace).
// Symmetric PSD; kernel = constants on a closed (genus-0) mesh.
Eigen::SparseMatrix<double> laplacian(Manifold& m) {
    using namespace geometrycentral::surface;

    Eigen::SparseMatrix<double> G = gradient(m);   // [3F × F]
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireFaceAreas();
    const int Fn = m.nF();

    std::vector<Eigen::Triplet<double>> TW;
    TW.reserve(static_cast<size_t>(3) * Fn);
    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        for (int d = 0; d < 3; ++d) TW.emplace_back(3 * fi + d, 3 * fi + d, A);
    }
    Eigen::SparseMatrix<double> WF(3 * Fn, 3 * Fn);
    WF.setFromTriplets(TW.begin(), TW.end());

    // Materialise Gᵀ before the chain to avoid the lazy-transpose aliasing trap.
    Eigen::SparseMatrix<double> K =
        (Eigen::SparseMatrix<double>(G.transpose()) * WF * G).pruned();
    K.makeCompressed();
    return K;
}

}  // namespace nxr::manifold::ops::facegrad
