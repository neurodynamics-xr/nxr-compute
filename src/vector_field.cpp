#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>

namespace nxr::field::interp {

using namespace geometrycentral;
using namespace geometrycentral::surface;
using nxr::manifold::Manifold;
using nxr::manifold::ops::DECOperators;

// ─── Whitney interpolation: 1-form (edges) → face vectors ────
//
// Given a 1-form ω on edges, compute face-centered tangent vectors via
// Whitney basis:
//
//   V(f) = (N_f × (a + b + c)) / (6 · A_f)
//
// where for face with vertices (pi, pj, pk) and edges:
//   e_ij = pj - pi,  e_jk = pk - pj,  e_ki = pi - pk
//   a = (e_ki - e_jk) · c_ij
//   b = (e_ij - e_ki) · c_jk
//   c = (e_jk - e_ij) · c_ki
// and c_* are the oriented 1-form values on each edge (with sign flipped
// if the halfedge traversal direction is opposite to the edge's canonical
// direction).
Eigen::MatrixXd whitney(
    Manifold& m,
    const DECOperators& /*dec*/,  // unused here; we use raw halfedge iteration
    const Eigen::VectorXd& oneForm
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    int nF = m.nF();

    geom.requireFaceNormals();
    geom.requireFaceAreas();

    Eigen::MatrixXd out(nF, 3);

    for (Face f : mesh.faces()) {
        int fi = static_cast<int>(f.getIndex());

        Halfedge h = f.halfedge();
        // Vertices in order i, j, k (halfedge traversal h, h.next(), h.next().next())
        Vector3 pi = geom.inputVertexPositions[h.tailVertex()];
        Vector3 pj = geom.inputVertexPositions[h.next().tailVertex()];
        Vector3 pk = geom.inputVertexPositions[h.next().next().tailVertex()];
        Vector3 eij = pj - pi;
        Vector3 ejk = pk - pj;
        Vector3 eki = pi - pk;

        // 1-form coefficients on the 3 edges, with sign correction:
        // geometry-central's d0 uses each edge's canonical halfedge direction
        // (Edge::halfedge()). If the face halfedge points the opposite way,
        // flip the sign so the 1-form reads consistently.
        auto getSignedValue = [&](Halfedge he) -> double {
            Edge e = he.edge();
            double val = oneForm(static_cast<int>(e.getIndex()));
            return (he == e.halfedge()) ? val : -val;
        };

        double cij = getSignedValue(h);
        double cjk = getSignedValue(h.next());
        double cki = getSignedValue(h.next().next());

        Vector3 a = (eki - ejk) * cij;
        Vector3 b = (eij - eki) * cjk;
        Vector3 c = (ejk - eij) * cki;

        Vector3 N = geom.faceNormals[f];
        double A = geom.faceAreas[f];

        Vector3 V = (A > 1e-30) ? (cross(N, a + b + c) / (6.0 * A)) : Vector3{0, 0, 0};

        out(fi, 0) = V.x;
        out(fi, 1) = V.y;
        out(fi, 2) = V.z;
    }

    return out;
}

} // namespace nxr::field::interp

namespace nxr::field::op {

using namespace geometrycentral;
using namespace geometrycentral::surface;
using nxr::manifold::Manifold;

// ─── Scalar gradient: vertex scalar field → face vectors ─────
//
// For a triangle with vertices (i, j, k) and face normal N, area A:
//   ∇u |_f = (1 / 2A) Σ (u_i) · (N × e_jk_perp_to_vertex_i)
// Equivalently, the gradient is constant on each triangle:
//   ∇u = (1 / 2A) [ u_i (N × e_jk) + u_j (N × e_ki) + u_k (N × e_ij) ]
// where e_jk, e_ki, e_ij are edges OPPOSITE to vertex i, j, k respectively.
Eigen::MatrixXd gradient(
    Manifold& m,
    const Eigen::VectorXd& u
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    int nF = m.nF();

    geom.requireFaceNormals();
    geom.requireFaceAreas();

    Eigen::MatrixXd out(nF, 3);

    for (Face f : mesh.faces()) {
        int fi = static_cast<int>(f.getIndex());

        Halfedge h = f.halfedge();
        Vertex vi = h.tailVertex();
        Vertex vj = h.next().tailVertex();
        Vertex vk = h.next().next().tailVertex();

        Vector3 pi = geom.inputVertexPositions[vi];
        Vector3 pj = geom.inputVertexPositions[vj];
        Vector3 pk = geom.inputVertexPositions[vk];

        // Edge opposite vertex i is jk = pk - pj, etc.
        Vector3 eOpp_i = pk - pj;  // opposite i
        Vector3 eOpp_j = pi - pk;  // opposite j
        Vector3 eOpp_k = pj - pi;  // opposite k

        double ui = u(static_cast<int>(vi.getIndex()));
        double uj = u(static_cast<int>(vj.getIndex()));
        double uk = u(static_cast<int>(vk.getIndex()));

        Vector3 N = geom.faceNormals[f];
        double A = geom.faceAreas[f];

        // ∇u = (1 / 2A) (ui · N×eOpp_i + uj · N×eOpp_j + uk · N×eOpp_k)
        Vector3 grad = (A > 1e-30)
            ? (cross(N, eOpp_i) * ui + cross(N, eOpp_j) * uj + cross(N, eOpp_k) * uk) / (2.0 * A)
            : Vector3{0, 0, 0};

        out(fi, 0) = grad.x;
        out(fi, 1) = grad.y;
        out(fi, 2) = grad.z;
    }

    std::cout << "[vector_field] gradient: " << nF << " face vectors" << std::endl;
    return out;
}

} // namespace nxr::field::op
