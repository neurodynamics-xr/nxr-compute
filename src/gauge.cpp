#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <queue>
#include <vector>
#include <complex>
#include <cmath>
#include <array>

namespace nxr::manifold::connection {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Face-basis helpers — copied from direction_field.cpp (keep in sync) ──
// Same conventions as faceOrthonormalBasis / transportNoRotation there, so
// the face rotations share the directionVectors anchor byte-for-byte.

static std::array<Vector3, 2> gaugeFaceBasis(
    VertexPositionGeometry& geom, Face f
) {
    geom.requireFaceNormals();

    Halfedge h = f.halfedge();
    Vector3 e0 = geom.inputVertexPositions[h.tipVertex()] -
                 geom.inputVertexPositions[h.tailVertex()];
    e0 = e0.normalize();

    Vector3 N = geom.faceNormals[f];
    Vector3 e1 = cross(N, e0).normalize();

    return {e0, e1};
}

// Crane's parallel transport angle across halfedge h: αⱼ = αᵢ - θᵢⱼ + θⱼᵢ
static double gaugeFaceTransport(
    VertexPositionGeometry& geom, Halfedge h
) {
    if (!h.isInterior() || !h.twin().isInterior()) return 0.0;

    Vector3 u = geom.inputVertexPositions[h.tipVertex()] -
                geom.inputVertexPositions[h.tailVertex()];

    auto [e1, e2] = gaugeFaceBasis(geom, h.face());
    double thetaIJ = std::atan2(dot(u, e2), dot(u, e1));

    auto [f1, f2] = gaugeFaceBasis(geom, h.twin().face());
    double thetaJI = std::atan2(dot(u, f2), dot(u, f1));

    return -thetaIJ + thetaJI;
}

// Integrate the trivial-connection 1-form φ into per-vertex and per-face
// unit-complex COMBING rotations (see GaugeRotations in compute.h).
//
// Convention — matches propagateAnglesVertex / propagateAngles in
// direction_field.cpp:
//   • vertex LC transport for halfedge he (u→w):
//       rho = transportVectorsAlongHalfedge[he]    (NOT he.twin())
//   • face LC transport across halfedge h (h.face() → h.twin().face()):
//       gaugeFaceTransport(geom, h)
//   • φ sign for halfedge he:
//       sign = (he == he.edge().halfedge()) ? +1 : -1
//   so the accumulated transport coefficient at the far element is
//       t[w] = t[u] · rho_complex · exp(i · sign · φ[edge])
//
// The accumulated t is the PARALLEL FIELD's complex coefficient in the
// Levi-Civita frame (identical recursion to the direction-field angles α:
// t = e^{iα}). The stored gauge rotation is its CONJUGATE — multiplying the
// grid by e^{-iα} actively rotates the frame by +α, so
//   real(rotation .* grid) == the trivial parallel field, and the combed
//   frame is rotation .* grid.
GaugeRotations integrateTrivialGaugeRotations(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap)
{
    // 1. Solve for φ (per-edge trivial-connection 1-form, length nE).
    //    Sign convention: φ(e) is positive in the direction of e.halfedge().
    Eigen::VectorXd phi = computeTrivialConnection(m, dec, cache, singularityMap);

    auto& mesh = m.mesh();
    auto& geom = m.geometry();

    geom.requireVertexIndices();
    geom.requireEdgeIndices();
    geom.requireTransportVectorsAlongHalfedge();

    int nV = m.nV();
    int nF = m.nF();
    GaugeRotations out;
    out.vertex.resize(nV);
    out.vertex.setZero();   // unvisited components (disconnected mesh) → 0+0i, not garbage
    out.face.resize(nF);
    out.face.setZero();
    std::vector<char> visited(nV, 0);

    // 2. Vertex BFS from vertex 0; root transport t[0] = 1 (identity in LC frame).
    Vertex root = mesh.vertex(0);
    int rootIdx = static_cast<int>(geom.vertexIndices[root]);
    out.vertex(rootIdx) = std::complex<double>(1.0, 0.0);
    visited[rootIdx] = 1;

    std::queue<Vertex> q;
    q.push(root);

    while (!q.empty()) {
        Vertex u = q.front(); q.pop();
        int ui = static_cast<int>(geom.vertexIndices[u]);

        for (Halfedge he : u.outgoingHalfedges()) {
            if (!he.isInterior()) continue; // skip exterior halfedges — matches propagateAnglesVertex in direction_field.cpp

            Vertex w = he.tipVertex();
            int wi = static_cast<int>(geom.vertexIndices[w]);
            if (visited[wi]) continue;

            // Levi-Civita transport from u-frame to w-frame along he.
            // geom.transportVectorsAlongHalfedge[he] is the unit Vector2
            // that maps x-axis of u-frame to x-axis of w-frame (same
            // convention used in direction_field.cpp::propagateAnglesVertex).
            const Vector2 t = geom.transportVectorsAlongHalfedge[he];
            std::complex<double> rho(t.x, t.y);

            // Trivial correction: exp(i · sign · φ[edge]).
            // sign tracks whether he is the canonical edge halfedge,
            // matching the sign convention in computeTrivialConnection /
            // propagateAnglesVertex (direction_field.cpp lines 220–222).
            Edge e = he.edge();
            double sign = (he == e.halfedge()) ? 1.0 : -1.0;
            int eIdx = static_cast<int>(geom.edgeIndices[e]);
            std::complex<double> corr = std::polar(1.0, sign * phi(eIdx));

            // Accumulate and renormalise against floating-point drift.
            std::complex<double> r = out.vertex(ui) * rho * corr;
            out.vertex(wi) = r / std::abs(r);
            visited[wi] = 1;
            q.push(w);
        }
    }

    // 3. Face BFS over the dual graph from face 0 — mirrors propagateAngles
    //    in direction_field.cpp, so out.face anchors to directionVectors.
    std::vector<char> visitedF(nF, 0);
    Face rootF = mesh.face(0);
    int rootFIdx = static_cast<int>(rootF.getIndex());
    out.face(rootFIdx) = std::complex<double>(1.0, 0.0);
    visitedF[rootFIdx] = 1;

    std::queue<Face> qf;
    qf.push(rootF);

    while (!qf.empty()) {
        Face f = qf.front(); qf.pop();
        int fi = static_cast<int>(f.getIndex());

        for (Halfedge h : f.adjacentHalfedges()) {
            if (!h.twin().isInterior()) continue; // boundary — matches propagateAngles

            Face g = h.twin().face();
            int gi = static_cast<int>(g.getIndex());
            if (visitedF[gi]) continue;

            double transport = gaugeFaceTransport(geom, h);

            Edge e = h.edge();
            double sign = (h == e.halfedge()) ? 1.0 : -1.0;
            int eIdx = static_cast<int>(geom.edgeIndices[e]);

            std::complex<double> r =
                out.face(fi) * std::polar(1.0, transport + sign * phi(eIdx));
            out.face(gi) = r / std::abs(r);
            visitedF[gi] = 1;
            qf.push(g);
        }
    }

    // 4. Store the combing rotation = conjugate of the accumulated transport
    //    coefficient (see header comment). Roots stay identity; unvisited
    //    components stay 0+0i.
    out.vertex = out.vertex.conjugate();
    out.face   = out.face.conjugate();

    return out;
}

} // namespace nxr::manifold::connection
