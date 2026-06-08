#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <queue>
#include <vector>
#include <complex>
#include <cmath>

namespace nxr::manifold::connection {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Integrate the trivial-connection 1-form φ into a per-vertex unit-complex
// rotation r_v = exp(iθ_v) by BFS over the primal vertex graph.
//
// Convention — matches propagateAnglesVertex in direction_field.cpp:
//   • LC transport for halfedge he (u→w):
//       rho = transportVectorsAlongHalfedge[he]    (NOT he.twin())
//   • φ sign for halfedge he:
//       sign = (he == he.edge().halfedge()) ? +1 : -1
//   so the complex update at neighbour w is:
//       r[w] = r[u] · rho_complex · exp(i · sign · φ[edge])
//
// The resulting r_v is the per-vertex complex rotation relative to the
// Levi-Civita vertex frame (geometry-central vertexTangentBasis), encoding
// the trivial-gauge frame as  r_v · vertexGrid[v].
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
    GaugeRotations out;
    out.vertex.resize(nV);
    std::vector<char> visited(nV, 0);

    // 2. BFS from vertex 0; root rotation r[0] = 1 (identity in LC frame).
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
            if (!he.isInterior()) continue;

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

    return out;
}

} // namespace nxr::manifold::connection
