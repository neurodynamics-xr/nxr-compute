#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <complex>

namespace nxr::manifold::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Corner arrays are sized to nH (not nC) to be safe on boundary meshes where
// geometry-central corner ids are halfedge-based and may be sparse. On a
// closed mesh nH == nC so the size is identical. Entries for halfedges that
// have no interior corner (exterior halfedges) remain zero.
MeshGeometry meshGeometry(Manifold& m) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    const int nV = m.nV(), nE = m.nE(), nF = m.nF();
    const int nH = static_cast<int>(mesh.nHalfedges());

    geom.requireVertexDualAreas();
    geom.requireVertexAngleSums();
    geom.requireEdgeLengths();
    geom.requireEdgeCotanWeights();
    geom.requireEdgeDihedralAngles();
    geom.requireFaceAreas();
    geom.requireVertexPositions();
    geom.requireHalfedgeCotanWeights();
    geom.requireTransportVectorsAlongHalfedge();
    geom.requireTransportVectorsAcrossHalfedge();
    geom.requireHalfedgeVectorsInVertex();
    geom.requireHalfedgeVectorsInFace();
    geom.requireCornerAngles();
    geom.requireCornerScaledAngles();

    MeshGeometry g;

    // ── vertex ────────────────────────────────────────────────
    g.vertexDualAreas.resize(nV);
    g.vertexAngleSums.resize(nV);
    for (Vertex v : mesh.vertices()) {
        int i = static_cast<int>(v.getIndex());
        g.vertexDualAreas(i) = geom.vertexDualAreas[v];
        g.vertexAngleSums(i) = geom.vertexAngleSums[v];
    }
    auto cv = vertexCurvature(m);
    g.vertexCurvatureDeviatoric = cv.deviatoric;
    g.vertexMeanCurvature       = cv.mean;
    g.vertexGrid                = vertexGrid(m);

    // ── edge ──────────────────────────────────────────────────
    g.edgeLengths.resize(nE);
    g.edgeCotanWeights.resize(nE);
    g.edgeDihedralAngles.resize(nE);
    for (Edge e : mesh.edges()) {
        int i = static_cast<int>(e.getIndex());
        g.edgeLengths(i)        = geom.edgeLengths[e];
        g.edgeCotanWeights(i)   = geom.edgeCotanWeights[e];
        g.edgeDihedralAngles(i) = geom.edgeDihedralAngles[e];
    }

    // ── face ──────────────────────────────────────────────────
    g.faceAreas.resize(nF);
    g.faceCentroids.resize(nF, 3);
    for (Face f : mesh.faces()) {
        int i = static_cast<int>(f.getIndex());
        g.faceAreas(i) = geom.faceAreas[f];
        Vector3 c{0.0, 0.0, 0.0};
        int k = 0;
        for (Vertex v : f.adjacentVertices()) {
            c += geom.vertexPositions[v];
            ++k;
        }
        c /= static_cast<double>(k);
        g.faceCentroids(i, 0) = c.x;
        g.faceCentroids(i, 1) = c.y;
        g.faceCentroids(i, 2) = c.z;
    }
    g.faceGrid = faceGrid(m);

    // ── halfedge ──────────────────────────────────────────────
    g.halfedgeCotanWeights.resize(nH);
    g.halfedgeVectorsInVertex.resize(nH);
    g.halfedgeVectorsInFace.resize(nH);
    g.halfedgeTransportAlong.resize(nH);
    g.halfedgeTransportAcross.resize(nH);
    for (Halfedge he : mesh.halfedges()) {
        int i = static_cast<int>(he.getIndex());
        g.halfedgeCotanWeights(i) = geom.halfedgeCotanWeights[he];
        Vector2 a = geom.halfedgeVectorsInVertex[he];
        g.halfedgeVectorsInVertex(i) = {a.x, a.y};
        Vector2 b = geom.halfedgeVectorsInFace[he];
        g.halfedgeVectorsInFace(i) = {b.x, b.y};
        Vector2 t = geom.transportVectorsAlongHalfedge[he];
        g.halfedgeTransportAlong(i) = {t.x, t.y};
        Vector2 u = geom.transportVectorsAcrossHalfedge[he];
        g.halfedgeTransportAcross(i) = {u.x, u.y};
    }

    // ── corner (sized nH; on closed mesh nH==nC) ──────────────
    g.cornerAngles.resize(nH);
    g.cornerAngles.setZero();
    g.cornerScaledAngles.resize(nH);
    g.cornerScaledAngles.setZero();
    for (Corner c : mesh.corners()) {
        int i = static_cast<int>(c.getIndex());
        if (i >= 0 && i < nH) {
            g.cornerAngles(i)       = geom.cornerAngles[c];
            g.cornerScaledAngles(i) = geom.cornerScaledAngles[c];
        }
    }

    g.totalArea = g.faceAreas.sum();
    return g;
}

} // namespace nxr::manifold::geometry
