#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/utilities/utilities.h"

namespace nxr::manifold::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

static long idxOrNone(size_t i) {
    return (i == INVALID_IND) ? -1L : static_cast<long>(i);
}

MeshTopology meshTopology(Manifold& m) {
    auto& mesh = m.mesh();

    const int nV = m.nV(), nE = m.nE(), nF = m.nF();
    const int nH = static_cast<int>(mesh.nHalfedges());
    const int nC = static_cast<int>(mesh.nCorners());

    MeshTopology t;
    t.nV = nV; t.nE = nE; t.nF = nF; t.nH = nH; t.nC = nC;

    t.vertexHalfedge.resize(nV);
    t.edgeHalfedge.resize(nE);
    t.faceHalfedge.resize(nF);
    // sized nH (== nCorners on a closed mesh); corner ids are halfedge-based
    // in geometry-central, so on a boundary mesh indices are sparse.
    t.cornerHalfedge.assign(nH, -1L);

    for (Vertex v : mesh.vertices())  t.vertexHalfedge[v.getIndex()] = idxOrNone(v.halfedge().getIndex());
    for (Edge e : mesh.edges())       t.edgeHalfedge  [e.getIndex()] = idxOrNone(e.halfedge().getIndex());
    for (Face f : mesh.faces())       t.faceHalfedge  [f.getIndex()] = idxOrNone(f.halfedge().getIndex());
    for (Corner c : mesh.corners())   t.cornerHalfedge[c.getIndex()] = idxOrNone(c.halfedge().getIndex());

    t.heTwin.resize(nH);      t.heNext.resize(nH);
    t.heVertex.resize(nH);    t.heEdge.resize(nH);
    t.heFace.resize(nH);      t.heCorner.resize(nH);
    t.heOrientation.resize(nH); t.heIsInterior.resize(nH);

    for (Halfedge he : mesh.halfedges()) {
        size_t i = he.getIndex();
        t.heTwin[i]       = idxOrNone(he.twin().getIndex());
        t.heNext[i]       = idxOrNone(he.next().getIndex());
        t.heVertex[i]     = idxOrNone(he.vertex().getIndex());
        t.heEdge[i]       = idxOrNone(he.edge().getIndex());
        t.heFace[i]       = he.isInterior() ? idxOrNone(he.face().getIndex())   : -1L;
        t.heCorner[i]     = he.isInterior() ? idxOrNone(he.corner().getIndex()) : -1L;
        t.heOrientation[i]= (he == he.edge().halfedge()) ? 1 : 0;
        t.heIsInterior[i] = he.isInterior() ? 1 : 0;
    }

    return t;
}

} // namespace nxr::manifold::geometry
