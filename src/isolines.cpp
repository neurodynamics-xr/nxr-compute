#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>

namespace nxr::field::extract {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Port of bioctreeapp/src/lib/compute/visualization/isoline.ts
// For each triangle, for each contour level, find the exactly 2 edges
// where the level crosses and emit a line segment between those crossings.
IsolineResult isoline(
    Manifold& m,
    const Eigen::VectorXd& scalarField,
    int numLevels,
    double minValue,
    double maxValue
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    int nV = m.nV();

    IsolineResult result;
    result.segmentCount = 0;

    if (numLevels < 2 || scalarField.size() != nV) {
        result.positions.resize(0, 3);
        return result;
    }

    // Auto-detect range if not provided
    double lo = minValue;
    double hi = maxValue;
    if (minValue == 0.0 && maxValue == 0.0) {
        lo = std::numeric_limits<double>::infinity();
        hi = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < nV; i++) {
            double v = scalarField(i);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    }

    double range = hi - lo;
    if (range <= 0) {
        result.positions.resize(0, 3);
        return result;
    }

    double levelSpacing = range / numLevels;

    // Collect segments into a dynamic vector of (x,y,z) endpoints
    std::vector<double> pts;

    for (Face f : mesh.faces()) {
        Halfedge h = f.halfedge();
        Vertex v0 = h.tailVertex();
        Vertex v1 = h.next().tailVertex();
        Vertex v2 = h.next().next().tailVertex();

        int i0 = static_cast<int>(v0.getIndex());
        int i1 = static_cast<int>(v1.getIndex());
        int i2 = static_cast<int>(v2.getIndex());

        double d0 = scalarField(i0);
        double d1 = scalarField(i1);
        double d2 = scalarField(i2);

        Vector3 p0 = geom.inputVertexPositions[v0];
        Vector3 p1 = geom.inputVertexPositions[v1];
        Vector3 p2 = geom.inputVertexPositions[v2];

        // Three edges: (v0,v1), (v1,v2), (v2,v0)
        struct Edge { int a, b; double da, db; Vector3 pa, pb; };
        Edge edges[3] = {
            {i0, i1, d0, d1, p0, p1},
            {i1, i2, d1, d2, p1, p2},
            {i2, i0, d2, d0, p2, p0},
        };

        for (int level = 1; level < numLevels; level++) {
            double levelValue = lo + level * levelSpacing;

            Vector3 crossings[2];
            int crossCount = 0;

            for (int e = 0; e < 3; e++) {
                double edgeLo = std::min(edges[e].da, edges[e].db);
                double edgeHi = std::max(edges[e].da, edges[e].db);
                if (levelValue > edgeLo && levelValue < edgeHi) {
                    double t = (levelValue - edges[e].da) / (edges[e].db - edges[e].da);
                    Vector3 p = edges[e].pa + (edges[e].pb - edges[e].pa) * t;
                    if (crossCount < 2) crossings[crossCount] = p;
                    crossCount++;
                }
            }

            if (crossCount == 2) {
                pts.push_back(crossings[0].x);
                pts.push_back(crossings[0].y);
                pts.push_back(crossings[0].z);
                pts.push_back(crossings[1].x);
                pts.push_back(crossings[1].y);
                pts.push_back(crossings[1].z);
            }
        }
    }

    int segCount = static_cast<int>(pts.size() / 6);
    result.segmentCount = segCount;
    result.positions.resize(segCount * 2, 3);
    for (int i = 0; i < segCount * 2; i++) {
        result.positions(i, 0) = pts[i * 3 + 0];
        result.positions(i, 1) = pts[i * 3 + 1];
        result.positions(i, 2) = pts[i * 3 + 2];
    }

    std::cout << "[isoline] Extracted " << segCount << " segments across "
              << numLevels << " levels" << std::endl;

    return result;
}

} // namespace nxr::field::extract