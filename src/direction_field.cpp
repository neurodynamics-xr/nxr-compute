/**
 * direction_field.cpp — Trivial Connections + streamline tracing.
 *
 * Ports geometry-processing-js projects/direction-field-design/
 *   - trivial-connections.js
 *   - streamlines.js
 *
 * Reference: "Trivial Connections on Discrete Surfaces"
 *            Crane, Desbrun, Schröder (2010)
 *
 * Algorithm summary:
 *   Given singularity indices σ_v at vertices (Gauss-Bonnet: Σ σ_v = χ),
 *   solve for a connection φ on edges such that a vector transported
 *   around any vertex picks up the prescribed angular defect.
 *
 *     φ = δβ + γ
 *
 *   where:
 *     - δβ is the co-exact component, β solves A β = -K + 2π σ
 *       (A = d0ᵀ ⋆₁ d0 is the 0-form Laplacian, K is Gaussian curvature)
 *     - γ is the harmonic component, zero for simply-connected surfaces
 *       (we currently don't implement the harmonic component — it only
 *        matters for genus > 0)
 */

#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>
#include <queue>
#include <cmath>
#include <vector>
#include <unordered_map>

namespace nxr::manifold::connection {

using namespace geometrycentral;
using namespace geometrycentral::surface;
using nxr::manifold::ops::DECOperators;
using nxr::manifold::ops::CholeskyCache;

// ─────────────────────────────────────────────────────────────
// Helper: 2D orthonormal basis on a face from its normal and
// first halfedge direction.
// ─────────────────────────────────────────────────────────────

static std::array<Vector3, 2> faceOrthonormalBasis(
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

// ─────────────────────────────────────────────────────────────
// transportNoRotation: Crane's parallel transport formula.
// αⱼ = αᵢ - θᵢⱼ + θⱼᵢ
// ─────────────────────────────────────────────────────────────

static double transportNoRotation(
    VertexPositionGeometry& geom, Halfedge h
) {
    if (!h.isInterior() || !h.twin().isInterior()) return 0.0;

    // Edge vector (tail → tip) of this halfedge
    Vector3 u = geom.inputVertexPositions[h.tipVertex()] -
                geom.inputVertexPositions[h.tailVertex()];

    // Bases for the two faces sharing this halfedge
    auto [e1, e2] = faceOrthonormalBasis(geom, h.face());
    double thetaIJ = std::atan2(dot(u, e2), dot(u, e1));

    auto [f1, f2] = faceOrthonormalBasis(geom, h.twin().face());
    double thetaJI = std::atan2(dot(u, f2), dot(u, f1));

    return -thetaIJ + thetaJI;
}

// ─────────────────────────────────────────────────────────────
// Solve Aβ = -K + 2π σ for the co-exact scalar potential β,
// then return δβ = ⋆₁ d₀ β (as an edge 1-form).
// ─────────────────────────────────────────────────────────────

static Eigen::VectorXd computeCoExactComponent(
    Manifold& m,
    const DECOperators& dec,
    CholeskyCache& cache,
    const std::map<int, double>& singularityMap,
    double& eulerChiOut
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    int nV = m.nV();

    geom.requireVertexGaussianCurvatures();

    // Build RHS: -K + 2π σ per vertex
    Eigen::VectorXd rhs(nV);
    double totalK = 0.0;
    for (Vertex v : mesh.vertices()) {
        int i = static_cast<int>(v.getIndex());
        double K = geom.vertexGaussianCurvatures[v];
        totalK += K;
        double sigma = 0.0;
        auto it = singularityMap.find(i);
        if (it != singularityMap.end()) sigma = it->second;
        rhs(i) = -K + 2.0 * M_PI * sigma;
    }

    // Euler characteristic (from Gauss-Bonnet: 2πχ = ΣK)
    eulerChiOut = totalK / (2.0 * M_PI);

    // Solve A β = rhs, where A = d0ᵀ ★₁ d0 + ε·I.
    // Same matrix as the Hodge α-solve, so the cached factor is shared.
    const auto& llt = cache.hodgeExact(dec);
    Eigen::VectorXd betaTilde = llt.solve(rhs);

    // δβ = ⋆₁ d₀ β (per-edge 1-form)
    return dec.hodge1 * (dec.d0 * betaTilde);
}

// ─────────────────────────────────────────────────────────────
// Propagate angles across faces via dual-spanning-tree BFS,
// following the connection φ on edges. Returns per-face angle
// α(f) that defines the direction field.
// ─────────────────────────────────────────────────────────────

static Eigen::VectorXd propagateAngles(
    Manifold& m,
    const Eigen::VectorXd& phi,   // connection 1-form on edges
    const DECOperators& /*dec*/
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    int nF = m.nF();

    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(nF);
    std::vector<bool> visited(nF, false);

    // BFS from face 0
    std::queue<Face> queue;
    Face root = mesh.face(0);
    visited[0] = true;
    alpha(0) = 0.0;
    queue.push(root);

    while (!queue.empty()) {
        Face f = queue.front();
        queue.pop();
        int fi = static_cast<int>(f.getIndex());
        double alphaI = alpha(fi);

        for (Halfedge h : f.adjacentHalfedges()) {
            if (!h.twin().isInterior()) continue;
            Face g = h.twin().face();
            int gi = static_cast<int>(g.getIndex());
            if (visited[gi]) continue;

            double transport = transportNoRotation(geom, h);

            // Sign of the connection φ on this edge, depending on halfedge orientation
            Edge e = h.edge();
            double sign = (h == e.halfedge()) ? 1.0 : -1.0;
            double phiEdge = phi(static_cast<int>(e.getIndex()));

            double alphaJ = alphaI + transport + sign * phiEdge;
            alpha(gi) = alphaJ;
            visited[gi] = true;
            queue.push(g);
        }
    }

    return alpha;
}

// ─────────────────────────────────────────────────────────────
// Build per-face direction vectors from per-face angles.
// For each face f, the direction vector is:
//   u = cos(α(f)) e₁(f) + sin(α(f)) e₂(f)
// where {e₁, e₂} is the face tangent basis.
// ─────────────────────────────────────────────────────────────

static Eigen::MatrixXd buildFaceVectorsFromAngles(
    Manifold& m,
    const Eigen::VectorXd& alpha
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    int nF = m.nF();

    Eigen::MatrixXd out(nF, 3);
    for (Face f : mesh.faces()) {
        int fi = static_cast<int>(f.getIndex());
        auto [e1, e2] = faceOrthonormalBasis(geom, f);
        Vector3 u = e1 * std::cos(alpha(fi)) + e2 * std::sin(alpha(fi));
        out(fi, 0) = u.x;
        out(fi, 1) = u.y;
        out(fi, 2) = u.z;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────
// trivial — public entry point
// ─────────────────────────────────────────────────────────────

DirectionFieldResult trivial(
    Manifold& m,
    const DECOperators& dec,
    CholeskyCache& cache,
    const std::map<int, double>& singularityMap
) {
    DirectionFieldResult result;
    result.eulerCharacteristic = 0.0;
    result.gaussBonnetSatisfied = false;

    // 1. Co-exact component δβ (vertex singularities → connection)
    result.connections = computeCoExactComponent(m, dec, cache, singularityMap,
                                                 result.eulerCharacteristic);

    // 2. Gauss-Bonnet check
    double sumSing = 0.0;
    for (const auto& [_, s] : singularityMap) sumSing += s;
    result.gaussBonnetSatisfied =
        std::abs(sumSing - result.eulerCharacteristic) < 1e-3;

    // 3. Propagate angles via dual spanning tree
    Eigen::VectorXd alpha = propagateAngles(m, result.connections, dec);

    // 4. Build direction field vectors
    result.directionVectors = buildFaceVectorsFromAngles(m, alpha);

    // 5. Orthogonal field: rotate by 90° on each face (α + π/2)
    Eigen::VectorXd alphaOrth = alpha.array() + M_PI / 2.0;
    result.orthogonalVectors = buildFaceVectorsFromAngles(m, alphaOrth);

    std::cout << "[direction_field] χ=" << result.eulerCharacteristic
              << ", Σσ=" << sumSing
              << ", Gauss-Bonnet "
              << (result.gaussBonnetSatisfied ? "OK" : "VIOLATED")
              << std::endl;

    return result;
}

} // namespace nxr::manifold::connection

// ─────────────────────────────────────────────────────────────
// Streamline Tracing
// ─────────────────────────────────────────────────────────────

namespace nxr::field::extract {

using namespace geometrycentral;
using namespace geometrycentral::surface;
using nxr::manifold::Manifold;

namespace {

// Barycentric coordinates of point p in triangle (p0, p1, p2).
// Returns (u, v, w) with u + v + w = 1.
struct Bary { double u, v, w; };
Bary barycentric(Vector3 p, Vector3 p0, Vector3 p1, Vector3 p2) {
    Vector3 e1 = p1 - p0;
    Vector3 e2 = p2 - p0;
    Vector3 ep = p - p0;
    double d00 = dot(e1, e1);
    double d01 = dot(e1, e2);
    double d11 = dot(e2, e2);
    double d20 = dot(ep, e1);
    double d21 = dot(ep, e2);
    double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-12) return {0, 0, 1};
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    return {1.0 - v - w, v, w};
}

// Line segment p1→p2 vs triangle edge pa→pb (in 2D face-local coords).
// Returns (crosses, t, intersection3D) where t ∈ [0,1] along p1→p2.
struct IntersectResult {
    bool crosses;
    Vector3 intersection;
};

IntersectResult lineIntersectsEdge2D(
    Vector3 p1, Vector3 p2, Vector3 pa, Vector3 pb,
    Vector3 basis1, Vector3 basis2
) {
    auto to2D = [&](Vector3 p) -> std::array<double, 2> {
        return {dot(p, basis1), dot(p, basis2)};
    };
    auto [p1x, p1y] = to2D(p1);
    auto [p2x, p2y] = to2D(p2);
    auto [pax, pay] = to2D(pa);
    auto [pbx, pby] = to2D(pb);

    double det = (pbx - pax) * (p1y - p2y) - (p1x - p2x) * (pby - pay);
    if (std::abs(det) < 1e-12) return {false, {}};

    double t = ((p1x - pax) * (p1y - p2y) - (p1y - pay) * (p1x - p2x)) / det;
    double s = ((pbx - pax) * (p1y - pay) - (pby - pay) * (p1x - pax)) / det;

    if (t >= 0 && t <= 1 && s >= 0 && s <= 1) {
        Vector3 inter = pa + (pb - pa) * t;
        return {true, inter};
    }
    return {false, {}};
}

} // namespace (anonymous)

StreamlineResult streamline(
    Manifold& m,
    const Eigen::MatrixXd& faceField,
    int numSeeds,
    double stepCoef,
    int maxSteps
) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();

    geom.requireFaceNormals();
    geom.requireEdgeLengths();

    // Mean edge length
    double meanEdge = 0.0;
    int nE = m.nE();
    for (Edge e : mesh.edges()) meanEdge += geom.edgeLengths[e];
    meanEdge /= std::max(1, nE);
    double stepSize = meanEdge * stepCoef;

    // Seed faces — quasi-uniform across mesh
    int nF = m.nF();
    int seedStep = std::max(1, nF / (numSeeds * numSeeds));

    std::vector<double> outPts;  // flat [x,y,z, x,y,z, ...] endpoint pairs

    // Trace a single streamline forward from (face, position).
    // Returns a list of points along the streamline.
    auto trace = [&](Face seedFace, Vector3 seedPos, bool negate)
        -> std::vector<Vector3>
    {
        std::vector<Vector3> curve;
        curve.push_back(seedPos);
        Face currentFace = seedFace;
        Vector3 currentPos = seedPos;

        for (int step = 0; step < maxSteps; step++) {
            int fi = static_cast<int>(currentFace.getIndex());
            Vector3 dir{faceField(fi, 0), faceField(fi, 1), faceField(fi, 2)};
            if (negate) dir = -dir;
            double mag = dir.norm();
            if (mag < 1e-12) break;
            dir /= mag;

            Vector3 nextPos = currentPos + dir * stepSize;

            // Try to stay in current face
            Vector3 p0, p1, p2;
            Halfedge hf = currentFace.halfedge();
            p0 = geom.inputVertexPositions[hf.tailVertex()];
            p1 = geom.inputVertexPositions[hf.next().tailVertex()];
            p2 = geom.inputVertexPositions[hf.next().next().tailVertex()];
            Bary b = barycentric(nextPos, p0, p1, p2);
            double eps = -0.001;
            if (b.u >= eps && b.v >= eps && b.w >= eps) {
                currentPos = nextPos;
                curve.push_back(currentPos);
                continue;
            }

            // Need to cross an edge — find which one
            auto [basis1, basis2] = nxr::manifold::connection::faceOrthonormalBasis(geom, currentFace);
            bool crossed = false;
            for (Halfedge h : currentFace.adjacentHalfedges()) {
                Vector3 pa = geom.inputVertexPositions[h.tailVertex()];
                Vector3 pb = geom.inputVertexPositions[h.tipVertex()];
                auto result = lineIntersectsEdge2D(
                    currentPos, nextPos, pa, pb, basis1, basis2);
                if (result.crosses) {
                    if (!h.twin().isInterior()) {
                        // Hit boundary, stop
                        break;
                    }
                    currentFace = h.twin().face();
                    currentPos = result.intersection + (nextPos - result.intersection) * 0.5;
                    // Just use intersection point to avoid overshooting into wrong face
                    currentPos = result.intersection;
                    curve.push_back(currentPos);
                    crossed = true;
                    break;
                }
            }
            if (!crossed) break;
        }
        return curve;
    };

    int nCurves = 0;
    for (int si = 0; si < nF; si += seedStep) {
        Face f = mesh.face(si);
        // Centroid
        Halfedge h = f.halfedge();
        Vector3 p0 = geom.inputVertexPositions[h.tailVertex()];
        Vector3 p1 = geom.inputVertexPositions[h.next().tailVertex()];
        Vector3 p2 = geom.inputVertexPositions[h.next().next().tailVertex()];
        Vector3 centroid = (p0 + p1 + p2) / 3.0;

        auto forward = trace(f, centroid, false);
        auto backward = trace(f, centroid, true);

        // Combine: reversed backward + forward (skip duplicate start)
        std::vector<Vector3> full(backward.rbegin(), backward.rend());
        for (size_t i = 1; i < forward.size(); i++) full.push_back(forward[i]);

        // Emit line segments
        for (size_t i = 0; i + 1 < full.size(); i++) {
            outPts.push_back(full[i].x);
            outPts.push_back(full[i].y);
            outPts.push_back(full[i].z);
            outPts.push_back(full[i + 1].x);
            outPts.push_back(full[i + 1].y);
            outPts.push_back(full[i + 1].z);
        }
        if (full.size() >= 2) nCurves++;
    }

    StreamlineResult result;
    int segCount = static_cast<int>(outPts.size() / 6);
    result.segmentCount = segCount;
    result.positions.resize(segCount * 2, 3);
    for (int i = 0; i < segCount * 2; i++) {
        result.positions(i, 0) = outPts[i * 3 + 0];
        result.positions(i, 1) = outPts[i * 3 + 1];
        result.positions(i, 2) = outPts[i * 3 + 2];
    }

    std::cout << "[streamlines] " << nCurves << " curves, "
              << segCount << " total segments" << std::endl;

    return result;
}

} // namespace nxr::field::extract
