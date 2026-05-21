#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>
#include <cmath>

namespace nxr::manifold::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Helper: geometric vector of a halfedge (tip - tail)
static Vector3 halfedgeVector(VertexPositionGeometry& geom, Halfedge h) {
    Vector3 tail = geom.inputVertexPositions[h.tailVertex()];
    Vector3 tip = geom.inputVertexPositions[h.tipVertex()];
    return tip - tail;
}

Eigen::MatrixXd normals(Manifold& m, NormalType type) {
    auto& mesh = m.mesh();
    auto& geometry = m.geometry();
    int nV = m.nV();

    geometry.requireFaceNormals();
    geometry.requireFaceAreas();
    geometry.requireCornerAngles();
    geometry.requireEdgeDihedralAngles();
    geometry.requireEdgeLengths();
    geometry.requireHalfedgeCotanWeights();

    Eigen::MatrixXd normals(nV, 3);

    for (Vertex v : mesh.vertices()) {
        Vector3 n{0.0, 0.0, 0.0};

        switch (type) {
            case NormalType::AngleWeighted: {
                for (Corner c : v.adjacentCorners()) {
                    Vector3 fn = geometry.faceNormals[c.face()];
                    double angle = geometry.cornerAngles[c];
                    n += fn * angle;
                }
                break;
            }
            case NormalType::AreaWeighted: {
                for (Face f : v.adjacentFaces()) {
                    Vector3 fn = geometry.faceNormals[f];
                    double area = geometry.faceAreas[f];
                    n += fn * area;
                }
                break;
            }
            case NormalType::EqualWeighted: {
                for (Face f : v.adjacentFaces()) {
                    n += geometry.faceNormals[f];
                }
                break;
            }
            case NormalType::SphereInscribed: {
                // Σ_c (u × v) / (|u|² |v|²)
                // where u, v are the two edge vectors at corner c
                for (Corner c : v.adjacentCorners()) {
                    // gpjs: u = vector(c.halfedge.prev)
                    //       v = vector(c.halfedge.next).negated()
                    Halfedge h = c.halfedge();
                    Vector3 u = halfedgeVector(geometry, h.next().next()); // prev
                    Vector3 vv = -halfedgeVector(geometry, h.next());
                    double denom = u.norm2() * vv.norm2();
                    if (denom > 1e-30) {
                        n += cross(u, vv) / denom;
                    }
                }
                break;
            }
            case NormalType::MeanCurvature: {
                // Σ_h 0.5 (cot(h) + cot(h.twin)) · vector(h)  (negated then normalised)
                for (Halfedge h : v.outgoingHalfedges()) {
                    double cotanSum = 0.0;
                    if (h.isInterior()) cotanSum += geometry.halfedgeCotanWeights[h];
                    if (h.twin().isInterior()) cotanSum += geometry.halfedgeCotanWeights[h.twin()];
                    double weight = 0.5 * cotanSum;
                    Vector3 vec = halfedgeVector(geometry, h);
                    n -= vec * weight;
                }
                break;
            }
            case NormalType::GaussCurvature: {
                // Σ_h 0.5 · dihedralAngle(h) / length(h.edge) · vector(h)  (negated)
                for (Halfedge h : v.outgoingHalfedges()) {
                    double len = geometry.edgeLengths[h.edge()];
                    if (len < 1e-30) continue;
                    double dihedral = geometry.edgeDihedralAngles[h.edge()];
                    double weight = 0.5 * dihedral / len;
                    Vector3 vec = halfedgeVector(geometry, h);
                    n -= vec * weight;
                }
                break;
            }
        }

        // Normalize (guard against degenerate case)
        double mag = n.norm();
        if (mag > 1e-30) n /= mag;

        int i = static_cast<int>(v.getIndex());
        normals(i, 0) = n.x;
        normals(i, 1) = n.y;
        normals(i, 2) = n.z;
    }

    const char* typeName = "?";
    switch (type) {
        case NormalType::AngleWeighted:   typeName = "angle"; break;
        case NormalType::AreaWeighted:    typeName = "area"; break;
        case NormalType::EqualWeighted:   typeName = "equal"; break;
        case NormalType::SphereInscribed: typeName = "sphere"; break;
        case NormalType::MeanCurvature:   typeName = "mean"; break;
        case NormalType::GaussCurvature:  typeName = "gauss"; break;
    }
    std::cout << "[normals] Computed " << nV << " vertex normals (" << typeName << ")" << std::endl;

    return normals;
}

} // namespace nxr::manifold::geometry