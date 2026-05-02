#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/direction_fields.h"

#include <iostream>
#include <cmath>
#include <complex>

namespace nxr::compute {

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

// For an n-RoSy field, geometry-central stores a single Vector2 per
// face/vertex whose argument is n × actual angle. The "actual"
// representative direction is therefore the n-th root of that
// complex number, picked at the principal branch. We return the
// principal representative; consumers wanting the full N symmetric
// directions can multiply by exp(2πik/n) for k = 0…n-1.
Vector2 nRosyRepresentative(const Vector2& v, int nSym) {
    if (nSym <= 1) return v;
    std::complex<double> c(v.x, v.y);
    std::complex<double> root = std::pow(c, 1.0 / static_cast<double>(nSym));
    return {root.real(), root.imag()};
}

Vector3 lift(const Vector2& t, const std::array<Vector3, 2>& basis) {
    return t.x * basis[0] + t.y * basis[1];
}

} // namespace

Eigen::MatrixXd computeSmoothFaceField(
    ComputeContext& ctx,
    int nSym,
    bool alignToCurvature
) {
    if (nSym < 1) {
        throw Error(ErrorCode::InvalidInput,
            "computeSmoothFaceField: nSym must be >= 1");
    }
    auto& mesh = ctx.mesh();
    auto& geom = ctx.geometry();

    geom.requireFaceTangentBasis();

    FaceData<Vector2> field;
    if (alignToCurvature) {
        field = computeCurvatureAlignedFaceDirectionField(geom, nSym);
    } else if (mesh.hasBoundary()) {
        field = computeSmoothestBoundaryAlignedFaceDirectionField(geom, nSym);
    } else {
        field = computeSmoothestFaceDirectionField(geom, nSym);
    }

    int nF = ctx.nF();
    Eigen::MatrixXd out(nF, 3);
    for (Face f : mesh.faces()) {
        int fi = static_cast<int>(f.getIndex());
        Vector2 rep = nRosyRepresentative(field[f], nSym);
        Vector3 w = lift(rep, geom.faceTangentBasis[f]);
        out(fi, 0) = w.x; out(fi, 1) = w.y; out(fi, 2) = w.z;
    }
    std::cout << "[smooth_field] face field, nSym=" << nSym
              << ", curvatureAligned=" << alignToCurvature
              << ", nF=" << nF << std::endl;
    return out;
}

SmoothFieldResult computeSmoothVertexField(
    ComputeContext& ctx,
    int nSym,
    bool alignToCurvature
) {
    if (nSym < 1) {
        throw Error(ErrorCode::InvalidInput,
            "computeSmoothVertexField: nSym must be >= 1");
    }
    auto& mesh = ctx.mesh();
    auto& geom = ctx.geometry();

    geom.requireVertexTangentBasis();
    geom.requireFaceTangentBasis();

    VertexData<Vector2> vfield;
    if (alignToCurvature) {
        vfield = computeCurvatureAlignedVertexDirectionField(geom, nSym);
    } else if (mesh.hasBoundary()) {
        vfield = computeSmoothestBoundaryAlignedVertexDirectionField(geom, nSym);
    } else {
        vfield = computeSmoothestVertexDirectionField(geom, nSym);
    }

    int nV = ctx.nV();
    int nF = ctx.nF();

    SmoothFieldResult out;
    out.nSym = nSym;
    out.vertexVectors.resize(nV, 3);
    out.vertexFieldRaw.resize(nV * 2);
    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(v.getIndex());
        Vector2 raw = vfield[v];
        out.vertexFieldRaw(vi * 2 + 0) = raw.x;
        out.vertexFieldRaw(vi * 2 + 1) = raw.y;
        Vector2 rep = nRosyRepresentative(raw, nSym);
        Vector3 w = lift(rep, geom.vertexTangentBasis[v]);
        out.vertexVectors(vi, 0) = w.x;
        out.vertexVectors(vi, 1) = w.y;
        out.vertexVectors(vi, 2) = w.z;
    }
    // Face vectors not produced for vertex fields — leave empty.
    out.faceVectors.resize(0, 3);
    out.faceFieldRaw.resize(0);
    (void)nF;
    std::cout << "[smooth_field] vertex field, nSym=" << nSym
              << ", curvatureAligned=" << alignToCurvature
              << ", nV=" << nV << std::endl;
    return out;
}

} // namespace nxr::compute
