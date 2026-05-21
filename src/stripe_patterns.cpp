#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/stripe_patterns.h"
#include "geometrycentral/utilities/vector3.h"

#include <iostream>
#include <vector>

namespace nxr::manifold::parametrization::stripes {

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

StripePatternResult buildStripeResult(
    Manifold& m,
    const VertexData<Vector2>& field,
    const VertexData<double>& freqs,
    bool connectOnSingularities
) {
    auto& geom = m.geometry();

    // The convenience overload returns 3D polyline segments directly;
    // skip the per-corner phase intermediate that's only needed if you
    // want to render the underlying scalar function.
    auto result = computeStripePatternPolylines(
        geom, freqs, field, connectOnSingularities);

    const auto& points = std::get<0>(result);
    const auto& edges  = std::get<1>(result);
    int segCount = static_cast<int>(edges.size());

    StripePatternResult out;
    out.segmentCount = segCount;
    out.positions.resize(static_cast<Eigen::Index>(segCount) * 2, 3);
    for (int i = 0; i < segCount; i++) {
        const Vector3& p0 = points[edges[i][0]];
        const Vector3& p1 = points[edges[i][1]];
        out.positions(i * 2 + 0, 0) = p0.x;
        out.positions(i * 2 + 0, 1) = p0.y;
        out.positions(i * 2 + 0, 2) = p0.z;
        out.positions(i * 2 + 1, 0) = p1.x;
        out.positions(i * 2 + 1, 1) = p1.y;
        out.positions(i * 2 + 1, 2) = p1.z;
    }
    std::cout << "[stripes] " << segCount << " segments" << std::endl;
    return out;
}

VertexData<Vector2> unpackVertexField(SurfaceMesh& mesh,
                                      const Eigen::VectorXd& raw) {
    int nV = static_cast<int>(mesh.nVertices());
    if (raw.size() != static_cast<Eigen::Index>(nV) * 2) {
        throw Error(ErrorCode::InvalidInput,
            "vertexFieldRaw must have length nV*2");
    }
    VertexData<Vector2> field(mesh);
    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(v.getIndex());
        field[v] = {raw(vi * 2 + 0), raw(vi * 2 + 1)};
    }
    return field;
}

} // namespace

StripePatternResult compute(
    Manifold& m,
    const Eigen::VectorXd& vertexFieldRaw,
    double uniformFrequency,
    bool connectOnSingularities
) {
    if (uniformFrequency <= 0) {
        throw Error(ErrorCode::InvalidInput,
            "uniformFrequency must be > 0");
    }
    auto& mesh = m.mesh();
    auto field = unpackVertexField(mesh, vertexFieldRaw);
    VertexData<double> freqs(mesh, uniformFrequency);
    return buildStripeResult(m, field, freqs, connectOnSingularities);
}

StripePatternResult computeFreq(
    Manifold& m,
    const Eigen::VectorXd& vertexFieldRaw,
    const Eigen::VectorXd& frequencies,
    bool connectOnSingularities
) {
    auto& mesh = m.mesh();
    int nV = m.nV();
    if (frequencies.size() != nV) {
        throw Error(ErrorCode::InvalidInput,
            "frequencies length must match nV");
    }
    auto field = unpackVertexField(mesh, vertexFieldRaw);
    VertexData<double> freqs(mesh);
    for (Vertex v : mesh.vertices()) {
        freqs[v] = frequencies(static_cast<int>(v.getIndex()));
    }
    return buildStripeResult(m, field, freqs, connectOnSingularities);
}

} // namespace nxr::manifold::parametrization::stripes