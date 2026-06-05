#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/vector2.h"

#include <Eigen/SparseCore>

#include <complex>
#include <vector>

namespace nxr::manifold::ops::laplacian::connection {

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

using ComplexTriplet = Eigen::Triplet<std::complex<double>>;

// Vertex connection Laplacian on the n-RoSy bundle.
//
// Mirrors the file-static `computeVertexConnectionLaplacian` in
// `geometry-central/src/surface/direction_fields.cpp:19` — applies
// `.pow(nSym)` to the Levi-Civita transport rotation along each
// halfedge. The cotangent weight is the standard mesh edge weight;
// the connection lifts the scalar Laplacian to the complex tangent
// bundle.
void assembleVertexCL(VertexPositionGeometry& geometry,
                      int nSym,
                      std::vector<ComplexTriplet>& out) {
    SurfaceMesh& mesh = geometry.mesh;

    geometry.requireVertexIndices();
    geometry.requireEdgeCotanWeights();
    geometry.requireTransportVectorsAlongHalfedge();

    out.reserve(2 * mesh.nHalfedges());
    for (Halfedge he : mesh.halfedges()) {
        const size_t iTail = geometry.vertexIndices[he.vertex()];
        const size_t iTip  = geometry.vertexIndices[he.next().vertex()];

        const Vector2 rot    = geometry.transportVectorsAlongHalfedge[he.twin()].pow(nSym);
        const double  weight = geometry.edgeCotanWeights[he.edge()];

        out.emplace_back(iTail, iTail, std::complex<double>(weight, 0.0));
        out.emplace_back(iTail, iTip,  std::complex<double>(-weight * rot.x,
                                                            -weight * rot.y));
    }
}

// Face connection Laplacian on the n-RoSy bundle.
//
// Mirrors the file-static `computeFaceConnectionLaplacian` in
// `geometry-central/src/surface/direction_fields.cpp:52`. Inherits
// geometry-central's FIXME on face weights (uniform 1.0). The
// numerical answer is still well-defined for nRoSy energy
// minimization; revisit if a more principled face weight emerges
// upstream.
void assembleFaceCL(VertexPositionGeometry& geometry,
                    int nSym,
                    std::vector<ComplexTriplet>& out) {
    SurfaceMesh& mesh = geometry.mesh;

    geometry.requireFaceIndices();
    geometry.requireTransportVectorsAcrossHalfedge();

    out.reserve(4 * mesh.nFaces());
    for (Face f : mesh.faces()) {
        const size_t i = geometry.faceIndices[f];

        std::complex<double> weightDiagSum(0.0, 0.0);
        for (Halfedge he : f.adjacentHalfedges()) {
            if (!he.twin().isInterior()) continue;

            const Face   neigh = he.twin().face();
            const size_t j     = geometry.faceIndices[neigh];

            const Vector2 rot = geometry.transportVectorsAcrossHalfedge[he.twin()].pow(nSym);

            const double weight = 1.0;
            out.emplace_back(i, j, std::complex<double>(-weight * rot.x,
                                                        -weight * rot.y));
            weightDiagSum += weight;
        }

        out.emplace_back(i, i, weightDiagSum);
    }
}

// Crouzeix-Raviart connection Laplacian on the n-RoSy bundle.
//
// Forks `IntrinsicGeometryInterface::computeCrouzeixRaviartConnectionLaplacian`
// at `geometry-central/src/surface/intrinsic_geometry_interface.cpp:639`,
// raising the signed connection rotation `s_ij · rot_ij` to the
// nSym power. For nSym=1 the matrix matches geometry-central's
// `crouzeixRaviartConnectionLaplacian` property exactly; for even
// nSym the orientation sign drops out (line / cross fields are
// orientation-agnostic on the n-RoSy bundle).
//
// Triangle-only — geometry-central's own assertion.
void assembleEdgeCRCL(VertexPositionGeometry& geometry,
                      int nSym,
                      std::vector<ComplexTriplet>& out) {
    SurfaceMesh& mesh = geometry.mesh;

    geometry.requireEdgeIndices();
    geometry.requireHalfedgeCotanWeights();
    geometry.requireFaceAreas();
    geometry.requireEdgeLengths();

    out.reserve(12 * mesh.nFaces());
    for (Face f : mesh.faces()) {
        for (Halfedge he : f.adjacentHalfedges()) {
            const Halfedge heA = he.next();
            const Halfedge heB = heA.next();
            if (heB.next() != he) {
                throw Error(ErrorCode::InvalidInput,
                            "Connection Laplacian (Crouzeix-Raviart) requires a triangle mesh",
                            "Face " + std::to_string(f.getIndex()) +
                            " is not a triangle. Use the vertex or face domain instead.");
            }

            const size_t iE_i = geometry.edgeIndices[heA.edge()];
            const size_t iE_j = geometry.edgeIndices[heB.edge()];

            const double weight =  4.0 * geometry.halfedgeCotanWeights[he];
            const double s_ij   = (heA.orientation() == heB.orientation()) ? 1.0 : -1.0;
            const double lOpp   = geometry.edgeLengths[he.edge()];
            const double lA     = geometry.edgeLengths[heB.edge()];
            const double lB     = geometry.edgeLengths[heA.edge()];
            const double cosT   = (lA * lA + lB * lB - lOpp * lOpp) / (2.0 * lA * lB);
            const double sinT   = 2.0 * geometry.faceAreas[f] / (lA * lB);

            // signed_rot = s_ij · (-cosT + i·sinT) — unit complex on the
            // n-RoSy bundle. Raising to nSym applies the connection rotation
            // n times in the line / cross / k-RoSy field.
            const std::complex<double> signed_rot(s_ij * (-cosT), s_ij * sinT);
            const std::complex<double> rot_n = std::pow(signed_rot, nSym);

            out.emplace_back(iE_i, iE_i, std::complex<double>(weight, 0.0));
            out.emplace_back(iE_j, iE_j, std::complex<double>(weight, 0.0));
            out.emplace_back(iE_i, iE_j, -weight * rot_n);
            out.emplace_back(iE_j, iE_i, -weight * std::conj(rot_n));
        }
    }
}

// Lower a complex Hermitian matrix to its 2N × 2N symmetric real
// representation: each entry a + bi maps to the 2 × 2 block
// [[a, -b], [b, a]] placed at (i, j), (i, j+N), (i+N, j), (i+N, j+N).
// For a Hermitian K, the result is symmetric real and positive
// (semi-)definite iff K is.
Eigen::SparseMatrix<double> lowerToReal2N(
    const Eigen::SparseMatrix<std::complex<double>>& K_complex,
    int N
) {
    std::vector<Eigen::Triplet<double>> real_triplets;
    real_triplets.reserve(4 * K_complex.nonZeros());

    for (int k = 0; k < K_complex.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(K_complex, k);
             it; ++it) {
            const int    i = static_cast<int>(it.row());
            const int    j = static_cast<int>(it.col());
            const double a = it.value().real();
            const double b = it.value().imag();
            real_triplets.emplace_back(i,     j,     a);
            real_triplets.emplace_back(i + N, j + N, a);
            real_triplets.emplace_back(i,     j + N, -b);
            real_triplets.emplace_back(i + N, j,     b);
        }
    }

    Eigen::SparseMatrix<double> K_real(2 * N, 2 * N);
    K_real.setFromTriplets(real_triplets.begin(), real_triplets.end());
    return K_real;
}

} // namespace

ConnectionLaplacian assembleConnectionLaplacian(
    Manifold& m,
    const ConnectionLaplacianOptions& opts
) {
    if (opts.nSym <= 0) {
        throw Error(ErrorCode::InvalidInput,
                    "Connection Laplacian: nSym must be positive (got " +
                    std::to_string(opts.nSym) + ")",
                    "Common values: 1 (vector field), 2 (line field), 4 (cross field).");
    }

    auto& geometry = m.geometry();

    std::vector<ComplexTriplet> triplets;
    int N = 0;
    switch (opts.domain) {
        case ConnectionDomain::Vertex:
            N = m.nV();
            assembleVertexCL(geometry, opts.nSym, triplets);
            break;
        case ConnectionDomain::Face:
            N = m.nF();
            assembleFaceCL(geometry, opts.nSym, triplets);
            break;
        case ConnectionDomain::EdgeCrouzeixRaviart:
            N = m.nE();
            assembleEdgeCRCL(geometry, opts.nSym, triplets);
            break;
    }

    // Add ε·I diagonal regularization.
    if (opts.regularization != 0.0) {
        for (int i = 0; i < N; ++i) {
            triplets.emplace_back(i, i, std::complex<double>(opts.regularization, 0.0));
        }
    }

    Eigen::SparseMatrix<std::complex<double>> K_complex(N, N);
    K_complex.setFromTriplets(triplets.begin(), triplets.end());

    ConnectionLaplacian result;
    result.baseDim         = N;
    result.domain          = opts.domain;
    result.nSym            = opts.nSym;
    result.regularization  = opts.regularization;
    result.format          = opts.format;

    if (opts.format == ConnectionLaplacianFormat::Real2N) {
        result.K_real    = lowerToReal2N(K_complex, N);
        result.outputDim = 2 * N;
    } else {
        result.K_complex = std::move(K_complex);
        result.outputDim = N;
    }
    return result;
}

ConnectionDomain parseConnectionDomain(const std::string& name) {
    if (name == "vertex") return ConnectionDomain::Vertex;
    if (name == "face")   return ConnectionDomain::Face;
    if (name == "edge" || name == "crouzeix-raviart" || name == "cr")
        return ConnectionDomain::EdgeCrouzeixRaviart;
    throw Error(ErrorCode::InvalidInput,
                "Unknown connection-Laplacian domain '" + name +
                "' — expected 'vertex', 'face', or 'edge'.");
}

ConnectionLaplacianFormat parseConnectionLaplacianFormat(const std::string& name) {
    if (name == "real2N" || name == "real2n" || name == "real")
        return ConnectionLaplacianFormat::Real2N;
    if (name == "complex")
        return ConnectionLaplacianFormat::Complex;
    throw Error(ErrorCode::InvalidInput,
                "Unknown connection-Laplacian format '" + name +
                "' — expected 'real2N' or 'complex'.");
}

ConnectionLaplacian assembleTrivialConnectionLaplacian(
    Manifold& m,
    const std::map<int, double>& singularityMap,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const ConnectionLaplacianOptions& opts
) {
    if (opts.domain != ConnectionDomain::Vertex) {
        throw Error(ErrorCode::InvalidInput,
            "assembleTrivialConnectionLaplacian: only ConnectionDomain::Vertex "
            "is currently supported.",
            "Face and EdgeCrouzeixRaviart trivial-connection Laplacians are "
            "not yet implemented. Use domain='vertex'.");
    }
    if (opts.nSym <= 0) {
        throw Error(ErrorCode::InvalidInput,
            "assembleTrivialConnectionLaplacian: nSym must be positive (got " +
            std::to_string(opts.nSym) + ")",
            "Common values: 1 (vector field), 2 (line field), 4 (cross field).");
    }

    Eigen::VectorXd phi =
        nxr::manifold::connection::computeTrivialConnection(m, dec, cache, singularityMap);

    auto& geometry = m.geometry();
    SurfaceMesh& mesh = m.mesh();
    int N = m.nV();

    geometry.requireVertexIndices();
    geometry.requireEdgeCotanWeights();
    geometry.requireTransportVectorsAlongHalfedge();
    geometry.requireEdgeIndices();

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(2 * mesh.nHalfedges());

    for (Halfedge he : mesh.halfedges()) {
        const size_t iTail  = geometry.vertexIndices[he.vertex()];
        const size_t iTip   = geometry.vertexIndices[he.next().vertex()];
        const double weight = geometry.edgeCotanWeights[he.edge()];
        const int    eIdx   = static_cast<int>(geometry.edgeIndices[he.edge()]);

        const double signTwin = (he.twin() == he.edge().halfedge()) ? 1.0 : -1.0;
        const Vector2 lcRotBase = geometry.transportVectorsAlongHalfedge[he.twin()];
        const std::complex<double> lcRotC(lcRotBase.x, lcRotBase.y);
        const std::complex<double> correction = std::polar(1.0, signTwin * phi(eIdx));
        const std::complex<double> tcRotC =
            std::pow(lcRotC * correction, static_cast<double>(opts.nSym));

        triplets.emplace_back(iTail, iTail, std::complex<double>(weight, 0.0));
        triplets.emplace_back(iTail, iTip,
            std::complex<double>(-weight * tcRotC.real(), -weight * tcRotC.imag()));
    }

    if (opts.regularization != 0.0) {
        for (int i = 0; i < N; ++i)
            triplets.emplace_back(i, i, std::complex<double>(opts.regularization, 0.0));
    }

    Eigen::SparseMatrix<std::complex<double>> K_complex(N, N);
    K_complex.setFromTriplets(triplets.begin(), triplets.end());

    ConnectionLaplacian result;
    result.baseDim        = N;
    result.domain         = opts.domain;
    result.nSym           = opts.nSym;
    result.regularization = opts.regularization;
    result.format         = opts.format;

    if (opts.format == ConnectionLaplacianFormat::Real2N) {
        result.K_real    = lowerToReal2N(K_complex, N);
        result.outputDim = 2 * N;
    } else {
        result.K_complex = std::move(K_complex);
        result.outputDim = N;
    }
    return result;
}

} // namespace nxr::manifold::ops::laplacian::connection