#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace nxr::manifold::differential {

// Build Fv = [e1 | e2 | n] (columns) for vertex v from precomputed frames.
static Eigen::Matrix3d frameOf(const geometry::VertexFrames& vf, int v) {
    Eigen::Matrix3d Fv;
    Fv.col(0) = vf.e1.row(v).transpose();
    Fv.col(1) = vf.e2.row(v).transpose();
    Fv.col(2) = vf.normals.row(v).transpose();
    return Fv;
}

Eigen::Matrix3d frameTransport(Manifold& m, int i, int j) {
    const int nV = m.nV();
    if (i < 0 || i >= nV || j < 0 || j >= nV)
        throw Error(ErrorCode::InvalidInput,
            "frameTransport: vertex index out of range",
            "Expected 0 <= i,j < " + std::to_string(nV) + ".");
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    return frameOf(vf, j).transpose() * frameOf(vf, i);   // P_ij = Fj^T Fi
}

Eigen::MatrixXd vertexFrameMatrices(Manifold& m) {
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    const int nV = m.nV();
    Eigen::MatrixXd out(nV, 9);
    for (int v = 0; v < nV; ++v) {
        Eigen::Matrix3d Fv = frameOf(vf, v);
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q)
                out(v, 3*p + q) = Fv(p, q);   // row-major flatten
    }
    return out;
}

Eigen::MatrixXd liftToWorld(Manifold& m, const Eigen::MatrixXd& Lloc) {
    const int nV = m.nV();
    if (Lloc.rows() != nV || Lloc.cols() != 3)
        throw Error(ErrorCode::InvalidInput,
            "liftToWorld: field must be [nV, 3]",
            "Expected [" + std::to_string(nV) + ", 3].");
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    Eigen::MatrixXd out(nV, 3);
    for (int v = 0; v < nV; ++v)
        out.row(v) = (frameOf(vf, v) * Lloc.row(v).transpose()).transpose();   // Fv * local
    return out;
}

Eigen::MatrixXd liftToFrame(Manifold& m, const Eigen::MatrixXd& Lworld) {
    const int nV = m.nV();
    if (Lworld.rows() != nV || Lworld.cols() != 3)
        throw Error(ErrorCode::InvalidInput,
            "liftToFrame: field must be [nV, 3]",
            "Expected [" + std::to_string(nV) + ", 3].");
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    Eigen::MatrixXd out(nV, 3);
    for (int v = 0; v < nV; ++v)
        out.row(v) = (frameOf(vf, v).transpose() * Lworld.row(v).transpose()).transpose();  // Fv^T * world
    return out;
}

Eigen::SparseMatrix<double> covariantGradient(Manifold& m) {
    using namespace geometrycentral::surface;
    geometry::VertexFrames vf = geometry::vertexFrames(m);
    auto& mesh = m.mesh();
    const int N = m.nV();
    const int E = m.nE();

    std::vector<Eigen::Triplet<double>> T;
    T.reserve(static_cast<size_t>(E) * 12);   // 3 (+I) + 9 (-P) per edge

    for (Edge e : mesh.edges()) {
        Halfedge he = e.halfedge();                       // canonical orientation
        const int i = static_cast<int>(he.tailVertex().getIndex());
        const int j = static_cast<int>(he.tipVertex().getIndex());
        const int eIdx = static_cast<int>(e.getIndex());

        // P_ij = Fj^T Fi  (transport i -> j)
        Eigen::Matrix3d Pij = frameOf(vf, j).transpose() * frameOf(vf, i);

        // delta_e[p] = L_j[p] - sum_q P_ij[p,q] L_i[q],  component-major rows {E*p + eIdx}.
        for (int p = 0; p < 3; ++p) {
            T.emplace_back(E*p + eIdx, N*p + j, 1.0);                 // +I at vertex j
            for (int q = 0; q < 3; ++q)
                T.emplace_back(E*p + eIdx, N*q + i, -Pij(p, q));      // -P_ij at vertex i
        }
    }

    Eigen::SparseMatrix<double> G(3*E, 3*N);
    G.setFromTriplets(T.begin(), T.end());
    G.makeCompressed();
    return G;
}

} // namespace nxr::manifold::differential
