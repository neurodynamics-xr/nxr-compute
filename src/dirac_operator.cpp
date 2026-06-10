#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <array>
#include <vector>

namespace nxr::manifold::ops::dirac {

// 4×4 real matrix of LEFT-multiplication by a purely imaginary quaternion
// v = x·i + y·j + z·k, in component order [w,x,y,z]. Antisymmetric, with an
// identically-zero diagonal (from the integer literals below — exact in IEEE
// 754). The `if (B(a,b) != 0.0)` filter in the assembly loop relies on that:
// it skips precisely these structural diagonal zeros, never a rounded-to-zero
// off-diagonal (those come from normal differences and are nonzero off flats).
static Eigen::Matrix4d leftMulImag(const Eigen::Vector3d& v) {
    const double x = v.x(), y = v.y(), z = v.z();
    Eigen::Matrix4d L;
    L << 0, -x, -y, -z,
         x,  0, -z,  y,
         y,  z,  0, -x,
         z, -y,  x,  0;
    return L;
}

Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m) {
    using namespace geometrycentral::surface;

    // Gauss map: per-vertex unit normals from the embedded geometry. The Dirac
    // extrinsic term is genuinely extrinsic, so it always uses the TRUE embedding
    // (geometry()), independent of intrinsicDelaunay normalization.
    geometry::VertexFrames vf = geometry::vertexFrames(m);   // vf.normals = N [nV,3]
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireFaceAreas();

    const int N = m.nV();
    const int Fn = m.nF();

    // Rectangular real Dirac matrix D : 4F × 4V.
    // Per face f=ijk, area A: for each cyclic (p,q,r), block on column p is
    //   D_{f,p} = −leftMulImag(N_r − N_q) / (2A).
    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve(static_cast<size_t>(Fn) * 3 * 16);

    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        // Fail loud (CLAUDE.md §6): a single zero-area face would poison the
        // whole E with inf/NaN via the 1/(2A) below — diagnose, don't propagate.
        if (A <= 0.0)
            throw Error(ErrorCode::InvalidInput,
                "dirac::extrinsicBlock: degenerate (zero-area) face",
                "Face index " + std::to_string(fi) + "; fix mesh quality first.");
        std::array<int,3> vid{};
        int c = 0;
        for (Vertex v : f.adjacentVertices()) vid[c++] = static_cast<int>(v.getIndex());

        for (int s = 0; s < 3; ++s) {
            const int p = vid[s];
            const int q = vid[(s + 1) % 3];
            const int r = vid[(s + 2) % 3];
            Eigen::Vector3d Nr = vf.normals.row(r).transpose();
            Eigen::Vector3d Nq = vf.normals.row(q).transpose();
            Eigen::Matrix4d B = (-1.0 / (2.0 * A)) * leftMulImag(Nr - Nq);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (B(a, b) != 0.0)
                        TD.emplace_back(4 * fi + a, 4 * p + b, B(a, b));
        }
    }
    Eigen::SparseMatrix<double> D(4 * Fn, 4 * N);
    D.setFromTriplets(TD.begin(), TD.end());

    // Face-area 2-form mass ⋆_F : 4F × 4F diagonal (A_f on each of the 4 rows).
    std::vector<Eigen::Triplet<double>> TW;
    TW.reserve(static_cast<size_t>(4) * Fn);
    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        for (int a = 0; a < 4; ++a) TW.emplace_back(4 * fi + a, 4 * fi + a, A);
    }
    Eigen::SparseMatrix<double> WF(4 * Fn, 4 * Fn);
    WF.setFromTriplets(TW.begin(), TW.end());

    // Materialise D.transpose() into its own matrix before the multiply chain:
    // the lazy Transpose proxy shares D's storage, and chaining * WF * D on it
    // would alias. The explicit temporary forces evaluation first.
    Eigen::SparseMatrix<double> E = (Eigen::SparseMatrix<double>(D.transpose()) * WF * D).pruned();
    E.makeCompressed();
    return E;
}

}  // namespace nxr::manifold::ops::dirac
