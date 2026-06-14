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

Eigen::SparseMatrix<double> matrix(Manifold& m) {
    using namespace geometrycentral::surface;

    // Gauss map: per-vertex unit normals from the embedded geometry. The Dirac
    // operator is genuinely extrinsic, so it always uses the TRUE embedding
    // (geometry()), independent of intrinsicDelaunay normalization.
    geometry::VertexFrames vf = geometry::vertexFrames(m);   // vf.normals = N [nV,3]
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireFaceAreas();

    const int N = m.nV();
    const int Fn = m.nF();

    // Rectangular real first-order Dirac matrix D : 4F × 4V — maps a quaternionic
    // field on vertices to one on faces. Per face f=ijk, area A: for each cyclic
    // (p,q,r), the block on column p is D_{f,p} = −leftMulImag(N_r − N_q) / (2A).
    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve(static_cast<size_t>(Fn) * 3 * 16);

    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        // Fail loud (CLAUDE.md §6): a single zero-area face would poison D (and
        // hence E) with inf/NaN via the 1/(2A) below — diagnose, don't propagate.
        if (A <= 0.0)
            throw Error(ErrorCode::InvalidInput,
                "dirac::matrix: degenerate (zero-area) face",
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
    D.makeCompressed();
    return D;
}

Eigen::SparseMatrix<double> matrixIntrinsic(Manifold& m) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    // INTRINSIC first-order Dirac D_int : 4F × 4V — identical block structure to
    // matrix(), but the per-block source is the IMMERSION (vertex positions f, i.e.
    // the opposite EDGE VECTOR f_r − f_q) instead of the Gauss map N. This is the
    // spin-connection / tangent-frame-transport root (Crane et al.); its area-
    // weighted Galerkin square Dᵀ ⋆_F D is the intrinsic Dirac² (scalar part = the
    // cotan Laplacian). Geometry-only (vertex positions + face areas).
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireFaceAreas();

    const int N = m.nV();
    const int Fn = m.nF();

    auto vec3 = [](const Vector3& u) { return Eigen::Vector3d(u.x, u.y, u.z); };

    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve(static_cast<size_t>(Fn) * 3 * 16);

    for (Face f : mesh.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        const double A = geom.faceAreas[f];
        if (A <= 0.0)
            throw Error(ErrorCode::InvalidInput,
                "dirac::matrixIntrinsic: degenerate (zero-area) face",
                "Face index " + std::to_string(fi) + "; fix mesh quality first.");
        std::array<int,3> vid{};
        std::array<Eigen::Vector3d,3> pos{};
        int c = 0;
        for (Vertex v : f.adjacentVertices()) {
            vid[c] = static_cast<int>(v.getIndex());
            pos[c] = vec3(geom.inputVertexPositions[v]);
            ++c;
        }
        for (int s = 0; s < 3; ++s) {
            const int p = vid[s];
            const Eigen::Vector3d& Pr = pos[(s + 2) % 3];
            const Eigen::Vector3d& Pq = pos[(s + 1) % 3];
            Eigen::Matrix4d B = (-1.0 / (2.0 * A)) * leftMulImag(Pr - Pq);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (B(a, b) != 0.0)
                        TD.emplace_back(4 * fi + a, 4 * p + b, B(a, b));
        }
    }
    Eigen::SparseMatrix<double> D(4 * Fn, 4 * N);
    D.setFromTriplets(TD.begin(), TD.end());
    D.makeCompressed();
    return D;
}

Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m) {
    using namespace geometrycentral::surface;

    // First-order Dirac D : 4F × 4V — the single source of truth (also exposed
    // directly as operators().diracD()). E is its ⋆_F-weighted Galerkin square,
    // so the two never drift: built from the same triplets.
    Eigen::SparseMatrix<double> D = matrix(m);

    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireFaceAreas();
    const int Fn = m.nF();

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

Eigen::SparseMatrix<double> matrixFace(Manifold& m) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    auto& mesh = m.mesh();
    auto& geom = m.geometry();

    // Closed-mesh v1 (fail loud, CLAUDE.md §6): a boundary vertex has an OPEN star,
    // so its cyclic difference (N_{k+1} − N_{k-1}) would wrap incorrectly across the
    // two boundary ends and silently corrupt the operator. The richer boundary
    // treatment (Liu's infinite potential well) is deferred. Cortical hemispheres
    // are closed topological spheres, so this is exact for the intended use.
    if (mesh.hasBoundary())
        throw Error(ErrorCode::InvalidInput,
            "dirac::matrixFace: open boundary unsupported (diracFace is closed-mesh v1)",
            "Vertex stars must be closed; pass a closed mesh (e.g. a FreeSurfer hemisphere).");

    geom.requireFaceNormals();        // EXACT per-face normals (the Gauss map at faces)
    geom.requireVertexDualAreas();    // ⋆_V measure

    const int N  = m.nV();
    const int Fn = m.nF();

    auto vec3 = [](const Vector3& u) { return Eigen::Vector3d(u.x, u.y, u.z); };

    // Rectangular real first-order face-domain Dirac operator D̃ : ℍ^F → ℍ^V  [4V × 4F].
    // For each vertex v (all interior — closed mesh) with cyclically ordered incident
    // faces f_0..f_{d-1}:  block(v, f_k) = -leftMulImag(N_{f_{k+1}} - N_{f_{k-1}}) / (2 Ã_v).
    std::vector<Eigen::Triplet<double>> TD;
    TD.reserve(static_cast<size_t>(Fn) * 6 * 16);

    for (Vertex v : mesh.vertices()) {
        const int vi = static_cast<int>(v.getIndex());
        const double Av = geom.vertexDualAreas[v];
        if (Av <= 0.0)
            throw Error(ErrorCode::InvalidInput,
                "dirac::matrixFace: degenerate (zero-area) vertex dual cell",
                "Vertex index " + std::to_string(vi) + "; fix mesh quality first.");

        std::vector<int> fid;
        std::vector<Eigen::Vector3d> nrm;
        for (Halfedge he : v.outgoingHalfedges()) {
            Face f = he.face();
            fid.push_back(static_cast<int>(f.getIndex()));
            nrm.push_back(vec3(geom.faceNormals[f]));
        }
        const int d = static_cast<int>(fid.size());
        const double s = -1.0 / (2.0 * Av);
        for (int k = 0; k < d; ++k) {
            Eigen::Vector3d dN = nrm[(k + 1) % d] - nrm[(k - 1 + d) % d];
            Eigen::Matrix4d B = s * leftMulImag(dN);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (B(a, b) != 0.0)
                        TD.emplace_back(4 * vi + a, 4 * fid[k] + b, B(a, b));
        }
    }
    Eigen::SparseMatrix<double> D(4 * N, 4 * Fn);
    D.setFromTriplets(TD.begin(), TD.end());
    D.makeCompressed();
    return D;
}

Eigen::SparseMatrix<double> extrinsicBlockFace(Manifold& m) {
    using namespace geometrycentral::surface;

    // First-order face-domain Dirac D̃ : 4V × 4F — the single source of truth (also
    // exposed as operators().diracFaceD()). It throws on an open boundary, so Ẽ
    // inherits the closed-mesh-v1 guard. Ẽ is its ⋆_V-weighted Galerkin square.
    Eigen::SparseMatrix<double> D = matrixFace(m);

    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireVertexDualAreas();
    const int N = m.nV();

    std::vector<Eigen::Triplet<double>> TW;
    TW.reserve(static_cast<size_t>(4) * N);
    for (Vertex v : mesh.vertices()) {
        const int vi = static_cast<int>(v.getIndex());
        const double Av = geom.vertexDualAreas[v];
        for (int a = 0; a < 4; ++a) TW.emplace_back(4 * vi + a, 4 * vi + a, Av);
    }
    Eigen::SparseMatrix<double> WV(4 * N, 4 * N);
    WV.setFromTriplets(TW.begin(), TW.end());

    // Materialise D.transpose() into its own matrix before the multiply chain:
    // the lazy Transpose proxy shares D's storage, and chaining * WV * D on it
    // would alias. The explicit temporary forces evaluation first.
    Eigen::SparseMatrix<double> E = (Eigen::SparseMatrix<double>(D.transpose()) * WV * D).pruned();
    E.makeCompressed();
    return E;
}

}  // namespace nxr::manifold::ops::dirac
