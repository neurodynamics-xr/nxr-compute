#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <cmath>
#include <iostream>
using namespace nxr::manifold;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testFrameTransport() {
    std::cout << "\n=== covariant-differential: frame transport ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    // orthogonality: P_ij^T P_ij = I (P is a product of rotations)
    Eigen::Matrix3d P = differential::frameTransport(m, 0, 3);
    EXPECT((P.transpose() * P - Eigen::Matrix3d::Identity()).norm() < 1e-12, "P_ij orthogonal");

    // flatness / path-independence: around face {0,11,5}, P_k i · P_j k · P_i j = I
    int i = 0, j = 11, k = 5;
    Eigen::Matrix3d loop = differential::frameTransport(m, k, i)
                         * differential::frameTransport(m, j, k)
                         * differential::frameTransport(m, i, j);
    EXPECT((loop - Eigen::Matrix3d::Identity()).norm() < 1e-12, "holonomy around triangle is identity (flat)");

    // self-transport is identity
    EXPECT((differential::frameTransport(m, 4, 4) - Eigen::Matrix3d::Identity()).norm() < 1e-12,
           "P_ii = I");

    // vertexFrameMatrices: [nV,9], each row an orthonormal frame
    Eigen::MatrixXd Fm = differential::vertexFrameMatrices(m);
    EXPECT(Fm.rows() == 12 && Fm.cols() == 9, "vertexFrameMatrices [12,9]");
    Eigen::Matrix3d F0;  // row 0 reshaped row-major
    F0 << Fm(0,0),Fm(0,1),Fm(0,2), Fm(0,3),Fm(0,4),Fm(0,5), Fm(0,6),Fm(0,7),Fm(0,8);
    EXPECT((F0.transpose()*F0 - Eigen::Matrix3d::Identity()).norm() < 1e-12, "Fv orthonormal");
}

static void testLifts() {
    std::cout << "\n=== covariant-differential: lifts ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);

    // round-trip: liftToFrame(liftToWorld(L)) == L
    Eigen::MatrixXd Lloc = Eigen::MatrixXd::Random(12, 3);
    Eigen::MatrixXd back = differential::liftToFrame(m, differential::liftToWorld(m, Lloc));
    EXPECT((back - Lloc).cwiseAbs().maxCoeff() < 1e-12, "liftToFrame∘liftToWorld == identity");

    // artifact removal end-to-end: a Cartesian-CONSTANT field lifted to frames has
    // DIFFERENT local coords at different vertices, but the SAME world vector everywhere.
    Eigen::MatrixXd Lworld(12, 3);
    for (int v = 0; v < 12; ++v) Lworld.row(v) = Eigen::RowVector3d(1.0, 0.0, 0.0);
    Eigen::MatrixXd locC = differential::liftToFrame(m, Lworld);
    Eigen::MatrixXd worldBack = differential::liftToWorld(m, locC);
    EXPECT((worldBack - Lworld).cwiseAbs().maxCoeff() < 1e-12, "lift recovers constant world field");
    // local coords genuinely differ between two vertices on the curved surface
    EXPECT((locC.row(0) - locC.row(6)).cwiseAbs().maxCoeff() > 1e-3,
           "same Cartesian vector has different local coords (the artifact lifts away)");
}

static void testCovariantGradient() {
    std::cout << "\n=== covariant-differential: covariant gradient G ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int N = m.nV(), E = m.nE();   // 12, 30

    Eigen::SparseMatrix<double> G = differential::covariantGradient(m);
    EXPECT(G.rows() == 3*E && G.cols() == 3*N, "G is [3E, 3N] = [90, 36]");

    // HEADLINE: a Cartesian-constant field has zero covariant gradient.
    Eigen::MatrixXd Lworld(N, 3);
    for (int v = 0; v < N; ++v) Lworld.row(v) = Eigen::RowVector3d(0.3, -0.7, 0.2);
    Eigen::MatrixXd Lloc = differential::liftToFrame(m, Lworld);     // [N,3]
    // component-major 3N: column-major [N,3] flattens to [a;b;c]
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(Lloc.data(), 3*N);
    EXPECT((G * x).cwiseAbs().maxCoeff() < 1e-10, "G*(Cartesian-constant) = 0 (artifact removed)");
    // ...while the naive component-wise difference is NOT zero on this curved mesh:
    EXPECT((Lloc.row(0) - Lloc.row(2)).cwiseAbs().maxCoeff() > 1e-3, "naive local difference is nonzero");

    // CONSISTENCY: G^T W G == the existing Ambient covariant Laplacian (default LC gauge).
    // W is the edge cotan weight, replicated across the 3 component blocks.
    auto& geom = m.operatorGeometry(); geom.requireEdgeCotanWeights();
    Eigen::VectorXd wEdge(E);
    for (auto e : m.mesh().edges()) wEdge(e.getIndex()) = geom.edgeCotanWeights[e];
    Eigen::VectorXd wDiag(3*E);
    for (int p = 0; p < 3; ++p) wDiag.segment(p*E, E) = wEdge;
    Eigen::SparseMatrix<double> W(3*E, 3*E);
    { std::vector<Eigen::Triplet<double>> tw; tw.reserve(3*E);
      for (int k = 0; k < 3*E; ++k) tw.emplace_back(k, k, wDiag(k));
      W.setFromTriplets(tw.begin(), tw.end()); }
    Eigen::SparseMatrix<double> GtWG = G.transpose() * W * G;

    namespace cl = ops::laplacian::connection;
    Eigen::SparseMatrix<double> Camb =
        m.operators().laplacian().covariant(cl::CovariantCoupling::Ambient);
    EXPECT((Eigen::MatrixXd(GtWG) - Eigen::MatrixXd(Camb)).cwiseAbs().maxCoeff() < 1e-9,
           "G^T W G == Ambient covariant Laplacian (consistency)");

    // gradient3D is cached on the handle via the operators facet (matches the other operators)
    EXPECT((m.operators().gradient3D() - G).norm() < 1e-12, "operators().gradient3D() == covariantGradient");
    EXPECT(m.isOperatorCached(OperatorId::Gradient3D), "gradient3D cached after request");
    m.releaseOperator(OperatorId::Gradient3D);
    EXPECT(!m.isOperatorCached(OperatorId::Gradient3D), "releaseOperator(Gradient3D) clears the cache");
}

// The motivating case: two frames with OPPOSED normals (opposite sulcal walls). A vector
// that is the SAME in Cartesian space transports correctly between them — the antiparallel
// reading is a frame artifact that frameTransport removes.
static void testSulcalWallTransport() {
    std::cout << "\n=== covariant-differential: sulcal-wall transport ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    Eigen::MatrixXd Fm = differential::vertexFrameMatrices(m);
    auto frame = [&](int v) {
        Eigen::Matrix3d Fv;
        Fv << Fm(v,0),Fm(v,1),Fm(v,2), Fm(v,3),Fm(v,4),Fm(v,5), Fm(v,6),Fm(v,7),Fm(v,8);
        return Fv;
    };
    // icosphere vertices 0 and 3 are nearly antipodal ⇒ opposed normals (the sulcal-wall case)
    Eigen::Matrix3d F0 = frame(0), F3 = frame(3);
    EXPECT(F0.col(2).dot(F3.col(2)) < -0.5, "vertices 0,3 have opposed normals (sulcal-wall fixture)");
    // a single Cartesian vector, expressed in each frame, is related EXACTLY by frameTransport
    Eigen::Vector3d w(0.2, 0.5, -0.3);
    Eigen::Vector3d loc0 = F0.transpose() * w;                              // w in frame 0
    Eigen::Vector3d loc3 = F3.transpose() * w;                              // w in frame 3
    Eigen::Vector3d moved = differential::frameTransport(m, 0, 3) * loc0;   // transport 0 -> 3
    EXPECT((moved - loc3).cwiseAbs().maxCoeff() < 1e-12,
           "frameTransport maps the shared Cartesian vector between opposed-normal frames");
}

int main() {
    testFrameTransport();
    testLifts();
    testCovariantGradient();
    testSulcalWallTransport();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
