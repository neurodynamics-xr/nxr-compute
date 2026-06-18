#include "nxr/compute.h"
#include "nxr/facets.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <Eigen/Eigenvalues>
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

// per-face area mass [3F×3F] and the block rotation R: per face, v ↦ n_f × v.
static void faceMass(Manifold& m, Eigen::SparseMatrix<double>& WF,
                     Eigen::SparseMatrix<double>& R) {
    using namespace geometrycentral;
    using namespace geometrycentral::surface;
    auto& mesh = m.mesh(); auto& geom = m.geometry();
    geom.requireFaceAreas(); geom.requireFaceNormals();
    const int Fn = m.nF();
    std::vector<Eigen::Triplet<double>> tw, tr;
    for (Face f : mesh.faces()) {
        const int fi = (int)f.getIndex();
        const double A = geom.faceAreas[f];
        for (int d = 0; d < 3; ++d) tw.emplace_back(3*fi+d, 3*fi+d, A);
        Vector3 n = geom.faceNormals[f];
        // cross-product matrix [n]_x so that [n]_x v = n × v
        const double nx=n.x, ny=n.y, nz=n.z;
        double Cm[3][3] = {{0,-nz,ny},{nz,0,-nx},{-ny,nx,0}};
        for (int a=0;a<3;++a) for (int b=0;b<3;++b)
            if (Cm[a][b]!=0.0) tr.emplace_back(3*fi+a, 3*fi+b, Cm[a][b]);
    }
    WF.resize(3*Fn,3*Fn); WF.setFromTriplets(tw.begin(),tw.end());
    R.resize(3*Fn,3*Fn);  R.setFromTriplets(tr.begin(),tr.end());
}

static void testGradFace() {
    std::cout << "\n=== facegrad::gradient (barycentric dual gradient) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int Fn = m.nF();   // 20

    Eigen::SparseMatrix<double> G = ops::facegrad::gradient(m);
    EXPECT(G.rows() == 3*Fn && G.cols() == Fn, "gradFace is [3F x F] = [60, 20]");

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(Fn);
    EXPECT((G*ones).norm() < 1e-10, "gradFace annihilates constants");

    // tangency: each per-face 3-vector ⟂ that face's normal
    m.geometry().requireFaceNormals();
    Eigen::VectorXd psi = Eigen::VectorXd::Random(Fn);
    Eigen::VectorXd g = G*psi;
    double maxdot = 0; int fi = 0;
    for (auto f : m.mesh().faces()) {
        auto n = m.geometry().faceNormals[f];
        Eigen::Vector3d gv(g[3*fi], g[3*fi+1], g[3*fi+2]);
        Eigen::Vector3d nn(n.x, n.y, n.z);
        maxdot = std::max(maxdot, std::abs(gv.dot(nn))); ++fi;
    }
    EXPECT(maxdot < 1e-9, "gradFace output is tangent (perp to face normal)");
}

static void testLapFace() {
    std::cout << "\n=== facegrad::laplacian (face Laplacian K̃) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const int Fn = m.nF();

    Eigen::SparseMatrix<double> G = ops::facegrad::gradient(m);
    Eigen::SparseMatrix<double> K = ops::facegrad::laplacian(m);
    EXPECT(K.rows()==Fn && K.cols()==Fn, "lapFace is [F x F] = [20, 20]");
    EXPECT((K - Eigen::SparseMatrix<double>(K.transpose())).norm() < 1e-10, "lapFace symmetric");

    Eigen::SparseMatrix<double> WF, R; faceMass(m, WF, R);
    Eigen::SparseMatrix<double> GtWG = (Eigen::SparseMatrix<double>(G.transpose())*WF*G).pruned();
    EXPECT((K-GtWG).norm() < 1e-9*K.norm(), "lapFace == gradFace' W_F gradFace");

    Eigen::MatrixXd Kd(K);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Kd);
    EXPECT(es.eigenvalues().minCoeff() > -1e-9, "lapFace PSD");
    EXPECT(std::abs(es.eigenvalues()(0)) < 1e-9 && es.eigenvalues()(1) > 1e-9,
           "lapFace kernel is 1-dim (constants)");

    // KEY identity (the spec's shared-Laplacian simplification): SkewG = R·G
    // (rotate each per-face gradient by n_f×) satisfies SkewG' W_F SkewG == K.
    Eigen::SparseMatrix<double> SkewG = (R*G).pruned();
    Eigen::SparseMatrix<double> Ksk = (Eigen::SparseMatrix<double>(SkewG.transpose())*WF*SkewG).pruned();
    EXPECT((K-Ksk).norm() < 1e-9*K.norm(), "SkewG' W_F SkewG == K (n× is an area-isometry)");

    // Exact Poisson round-trip (adjoint): curlS = SkewG' W_F (SkewG psi0) must equal K psi0,
    // so solving K psi = curlS recovers psi0 — proven here as the operator identity on psi0.
    Eigen::VectorXd psi0 = Eigen::VectorXd::Random(Fn); psi0.array() -= psi0.mean();
    Eigen::VectorXd Vsol = SkewG*psi0;
    Eigen::VectorXd curlS = Eigen::SparseMatrix<double>(SkewG.transpose())*(WF*Vsol);
    EXPECT((curlS - K*psi0).norm() < 1e-9*(K*psi0).norm(), "round-trip: SkewG' W_F SkewG psi0 == K psi0");
}

static void testFacetAccessors() {
    std::cout << "\n=== facet accessors gradFace()/lapFace() (cached) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    const Eigen::SparseMatrix<double>& G1 = m.operators().gradFace();
    const Eigen::SparseMatrix<double>& G2 = m.operators().gradFace();
    EXPECT(&G1 == &G2, "gradFace() returns the same cached object");
    EXPECT(m.operators().lapFace().rows() == m.nF(), "lapFace() cached, [F x F]");
}

int main() {
    testGradFace();
    testLapFace();
    testFacetAccessors();
    if (g_failures) { std::cout << "\n" << g_failures << " FAILURE(S)\n"; return 1; }
    std::cout << "\nALL PASS\n"; return 0;
}
