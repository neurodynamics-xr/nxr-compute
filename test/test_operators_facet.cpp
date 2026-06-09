#include "nxr/facets.h"
#include <complex>
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

static void testLaplacianCotanGraph() {
    std::cout << "\n=== operators: laplacian cotan/graph + independent cache ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto ops = m.operators();

    const auto& L = ops.laplacian().cotan();
    EXPECT(L.rows() == 12 && L.cols() == 12, "cotan Laplacian [12,12]");
    EXPECT(m.isOperatorCached(OperatorId::LaplacianCotan), "cotan cached after request");
    // INDEPENDENT cache: requesting cotan did NOT build mass
    EXPECT(!m.isOperatorCached(OperatorId::MassLumped), "cotan request did not build mass (decoupled)");

    const auto& G = ops.laplacian().graph();
    EXPECT(G.rows() == 12, "graph Laplacian [12,12]");
    // graph Laplacian diagonal = vertex degree (>0)
    EXPECT(G.coeff(0,0) > 0.5, "graph Laplacian diagonal = degree");

    // release drops the slot; re-request rebuilds
    m.releaseOperator(OperatorId::LaplacianCotan);
    EXPECT(!m.isOperatorCached(OperatorId::LaplacianCotan), "release(cotan) clears slot");
    (void)ops.laplacian().cotan();
    EXPECT(m.isOperatorCached(OperatorId::LaplacianCotan), "re-request rebuilds");

    // stored view stays valid after the OperatorsFacet temporary expires
    // (LaplacianView holds Manifold&, not OperatorsFacet&)
    auto view = m.operators().laplacian();    // temporary OperatorsFacet expires here
    EXPECT(view.cotan().rows() == 12, "stored laplacian view is safe (holds Manifold&)");
}

static void testMassDecHodge() {
    std::cout << "\n=== operators: dec/mass/hodge ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    auto ops = m.operators();

    EXPECT(ops.mass().lumped().rows() == 12, "mass.lumped [12,12]");
    EXPECT(ops.mass().lumped().nonZeros() == 12, "mass.lumped is diagonal (12 nnz)");
    EXPECT(ops.mass().galerkin().nonZeros() > 12, "mass.galerkin has off-diagonals");
    // lumped mass sums to total area
    double s = 0; auto Ml = ops.mass().lumped();
    for (int k = 0; k < Ml.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ml,k); it; ++it) s += it.value();
    EXPECT(std::abs(s - m.lightGeometry().totalArea) < 1e-9, "lumped mass sums to total area");

    const auto& dec = ops.dec();
    EXPECT(dec.d0.rows() == 30 && dec.d0.cols() == 12, "dec.d0 [E,V] = [30,12]");
    EXPECT(dec.d1.rows() == 20 && dec.d1.cols() == 30, "dec.d1 [F,E] = [20,30]");
    EXPECT(ops.hodge().h1().rows() == 30, "hodge.h1 [E,E] = [30,30]");
    // requesting mass.lumped only built the lumped slot, not galerkin (independent)
    Manifold m2(V.data(), 12, F.data(), 20);
    (void)m2.operators().mass().lumped();
    EXPECT(m2.isOperatorCached(OperatorId::MassLumped) && !m2.isOperatorCached(OperatorId::MassGalerkin),
           "mass.lumped did not build galerkin (independent)");
}

#include <Eigen/Eigenvalues>

static double minEigC(const Eigen::SparseMatrix<std::complex<double>>& K) {
    Eigen::MatrixXcd Kd(K);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Kd);
    return es.eigenvalues().minCoeff();
}

static void testConnectionCovariant() {
    std::cout << "\n=== operators: connection + covariant (active gauge) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);     // LC active
    auto ops = m.operators();
    namespace cl = ops::laplacian::connection;

    const auto& K = ops.laplacian().connection();
    EXPECT(K.rows() == 12 && K.cols() == 12, "connection L [12,12] complex");
    auto herm = (K - Eigen::SparseMatrix<std::complex<double>>(K.adjoint())).norm();
    EXPECT(herm < 1e-9, "LC connection L Hermitian");
    EXPECT(minEigC(K) > -1e-9, "LC connection L PSD on icosphere (Delaunay)");

    const auto& C = ops.laplacian().covariant(cl::CovariantCoupling::Ambient);
    EXPECT(C.rows() == 36 && C.cols() == 36, "ambient covariant [3V,3V] = [36,36]");
    EXPECT((C - Eigen::SparseMatrix<double>(C.transpose())).norm() < 1e-9, "covariant symmetric");

    // active-gauge sensitivity: trivial gauge yields a different connection L
    std::map<int,double> sing{{0,1.0},{3,1.0}};
    Manifold mt(V.data(), 12, F.data(), 20, sing);    // trivial active
    const auto& Kt = mt.operators().laplacian().connection();
    EXPECT((Kt - K).norm() > 1e-6, "trivial-gauge connection L differs from LC");

    // setGauge invalidates the cached connection/covariant: a single instance
    // rebuilds in the new gauge rather than returning the stale LC matrix.
    Eigen::SparseMatrix<std::complex<double>> Klc = m.operators().laplacian().connection(); // LC, cached
    m.setGauge(GaugeType::Trivial, sing);
    EXPECT(!m.isOperatorCached(OperatorId::LaplacianConnection), "setGauge drops stale connection cache");
    Eigen::SparseMatrix<std::complex<double>> Ktri = m.operators().laplacian().connection(); // rebuilt
    EXPECT((Ktri - Klc).norm() > 1e-6, "connection rebuilt for new gauge after setGauge (not stale)");
}

static void testCovariantCouplingCacheKey() {
    std::cout << "\n=== operators: covariant cache keys on coupling ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    Manifold m(V.data(), 12, F.data(), 20);
    namespace cl = ops::laplacian::connection;
    // Request Product then Ambient on the SAME instance — must NOT alias.
    Eigen::SparseMatrix<double> Cp = m.operators().laplacian().covariant(cl::CovariantCoupling::Product);
    Eigen::SparseMatrix<double> Ca = m.operators().laplacian().covariant(cl::CovariantCoupling::Ambient);
    EXPECT((Cp - Ca).norm() > 1e-6, "product covariant != ambient covariant (cache keys on coupling)");
    // Re-requesting Product returns the Product matrix again (rebuild on mismatch).
    Eigen::SparseMatrix<double> Cp2 = m.operators().laplacian().covariant(cl::CovariantCoupling::Product);
    EXPECT((Cp2 - Cp).norm() < 1e-12, "re-request of product coupling is stable");

    // Ambient is the default coupling (full 3D-frame embedded covariant).
    Manifold md(V.data(), 12, F.data(), 20);
    Eigen::SparseMatrix<double> Cdef = md.operators().laplacian().covariant();   // no arg
    Eigen::SparseMatrix<double> Camb = md.operators().laplacian().covariant(cl::CovariantCoupling::Ambient);
    EXPECT((Cdef - Camb).norm() < 1e-12, "default covariant() == Ambient");

    // Ambient must NOT build the connection Laplacian K (it ignores it) — the
    // intrinsic connection assembly is skipped for the 3D-frame view.
    Manifold ma(V.data(), 12, F.data(), 20);
    (void)ma.operators().laplacian().covariant(cl::CovariantCoupling::Ambient);
    EXPECT(!ma.isOperatorCached(OperatorId::LaplacianConnection),
           "Ambient covariant does not build the connection Laplacian (no wasted K)");
    // Product DOES build K (it puts the intrinsic connection on the tangent block).
    Manifold mp(V.data(), 12, F.data(), 20);
    (void)mp.operators().laplacian().covariant(cl::CovariantCoupling::Product);
    EXPECT(mp.isOperatorCached(OperatorId::LaplacianConnection),
           "Product covariant builds the connection Laplacian (needs K)");
}

static double minEigReal(const Eigen::SparseMatrix<double>& K) {
    Eigen::MatrixXd dense(K);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
    return es.eigenvalues().minCoeff();
}

static void testOperatorsUnderIntrinsicDelaunay() {
    std::cout << "\n=== operators: certified-PSD cotan under intrinsicDelaunay ===\n";
    // Non-Delaunay rhombus (long-diagonal split): raw cotan is indefinite,
    // intrinsic-Delaunay normalization makes the cotan operator PSD.
    std::vector<double>  V = {0,0,0,  2,0,0,  1,0.2,0,  1,-0.2,0};
    std::vector<int32_t> F = {0,2,1,  0,1,3};
    Manifold mN(V.data(), 4, F.data(), 2, /*intrinsicDelaunay=*/true);
    const auto& L = mN.operators().laplacian().cotan();   // sources from operatorGeometry()
    EXPECT(L.rows() == 4, "normalized cotan operator [4,4]");
    EXPECT(minEigReal(L) > -1e-9, "operators().laplacian().cotan() is PSD under intrinsicDelaunay");
}

int main() {
    testLaplacianCotanGraph();
    testMassDecHodge();
    testConnectionCovariant();
    testCovariantCouplingCacheKey();
    testOperatorsUnderIntrinsicDelaunay();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
