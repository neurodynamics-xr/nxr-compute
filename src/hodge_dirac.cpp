// Hodge–Dirac d + δ on the intrinsic Delaunay triangulation, and its eigenmodes by the exact/coexact split.
// Reference: nxr-cortical-flow hodgeDirac.ts (+ solve-lb-hodge.py, make-hodge-package.py); floor of ⋆1 as Rheome PR #94.
#include "nxr/compute.h"

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nxr::manifold::ops::hodgedirac {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Mixed-Voronoi lumped mass of a triangulation from its edge lengths (obtuse corner: A/2 and A/4 elsewhere).
// cortical-flow takes it from the INPUT surface (mollified, before the flips): the flips are the stiffness's, and a flipped
// mesh can crush a vertex's area (lumpedMass in intrinsicDelaunay.ts).
static Eigen::VectorXd mixedVoronoiMass(ManifoldSurfaceMesh& mesh, const EdgeData<double>& len) {
    Eigen::VectorXd M = Eigen::VectorXd::Zero(mesh.nVertices());
    for (Face f : mesh.faces()) {
        Halfedge h0 = f.halfedge(), hk[3] = {h0, h0.next(), h0.next().next()};
        double l[3];
        int v[3];
        for (int k = 0; k < 3; ++k) { l[k] = len[hk[k].edge()]; v[k] = static_cast<int>(hk[k].tailVertex().getIndex()); }
        const double s = 0.5 * (l[0] + l[1] + l[2]);
        const double A = std::sqrt(std::max(0.0, s * (s - l[0]) * (s - l[1]) * (s - l[2])));
        double cs[3];   // l[k] runs corner k → k+1; corner k's angle is opposite l[(k+1)%3]
        for (int k = 0; k < 3; ++k) {
            const double a = l[k], b = l[(k + 2) % 3], o = l[(k + 1) % 3];
            cs[k] = (a * a + b * b - o * o) / (2 * a * b);
        }
        int obt = -1;
        for (int k = 0; k < 3; ++k) if (cs[k] < 0) { obt = k; break; }
        if (obt >= 0) {
            for (int k = 0; k < 3; ++k) M(v[k]) += (k == obt) ? A / 2 : A / 4;
        } else {
            double cot[3];
            for (int k = 0; k < 3; ++k) cot[k] = cs[k] / std::sqrt(std::max(1e-300, 1 - cs[k] * cs[k]));
            for (int k = 0; k < 3; ++k) {
                const int e2 = (k + 2) % 3;
                M(v[k]) += (l[k] * l[k] * cot[(k + 2) % 3] + l[e2] * l[e2] * cot[(e2 + 2) % 3]) / 8;
            }
        }
    }
    return M;
}

HodgeDirac assemble(Manifold& m, double star1Floor, double mollify) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();
    geom.requireEdgeLengths();
    EdgeData<double> len = geom.edgeLengths;
    HodgeDirac h;
    if (mollify > 0.0) {
        const double e0 = len[mesh.edge(0)];
        mollifyIntrinsic(mesh, len, mollify);
        h.mollifyEps = len[mesh.edge(0)] - e0;
    }
    h.star0 = mixedVoronoiMass(mesh, len);
    EdgeLengthGeometry eg(mesh, len);
    SignpostIntrinsicTriangulation tri(mesh, eg);
    tri.flipToDelaunay();
    ManifoldSurfaceMesh& im = *tri.intrinsicMesh;
    tri.requireEdgeLengths();
    tri.requireEdgeCotanWeights();
    tri.requireFaceAreas();

    const int nV = static_cast<int>(im.nVertices()), nE = static_cast<int>(im.nEdges()), nF = static_cast<int>(im.nFaces());
    h.nV = nV; h.nE = nE; h.nF = nF;
    h.star1.resize(nE);
    h.star2.resize(nF);

    std::vector<Eigen::Triplet<double>> t0, t1;
    for (Edge e : im.edges()) {
        const int ei = static_cast<int>(e.getIndex());
        Halfedge he = e.halfedge();                       // the edge's orientation: tail → tip
        t0.emplace_back(ei, static_cast<int>(he.tipVertex().getIndex()), 1.0);
        t0.emplace_back(ei, static_cast<int>(he.tailVertex().getIndex()), -1.0);
        double w = tri.edgeCotanWeights[e];
        if (w < star1Floor) { ++h.star1Clamped; w = star1Floor; }
        h.star1(ei) = w;
    }
    for (Face f : im.faces()) {
        const int fi = static_cast<int>(f.getIndex());
        Halfedge h0 = f.halfedge(), hk[3] = {h0, h0.next(), h0.next().next()};
        double l[3], A = tri.faceAreas[f];
        int v[3];
        for (int k = 0; k < 3; ++k) {
            l[k] = tri.edgeLengths[hk[k].edge()];
            v[k] = static_cast<int>(hk[k].tailVertex().getIndex());
            t1.emplace_back(fi, static_cast<int>(hk[k].edge().getIndex()), hk[k] == hk[k].edge().halfedge() ? 1.0 : -1.0);
        }
        h.star2(fi) = 1.0 / A;
    }
    h.d0.resize(nE, nV); h.d0.setFromTriplets(t0.begin(), t0.end());
    h.d1.resize(nF, nE); h.d1.setFromTriplets(t1.begin(), t1.end());
    h.d0.makeCompressed(); h.d1.makeCompressed();

    const int N = nV + nE + nF;
    Eigen::SparseMatrix<double> A = h.star1.asDiagonal() * h.d0;      // ⋆1 d0  [E×V]
    Eigen::SparseMatrix<double> B = h.star2.asDiagonal() * h.d1;      // ⋆2 d1  [F×E]
    std::vector<Eigen::Triplet<double>> tk;
    for (int o = 0; o < A.outerSize(); ++o)
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, o); it; ++it) {
            tk.emplace_back(nV + static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
            tk.emplace_back(static_cast<int>(it.col()), nV + static_cast<int>(it.row()), it.value());
        }
    for (int o = 0; o < B.outerSize(); ++o)
        for (Eigen::SparseMatrix<double>::InnerIterator it(B, o); it; ++it) {
            tk.emplace_back(nV + nE + static_cast<int>(it.row()), nV + static_cast<int>(it.col()), it.value());
            tk.emplace_back(nV + static_cast<int>(it.col()), nV + nE + static_cast<int>(it.row()), it.value());
        }
    h.K.resize(N, N); h.K.setFromTriplets(tk.begin(), tk.end()); h.K.makeCompressed();
    h.M.resize(N);
    h.M << h.star0, h.star1, h.star2;
    return h;
}

Eigen::SparseMatrix<double> massMatrix(const HodgeDirac& h) {
    Eigen::SparseMatrix<double> M(h.M.size(), h.M.size());
    M = h.M.asDiagonal();
    return M;
}

}  // namespace nxr::manifold::ops::hodgedirac

namespace nxr::manifold::solve {

using ops::hodgedirac::HodgeDirac;

Eigen::VectorXi classifyHodgeModes(const HodgeDirac& h, const Eigen::MatrixXd& X, const Eigen::VectorXd& mu, double tol) {
    const int k = static_cast<int>(X.cols());
    Eigen::VectorXi kind(k);
    const double mmax = mu.size() ? mu.cwiseAbs().maxCoeff() : 0.0;
    for (int c = 0; c < k; ++c) {
        double e0 = 0, e2 = 0;
        for (int i = 0; i < h.nV; ++i) e0 += h.M(i) * X(i, c) * X(i, c);
        for (int i = 0; i < h.nF; ++i) e2 += h.M(h.nV + h.nE + i) * X(h.nV + h.nE + i, c) * X(h.nV + h.nE + i, c);
        kind(c) = (std::abs(mu(c)) <= tol * mmax) ? 1 : (e0 >= e2 ? 0 : 2);
    }
    return kind;
}

EigenResult hodgeDiracModes(const HodgeDirac& h, int count, bool bothSigns, int slice,
                            const CancellationToken& cancel, const ProgressObserver& progress) {
    if (count < 1) throw Error(ErrorCode::EigensolveInvalidK, "hodgeDiracModes: count must be >= 1");
    const int nV = h.nV, nE = h.nE, nF = h.nF, N = nV + nE + nF;
    Eigen::SparseMatrix<double> s1(nE, nE), s1inv(nE, nE);
    s1 = h.star1.asDiagonal();
    s1inv = h.star1.cwiseInverse().asDiagonal();
    // Δ0 = d0ᵀ⋆1 d0 (mass ⋆0);  Δ2 = (⋆2d1) ⋆1⁻¹ (⋆2d1)ᵀ (mass ⋆2)
    Eigen::SparseMatrix<double> L0 = h.d0.transpose() * s1 * h.d0;
    Eigen::SparseMatrix<double> B = h.star2.asDiagonal() * h.d1;
    Eigen::SparseMatrix<double> S2 = B * s1inv * B.transpose();
    Eigen::SparseMatrix<double> M0(nV, nV), M2(nF, nF);
    M0 = h.star0.asDiagonal(); M2 = h.star2.asDiagonal();

    struct Cand { double lam; int kind; int col; };
    std::vector<Cand> cand;
    EigenResult r0, r2;
    // count+1 each: one is the constant (λ = 0). Components beyond the first only shorten the list.
    const int k0 = std::min(count + 1, nV - 1), k2 = std::min(count + 1, nF - 1);
    r0 = eigenSliced(L0, M0, k0, slice, true, cancel, progress);
    r2 = eigenSliced(S2, M2, k2, slice, true, cancel, progress);
    const double lmax = std::min(r0.eigenvalues(r0.k - 1), r2.eigenvalues(r2.k - 1));   // both ladders complete up to here
    const double lmaxAll = std::max(r0.eigenvalues(r0.k - 1), r2.eigenvalues(r2.k - 1));
    for (int c = 0; c < r0.k; ++c) if (r0.eigenvalues(c) > 1e-6 * lmaxAll && r0.eigenvalues(c) <= lmax) cand.push_back({r0.eigenvalues(c), 0, c});
    for (int c = 0; c < r2.k; ++c) if (r2.eigenvalues(c) > 1e-6 * lmaxAll && r2.eigenvalues(c) <= lmax) cand.push_back({r2.eigenvalues(c), 2, c});
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.lam < b.lam; });
    if (static_cast<int>(cand.size()) > count) cand.resize(count);
    const int kp = static_cast<int>(cand.size());
    const int kOut = bothSigns ? 2 * kp : kp;

    EigenResult out;
    out.eigenvectors = Eigen::MatrixXd::Zero(N, kOut);
    out.eigenvalues.resize(kOut);
    out.kind.resize(kOut);
    const double r2h = 1.0 / std::sqrt(2.0);
    for (int c = 0; c < kp; ++c) {
        const double mu = std::sqrt(cand[c].lam);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
        if (cand[c].kind == 0) {
            Eigen::VectorXd psi = r0.eigenvectors.col(cand[c].col);
            x.head(nV) = psi * r2h;
            x.segment(nV, nE) = (h.d0 * psi) * (r2h / mu);
        } else {
            Eigen::VectorXd chi = r2.eigenvectors.col(cand[c].col);
            x.tail(nF) = chi * r2h;
            x.segment(nV, nE) = (s1inv * (B.transpose() * chi)) * (r2h / mu);
        }
        out.eigenvectors.col(c) = x;
        out.eigenvalues(c) = mu;
        out.kind(c) = cand[c].kind;
        if (bothSigns) {
            Eigen::VectorXd y = x;
            y.segment(nV, nE) *= -1.0;                     // (ψ, −ω, χ) has eigenvalue −μ
            out.eigenvectors.col(kp + c) = y;
            out.eigenvalues(kp + c) = -mu;
            out.kind(kp + c) = cand[c].kind;
        }
    }
    out.k = kOut; out.nConverged = kOut;
    return out;
}

}  // namespace nxr::manifold::solve
